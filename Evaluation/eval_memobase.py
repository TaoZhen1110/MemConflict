import argparse
import copy
import json
import os
import sys
import time
import traceback
import re
import inspect
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import ipaddress

from dotenv import load_dotenv

try:
    import jsonlines
except ImportError:
    jsonlines = None

try:
    from memobase import MemoBaseClient, ChatBlob
    import memobase.error
except ImportError:
    MemoBaseClient = None
    ChatBlob = None
    memobase = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from llm_request import llm_request, calculate_cumulative_cost
except Exception:
    llm_request = None

    def calculate_cumulative_cost(previous_cost: Optional[Dict], current_cost: Dict) -> Dict:
        if not isinstance(previous_cost, dict):
            return copy.deepcopy(current_cost)
        merged = copy.deepcopy(previous_cost)
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            merged[key] = (merged.get(key, 0) or 0) + (current_cost.get(key, 0) or 0)
        merged["total_cost_usd"] = (merged.get("total_cost_usd", 0.0) or 0.0) + (
            current_cost.get("total_cost_usd", 0.0) or 0.0
        )
        if merged.get("model") is None:
            merged["model"] = current_cost.get("model")
        merged["pricing_available"] = bool(
            merged.get("pricing_available") or current_cost.get("pricing_available")
        )
        return merged

load_dotenv()

MEMOBASE_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

MEMOBASE_ADD_MAX_RETRIES = 3
MEMOBASE_ADD_RETRY_SECONDS = 5
MEMOBASE_SEARCH_MAX_RETRIES = 3
MEMOBASE_SEARCH_RETRY_SECONDS = 3
MEMOBASE_POST_ADD_BUFFER_SECONDS = 15
MEMOBASE_RETRIEVAL_READY_MAX_RETRIES = 30
MEMOBASE_RETRIEVAL_READY_POLL_INTERVAL_SECONDS = 3
MAX_STORED_RETRIEVED_MEMORIES = 5
MEMOBASE_CONTEXT_MIN_TOKENS = 800
MEMOBASE_CONTEXT_TOKENS_PER_MEMORY = 200
MEMOBASE_CONTEXT_PROFILE_EVENT_RATIO = 0.75
MEMOBASE_CONTEXT_TIME_RANGE_DAYS = 3650
MEMOBASE_CONTEXT_EVENT_SIMILARITY_THRESHOLD = 0.2
MEMOBASE_RERANK_CANDIDATE_MULTIPLIER = 3
MEMOBASE_RERANK_MIN_CANDIDATES = 8


def Build_Observable_Token_Cost_Summary(stage_cost: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    return {
        "Stage_Name": stage_name,
        "Input_Tokens": stage_cost.get("input_tokens", 0) or 0,
        "Output_Tokens": stage_cost.get("output_tokens", 0) or 0,
        "Total_Tokens": stage_cost.get("total_tokens", 0) or 0,
        "Total_Cost_USD": stage_cost.get("total_cost_usd", 0.0) or 0.0,
        "Model": stage_cost.get("model"),
        "Pricing_Available": bool(stage_cost.get("pricing_available")),
    }


def load_jsonl_items(input_file: str) -> List[Dict[str, Any]]:
    items = []
    if jsonlines is not None:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                items.append(item)
        return items
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl_items(output_file: str, items: List[Dict[str, Any]]):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if jsonlines is not None:
        with jsonlines.open(output_file, "w") as writer:
            for item in items:
                writer.write(item)
        return
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_dialogue_turn_order(key_name: str) -> int:
    try:
        return int(str(key_name).split("_")[-1])
    except Exception:
        return 10**9


def Build_Session_Dialogue_List(session_dialogue: Any) -> List[Dict[str, Any]]:
    if not isinstance(session_dialogue, dict):
        return []
    flattened_dialogue = []
    ordered_keys = sorted(session_dialogue.keys(), key=extract_dialogue_turn_order)
    for turn_key in ordered_keys:
        turn_value = session_dialogue.get(turn_key, [])
        if not isinstance(turn_value, list):
            continue
        for message_item in turn_value:
            if not isinstance(message_item, dict):
                continue
            role = message_item.get("role")
            content = message_item.get("content")
            if role in ["user", "assistant"] and content not in [None, ""]:
                flattened_item = {"role": role, "content": str(content)}
                if message_item.get("timestamp"):
                    flattened_item["timestamp"] = str(message_item.get("timestamp"))
                elif message_item.get("created_at"):
                    flattened_item["timestamp"] = str(message_item.get("created_at"))
                flattened_dialogue.append(flattened_item)
    return flattened_dialogue


def Build_Memobase_User_ID(persona_item: Dict[str, Any], version: str) -> str:
    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown_persona")
    return f"{persona_id}_{version}_{time.time_ns()}_{uuid.uuid4().hex[:12]}"


def Build_Retrieval_Probe_Queries(dialogue_messages: List[Dict[str, Any]], session_questions: List[Dict[str, Any]]) -> List[str]:
    queries = []
    for question_item in session_questions[:2]:
        question_text = str(question_item.get("question", "")).strip()
        if question_text and question_text not in queries:
            queries.append(question_text)
    for message_item in reversed(dialogue_messages):
        content = str(message_item.get("content", "")).strip()
        if content and content not in queries:
            queries.append(content)
        if message_item.get("role") == "user" and content:
            break
    return queries[:3]


def Detect_Memobase_Backend_Mode(project_url: str) -> str:
    hostname = (urlparse(project_url).hostname or "").strip().lower()
    if not hostname:
        return "unknown"
    if hostname in ["localhost", "127.0.0.1", "::1"]:
        return "self_hosted"
    if hostname.endswith("memobase.io"):
        return "cloud"
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if ip_obj.is_private or ip_obj.is_loopback:
            return "self_hosted"
        return "self_hosted"
    except ValueError:
        return "self_hosted"


def Build_Memobase_Backend_Config() -> Dict[str, str]:
    project_url = os.getenv("MEMOBASE_PROJECT_URL")
    project_token = os.getenv("MEMOBASE_PROJECT_TOKEN")
    if not project_url:
        raise ValueError("MEMOBASE_PROJECT_URL is not set in the environment.")
    backend_mode = Detect_Memobase_Backend_Mode(project_url)
    if not project_token:
        if backend_mode == "self_hosted":
            project_token = "secret"
            print("[DEBUG] MEMOBASE_PROJECT_TOKEN is not set. Falling back to default self-hosted token 'secret'.")
        else:
            raise ValueError("MEMOBASE_PROJECT_TOKEN is not set in the environment.")
    hostname = (urlparse(project_url).hostname or "").strip().lower()
    return {
        "project_url": project_url,
        "project_token": project_token,
        "backend_mode": backend_mode,
        "backend_host": hostname or "unknown",
    }


def Setup_Memobase_Client() -> Tuple[Any, Dict[str, str]]:
    if MemoBaseClient is None or ChatBlob is None:
        raise ImportError("memobase package is not available. Please install memobase first.")
    backend_config = Build_Memobase_Backend_Config()
    client = MemoBaseClient(project_url=backend_config["project_url"], api_key=backend_config["project_token"])
    return client, backend_config


def Resolve_Memobase_User_ID(client: Any, requested_name: str) -> str:
    try:
        add_result = client.add_user({"name": requested_name})
    except Exception as e:
        raise RuntimeError(f"Memobase add_user failed for requested_name={requested_name}: {e}") from e

    if isinstance(add_result, str) and add_result.strip():
        return add_result.strip()

    if isinstance(add_result, dict):
        for key in ["id", "user_id", "uuid", "name"]:
            value = add_result.get(key)
            if value:
                return str(value)
        nested_data = add_result.get("data")
        if isinstance(nested_data, dict):
            for key in ["id", "user_id", "uuid", "name"]:
                value = nested_data.get(key)
                if value:
                    return str(value)

    for attr in ["id", "user_id", "uuid", "name"]:
        value = getattr(add_result, attr, None)
        if value:
            return str(value)

    nested_data = getattr(add_result, "data", None)
    if isinstance(nested_data, dict):
        for key in ["id", "user_id", "uuid", "name"]:
            value = nested_data.get(key)
            if value:
                return str(value)
    elif nested_data is not None:
        for attr in ["id", "user_id", "uuid", "name"]:
            value = getattr(nested_data, attr, None)
            if value:
                return str(value)

    raise RuntimeError(
        "Memobase add_user returned an unsupported result; cannot resolve a usable user id. "
        f"Raw result type={type(add_result).__name__}, value={add_result}"
    )


def Build_Memobase_Timestamp(session_date: Optional[str] = None, message_index: int = 0) -> str:
    if session_date:
        try:
            base_dt = datetime.strptime(str(session_date), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return (base_dt + timedelta(seconds=message_index)).isoformat()
        except Exception:
            pass
    return (datetime.now(timezone.utc) + timedelta(seconds=message_index)).isoformat()


def Add_Session_Dialogue_To_Memobase(client: Any, user_id: str, dialogue_messages: List[Dict[str, Any]], session_date: Optional[str] = None) -> Tuple[float, List[Dict[str, Any]]]:
    if len(dialogue_messages) == 0:
        return 0.0, []
    messages = []
    for message_index, item in enumerate(dialogue_messages):
        created_at = (
            item.get("timestamp")
            or item.get("created_at")
            or Build_Memobase_Timestamp(session_date=session_date, message_index=message_index)
        )
        messages.append({
            "role": item.get("role"),
            "content": item.get("content"),
            "created_at": str(created_at),
        })
    duration_ms = 0.0
    add_exception = None
    add_attempt_count = 0
    for attempt_idx in range(MEMOBASE_ADD_MAX_RETRIES):
        add_attempt_count = attempt_idx + 1
        start_time = time.time()
        try:
            user_obj = client.get_user(user_id)
            user_obj.insert(ChatBlob(messages=messages))
            user_obj.flush(sync=True)
            duration_ms += (time.time() - start_time) * 1000
            add_exception = None
            break
        except Exception as e:
            duration_ms += (time.time() - start_time) * 1000
            add_exception = e
            print(f"[DEBUG] Memobase add attempt {attempt_idx + 1}/{MEMOBASE_ADD_MAX_RETRIES} failed: {e}")
            if attempt_idx < MEMOBASE_ADD_MAX_RETRIES - 1:
                time.sleep(MEMOBASE_ADD_RETRY_SECONDS)
    if add_exception is not None:
        raise add_exception
    return duration_ms, [{
        "Batch_Index": 1,
        "Batch_Size": len(messages),
        "Start_Message_Index": 0,
        "End_Message_Index": len(messages) - 1,
        "Add_Attempt_Count": add_attempt_count,
        "Add_Duration_ms": duration_ms,
    }]


def Parse_Memobase_Context_To_Retrieved_Memories(context: str, top_k: int) -> List[Dict[str, Any]]:
    retrieved_memories = []
    current_section = "context"
    seen_memory_texts = set()

    for raw_line in str(context).split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue

        lower_line = stripped.lower()
        if stripped.startswith("# Memory"):
            continue
        if lower_line.startswith("## user background"):
            current_section = "profile"
            continue
        if lower_line.startswith("## latest events"):
            current_section = "event"
            continue
        if not stripped.startswith("- "):
            continue

        memory_text = stripped[2:].strip()
        if not memory_text:
            continue
        if memory_text in {"---", "--", "-", "___", "***"}:
            continue
        if memory_text in seen_memory_texts:
            continue
        seen_memory_texts.add(memory_text)

        retrieved_memories.append(
            {
                "memory": memory_text,
                "source": current_section,
                "rank": len(retrieved_memories) + 1,
            }
        )
        if len(retrieved_memories) >= top_k:
            break

    return retrieved_memories


def Clean_Memobase_Memory_Text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("- "):
        text = text[2:].strip()
    if text in {"---", "--", "-", "___", "***", "None", "null"}:
        return ""
    return text


def Extract_Memobase_Text_Candidates(payload: Any, candidate_keys: List[str]) -> List[str]:
    collected: List[str] = []
    seen_object_ids = set()
    candidate_key_set = set(candidate_keys)

    def append_text(value: Any):
        cleaned = Clean_Memobase_Memory_Text(value)
        if cleaned:
            collected.append(cleaned)

    def walk(obj: Any):
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen_object_ids:
            return
        seen_object_ids.add(obj_id)

        if isinstance(obj, str):
            if "\n" in obj:
                for line in obj.split("\n"):
                    append_text(line)
            else:
                append_text(obj)
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in candidate_key_set and not isinstance(value, (dict, list, tuple, set)):
                    append_text(value)
            for key in ["data", "results", "items", "profiles", "events", "event_gists", "gists", "profile_delta"]:
                if key in obj:
                    walk(obj.get(key))
            for value in obj.values():
                if isinstance(value, (dict, list, tuple, set)):
                    walk(value)
            return

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                walk(item)
            return

        for attr in candidate_key_set:
            value = getattr(obj, attr, None)
            if value is not None and not isinstance(value, (dict, list, tuple, set)):
                append_text(value)
        for attr in ["data", "results", "items", "profiles", "events", "event_gists", "gists"]:
            value = getattr(obj, attr, None)
            if value is not None:
                walk(value)

    walk(payload)
    return collected


def Build_Memobase_Retrieved_Memories(memory_texts: List[str], source: str, top_k: int) -> List[Dict[str, Any]]:
    retrieved_memories: List[Dict[str, Any]] = []
    seen_memory_texts = set()
    for text in memory_texts:
        cleaned = Clean_Memobase_Memory_Text(text)
        if not cleaned or cleaned in seen_memory_texts:
            continue
        seen_memory_texts.add(cleaned)
        retrieved_memories.append({
            "memory": cleaned,
            "source": source,
            "rank": len(retrieved_memories) + 1,
        })
        if len(retrieved_memories) >= top_k:
            break
    return retrieved_memories


def Filter_Memobase_Method_Kwargs(method: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        supported_params = set(inspect.signature(method).parameters.keys())
        return {key: value for key, value in kwargs.items() if key in supported_params}
    except Exception:
        return kwargs


def Call_Memobase_Profile(user_obj: Any, query: str, top_k: int) -> Any:
    profile_method = getattr(user_obj, "profile", None)
    if not callable(profile_method):
        return None

    kwargs_candidates = [
        {"chats": [{"role": "user", "content": str(query)}], "need_json": True, "topk": top_k},
        {"chats": [{"role": "user", "content": str(query)}], "need_json": True, "top_k": top_k},
        {"chats_str": str(query), "need_json": True, "topk": top_k},
        {"chats_str": str(query), "need_json": True, "top_k": top_k},
        {"chats": [{"role": "user", "content": str(query)}], "topk": top_k},
        {"chats": [{"role": "user", "content": str(query)}], "top_k": top_k},
        {"chats_str": str(query), "topk": top_k},
        {"chats_str": str(query), "top_k": top_k},
    ]

    last_exception = None
    for raw_kwargs in kwargs_candidates:
        kwargs = Filter_Memobase_Method_Kwargs(profile_method, raw_kwargs)
        if not kwargs:
            continue
        try:
            return profile_method(**kwargs)
        except TypeError as e:
            last_exception = e
            continue
        except Exception as e:
            print(f"[DEBUG] Memobase profile retrieval failed: {e}")
            return None

    if last_exception is not None:
        print(f"[DEBUG] Memobase profile compatibility fallback exhausted: {last_exception}")
    return None


def Call_Memobase_Event_Search(user_obj: Any, query: str, top_k: int) -> Any:
    method_names = ["search_event_gists", "search_event_gist", "event", "events"]
    kwargs_candidates = [
        {"query": str(query), "topk": top_k, "need_json": True},
        {"query": str(query), "top_k": top_k, "need_json": True},
        {"query": str(query), "topk": top_k},
        {"query": str(query), "top_k": top_k},
    ]

    for method_name in method_names:
        method = getattr(user_obj, method_name, None)
        if not callable(method):
            continue

        last_exception = None
        for raw_kwargs in kwargs_candidates:
            kwargs = Filter_Memobase_Method_Kwargs(method, raw_kwargs)
            try:
                if kwargs:
                    return method(**kwargs)
                return method(str(query))
            except TypeError as e:
                last_exception = e
                continue
            except Exception as e:
                print(f"[DEBUG] Memobase {method_name} retrieval failed: {e}")
                break

        try:
            return method(str(query), topk=top_k)
        except TypeError as e:
            last_exception = e
        except Exception as e:
            print(f"[DEBUG] Memobase {method_name} positional retrieval failed: {e}")

        if last_exception is not None:
            print(f"[DEBUG] Memobase {method_name} compatibility fallback exhausted: {last_exception}")

    return None


def Is_Memobase_Change_Focused_Query(query: str) -> bool:
    lowered = str(query or "").strip().lower()
    if not lowered:
        return False
    change_markers = [
        "change",
        "changed",
        "recently",
        "remain the same",
        "remained the same",
        "stayed the same",
        "stay the same",
        "switch",
        "switched",
        "from and to",
        "what changed",
        "how did",
        "how has",
        "did the user's",
        "has the user's",
    ]
    return any(marker in lowered for marker in change_markers)


def Tokenize_Memobase_Text(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
        "did", "does", "do", "has", "have", "had", "how", "what", "which", "user", "their", "them", "they",
        "about", "from", "into", "that", "this", "these", "those", "it", "its", "as", "at", "by", "be",
        "recent", "recently", "same", "remain", "remained", "stayed", "stay"
    }
    return [token for token in tokens if token and token not in stopwords]


def Infer_Memobase_Query_Topical_Keywords(query: str) -> set:
    lowered = str(query or "").lower()
    topic_groups = {
        "residence": {"residence", "live", "lives", "living", "city", "state", "home", "address", "move", "moved", "relocate", "relocated"},
        "marital": {"marital", "married", "divorced", "divorce", "spouse", "partner", "relationship", "single"},
        "career": {"career", "company", "job", "work", "employment", "industry", "title", "role", "internship", "profession", "employer"},
        "education": {"university", "college", "school", "degree", "major", "study", "studied", "education"},
        "health": {"health", "medical", "diagnosis", "symptom", "sick", "ill", "injury", "therapy", "stress", "fatigued"},
        "social": {"social", "friend", "friends", "family", "network", "relationship", "community"},
    }
    matched = set()
    for words in topic_groups.values():
        if any(word in lowered for word in words):
            matched.update(words)
    return matched


def Score_Memobase_Memory(query: str, memory_text: str, source: str, original_rank: int) -> float:
    query_tokens = set(Tokenize_Memobase_Text(query))
    memory_tokens = set(Tokenize_Memobase_Text(memory_text))
    overlap = len(query_tokens & memory_tokens)
    topical_overlap = len(Infer_Memobase_Query_Topical_Keywords(query) & memory_tokens)
    change_focused = Is_Memobase_Change_Focused_Query(query)

    score = 0.0
    score += overlap * 4.0
    score += topical_overlap * 2.5

    lowered_memory = str(memory_text or "").lower()
    if change_focused:
        if any(marker in lowered_memory for marker in ["changed", "started", "moved", "switched", "divorced", "married", "new", "previous", "former"]):
            score += 2.0
        if source == "event":
            score += 1.5
    else:
        if source == "profile":
            score += 1.0

    score += max(0.0, 1.0 - (original_rank - 1) * 0.1)
    return score


def Rerank_Memobase_Profile_And_Event(query: str,
                                      profile_memories: List[Dict[str, Any]],
                                      event_memories: List[Dict[str, Any]],
                                      top_k: int) -> List[Dict[str, Any]]:
    merged_candidates: List[Dict[str, Any]] = []
    seen_memory_texts = set()
    for item in list(profile_memories) + list(event_memories):
        if not isinstance(item, dict):
            continue
        memory_text = Clean_Memobase_Memory_Text(item.get("memory"))
        if not memory_text or memory_text in seen_memory_texts:
            continue
        seen_memory_texts.add(memory_text)
        merged_candidates.append({
            "memory": memory_text,
            "source": item.get("source", "context"),
            "original_rank": int(item.get("rank", len(merged_candidates) + 1) or (len(merged_candidates) + 1)),
        })

    ranked = sorted(
        merged_candidates,
        key=lambda item: (
            -Score_Memobase_Memory(query, item.get("memory", ""), item.get("source", "context"), item.get("original_rank", 1)),
            item.get("original_rank", 1),
        ),
    )

    result: List[Dict[str, Any]] = []
    for item in ranked[:top_k]:
        result.append({
            "memory": item.get("memory", ""),
            "source": item.get("source", "context"),
            "rank": len(result) + 1,
        })
    return result


def Retrieve_Memobase_Profile_And_Event(user_obj: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
    candidate_top_k = max(top_k * MEMOBASE_RERANK_CANDIDATE_MULTIPLIER, MEMOBASE_RERANK_MIN_CANDIDATES)

    profile_payload = Call_Memobase_Profile(user_obj, query, candidate_top_k)
    profile_texts = Extract_Memobase_Text_Candidates(
        profile_payload,
        ["content", "memory", "text", "value", "summary", "profile", "description", "preference", "preference_note"],
    )
    profile_memories = Build_Memobase_Retrieved_Memories(profile_texts, "profile", candidate_top_k)

    event_payload = Call_Memobase_Event_Search(user_obj, query, candidate_top_k)
    event_texts = Extract_Memobase_Text_Candidates(
        event_payload,
        ["gist", "event_tip", "content", "memory", "text", "value", "summary", "event_data"],
    )
    event_memories = Build_Memobase_Retrieved_Memories(event_texts, "event", candidate_top_k)

    return Rerank_Memobase_Profile_And_Event(query, profile_memories, event_memories, top_k)



def Call_Memobase_Context(user_obj: Any, query: str, top_k: int) -> str:
    max_context_tokens = max(MEMOBASE_CONTEXT_MIN_TOKENS, top_k * MEMOBASE_CONTEXT_TOKENS_PER_MEMORY)
    primary_kwargs = {
        "max_token_size": max_context_tokens,
        "chats": [{"role": "user", "content": str(query)}],
        "event_similarity_threshold": MEMOBASE_CONTEXT_EVENT_SIMILARITY_THRESHOLD,
        "fill_window_with_events": True,
    }

    try:
        return user_obj.context(**primary_kwargs)
    except TypeError as e:
        print(f"[DEBUG] Memobase context compatibility fallback triggered: {e}")
        fallback_kwargs = {
            "max_tokens": max_context_tokens,
            "chats": [{"role": "user", "content": str(query)}],
            "event_similarity_threshold": MEMOBASE_CONTEXT_EVENT_SIMILARITY_THRESHOLD,
            "fill_window_with_events": True,
        }
        return user_obj.context(**fallback_kwargs)


def Search_Memobase(client: Any, user_id: str, query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    duration_ms = 0.0
    search_exception = None
    context = ""
    retrieved_memories: List[Dict[str, Any]] = []

    for attempt_idx in range(MEMOBASE_SEARCH_MAX_RETRIES):
        start_time = time.time()
        try:
            user_obj = client.get_user(user_id)
            retrieved_memories = Retrieve_Memobase_Profile_And_Event(user_obj, query, top_k)
            if len(retrieved_memories) == 0:
                context = Call_Memobase_Context(user_obj, query, top_k)
                retrieved_memories = Parse_Memobase_Context_To_Retrieved_Memories(context, top_k)
            duration_ms += (time.time() - start_time) * 1000
            search_exception = None
            break
        except Exception as e:
            duration_ms += (time.time() - start_time) * 1000
            search_exception = e
            print(f"[DEBUG] Memobase search attempt {attempt_idx + 1}/{MEMOBASE_SEARCH_MAX_RETRIES} failed: {e}")
            if attempt_idx < MEMOBASE_SEARCH_MAX_RETRIES - 1:
                time.sleep(MEMOBASE_SEARCH_RETRY_SECONDS)
    if search_exception is not None:
        raise search_exception
    lines = [f"Memories for user {user_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for item in retrieved_memories:
            lines.append(f"{item.get('rank')}. {item.get('memory')} [{item.get('source')}]")
    context_text = "\n".join(lines)
    return context_text, retrieved_memories, duration_ms


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], user_id: str) -> str:
    lines = [f"Memories for user {user_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for item in retrieved_memories:
            lines.append(f"{item.get('rank')}. {item.get('memory')} [{item.get('source')}]")
    return "\n".join(lines)


def Wait_For_Memobase_Retrieval_Ready(
    client: Any,
    user_id: str,
    probe_queries: List[str],
    top_k: int,
    max_retries: int = MEMOBASE_RETRIEVAL_READY_MAX_RETRIES,
    poll_interval_seconds: int = MEMOBASE_RETRIEVAL_READY_POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    if len(probe_queries) == 0:
        return {"status": "SKIPPED", "attempt_count": 0, "probe_queries": [], "matched_query": None, "matched_count": 0}
    last_probe_result = {"status": "PENDING", "matched_query": None, "matched_count": 0}
    for attempt_idx in range(max_retries):
        for probe_query in probe_queries:
            try:
                _, retrieved_memories, _ = Search_Memobase(client, user_id, probe_query, max(1, min(top_k, 3)))
                if len(retrieved_memories) > 0:
                    return {
                        "status": "READY",
                        "attempt_count": attempt_idx + 1,
                        "probe_queries": probe_queries,
                        "matched_query": probe_query,
                        "matched_count": len(retrieved_memories),
                    }
                last_probe_result = {"status": "WAITING", "matched_query": probe_query, "matched_count": 0}
            except Exception as e:
                last_probe_result = {"status": "SEARCH_ERROR", "matched_query": probe_query, "matched_count": 0, "error": str(e)}
        print(f"[DEBUG] Memobase retrieval readiness poll {attempt_idx + 1}/{max_retries}: status={last_probe_result.get('status')}")
        time.sleep(poll_interval_seconds)
    last_probe_result.update({"status": "TIMEOUT", "attempt_count": max_retries, "probe_queries": probe_queries})
    return last_probe_result


def Generate_Answer_With_Retrieved_Memory(system_prompt: str, context_text: str, question_text: str) -> Tuple[str, Dict[str, Any], float]:
    zero_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False}
    if llm_request is None:
        raise ImportError("OpenAI dependencies are not available. Please install the required LLM dependencies.")
    user_prompt = "Retrieved Memory Context:\n" + f"{context_text}\n\n" + "Question:\n" + f"{question_text}\n\n" + "Answer:"
    start_time = time.time()
    answer_text, cost_info = llm_request(system_prompt=system_prompt, user_prompt=user_prompt, return_parsed_json=False, extract_json=False)
    duration_ms = (time.time() - start_time) * 1000
    if isinstance(answer_text, tuple):
        answer_text = answer_text[0]
    return str(answer_text).strip(), cost_info or zero_cost, duration_ms


def Answer_Questions_For_One_Session(client: Any, session_item: Dict[str, Any], user_id: str, top_k: int, system_prompt: str, overwrite_existing_answers: bool) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "Memobase retrieval and question answering"}
    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = "memobase"
    session_memory_metadata["Top_K"] = top_k
    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue
        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue
        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        _, retrieved_memories, search_duration_ms = Search_Memobase(client, user_id, question_text, search_top_k)
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], user_id)
        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(system_prompt, answer_context_text, question_text)
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = "memobase"
        session_questions[q_idx] = question_item
        session_retrieval_time_ms += search_duration_ms
        session_response_time_ms += response_duration_ms
        answered_question_count += 1
        current_stage_total_cost["input_tokens"] += cost_info.get("input_tokens", 0) or 0
        current_stage_total_cost["output_tokens"] += cost_info.get("output_tokens", 0) or 0
        current_stage_total_cost["total_tokens"] += cost_info.get("total_tokens", 0) or 0
        current_stage_total_cost["total_cost_usd"] += cost_info.get("total_cost_usd", 0.0) or 0.0
        if current_stage_total_cost["model"] is None:
            current_stage_total_cost["model"] = cost_info.get("model")
        if cost_info.get("pricing_available") is True:
            current_stage_total_cost["pricing_available"] = True
    session_memory_metadata["Session_Retrieval_Time_ms"] = session_retrieval_time_ms
    session_memory_metadata["Session_Response_Time_ms"] = session_response_time_ms
    session_memory_metadata["Session_Answered_Question_Count"] = answered_question_count
    session_item["Session_Questions"] = session_questions
    session_item["Session_Memory_Metadata"] = session_memory_metadata
    return session_item, answered_question_count, current_stage_total_cost


def place_session_memory_metadata_before_event_types(session_item: Dict[str, Any]) -> Dict[str, Any]:
    target_key = "Session_Memory_Metadata"
    metadata_value = copy.deepcopy(session_item.get(target_key))
    reordered = {}
    inserted = False
    for key, value in session_item.items():
        if key == target_key:
            continue
        if key == "Event_Types" and not inserted:
            reordered[target_key] = metadata_value
            inserted = True
        reordered[key] = value
    if not inserted:
        reordered[target_key] = metadata_value
    return reordered


def Build_Compact_Memobase_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
    compact_question = {
        "question_id": question_item.get("question_id"),
        "question": question_item.get("question"),
        "answer": question_item.get("answer"),
        "conflict_type": question_item.get("conflict_type"),
        "ability_target": question_item.get("ability_target"),
        "difficulty": question_item.get("difficulty"),
        "Model_Answer": question_item.get("Model_Answer"),
        "Memory_Search_Duration_ms": question_item.get("Memory_Search_Duration_ms"),
        "Response_Duration_ms": question_item.get("Response_Duration_ms"),
        "Actual_Top_K": question_item.get("Actual_Top_K"),
        "Memory_System": question_item.get("Memory_System"),
    }
    retrieved_memories = question_item.get("Retrieved_Memories", [])
    compact_question["Retrieved_Memories"] = (
        retrieved_memories[:max(keep_top_k, MAX_STORED_RETRIEVED_MEMORIES)]
        if isinstance(retrieved_memories, list)
        else []
    )
    return compact_question


def Build_Compact_Memobase_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
    compact_session = {
        "Session_ID": session_item.get("Session_ID"),
        "Date": session_item.get("Date"),
        "Question_Trigger_Types": copy.deepcopy(session_item.get("Question_Trigger_Types", [])),
        "Session_Question_Count": session_item.get("Session_Question_Count", 0),
        "Session_Memory_Metadata": copy.deepcopy(session_item.get("Session_Memory_Metadata", {})),
        "Session_Questions": [],
    }
    session_questions = session_item.get("Session_Questions", [])
    if isinstance(session_questions, list):
        compact_session["Session_Questions"] = [Build_Compact_Memobase_Question(question_item, keep_top_k) for question_item in session_questions if isinstance(question_item, dict)]
    return compact_session


def Build_Compact_Memobase_Result_Item(persona_item: Dict[str, Any], updated_chain: List[Dict[str, Any]], total_answered_question_count: int, answered_session_count: int, final_cost: Dict[str, Any], user_id: str, eval_top_k: int, runtime_summary: Dict[str, Any], observable_token_cost_summary: Dict[str, Any], backend_config: Dict[str, str]) -> Dict[str, Any]:
    return {
        "ID": persona_item.get("ID"),
        "Memory_System": "memobase",
        "Memobase_User_ID": user_id,
        "Memobase_Backend_Mode": backend_config.get("backend_mode"),
        "Memobase_Backend_Host": backend_config.get("backend_host"),
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "Memobase_Runtime_Summary": runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [Build_Compact_Memobase_Session(session_item, eval_top_k) for session_item in updated_chain],
    }


def Generate_Single_Persona_Memobase_Eval(persona_item: Dict[str, Any], system_prompt: str, top_k: int, version: str, overwrite_existing_answers: bool):
    try:
        client, backend_config = Setup_Memobase_Client()
        print(f"[DEBUG] Memobase backend mode={backend_config.get('backend_mode')} host={backend_config.get('backend_host')}")
        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()
        user_id = Build_Memobase_User_ID(persona_item, version)
        real_user_id = Resolve_Memobase_User_ID(client, user_id)
        total_answered_question_count = 0
        answered_session_count = 0
        persona_add_time_ms = 0.0
        persona_retrieval_time_ms = 0.0
        persona_response_time_ms = 0.0
        session_total_runtime_ms_sum = 0.0
        current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "Memobase retrieval and question answering"}
        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for Memobase")
            session_start_time = time.time()
            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            add_duration_ms, add_batch_results = Add_Session_Dialogue_To_Memobase(
                client,
                real_user_id,
                dialogue_messages,
                session_date=session_item.get("Date"),
            )
            persona_add_time_ms += add_duration_ms
            time.sleep(MEMOBASE_POST_ADD_BUFFER_SECONDS)
            session_questions = session_item.get("Session_Questions", [])
            probe_queries = Build_Retrieval_Probe_Queries(dialogue_messages, session_questions)
            retrieval_ready_result = Wait_For_Memobase_Retrieval_Ready(client, real_user_id, probe_queries, top_k)
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": "memobase",
                "Top_K": top_k,
                "Dialogue_Added_To_Memory": len(dialogue_messages) > 0,
                "Dialogue_Message_Count": len(dialogue_messages),
                "Real_User_ID": real_user_id,
                "Backend_Mode": backend_config.get("backend_mode"),
                "Backend_Host": backend_config.get("backend_host"),
                "Context_Min_Tokens": MEMOBASE_CONTEXT_MIN_TOKENS,
                "Context_Tokens_Per_Memory": MEMOBASE_CONTEXT_TOKENS_PER_MEMORY,
                "Context_Event_Similarity_Threshold": MEMOBASE_CONTEXT_EVENT_SIMILARITY_THRESHOLD,
                "Context_Fill_Window_With_Events": True,
                "Context_Mode": "profile_event_dual_rerank_with_context_fallback",
                "Add_Duration_ms": add_duration_ms,
                "Add_Batch_Count": len(add_batch_results),
                "Add_Batch_Size": len(dialogue_messages),
                "Add_Batch_Results": add_batch_results,
                "Retrieval_Ready_Status": retrieval_ready_result.get("status"),
                "Retrieval_Ready_Attempt_Count": retrieval_ready_result.get("attempt_count", 0),
                "Retrieval_Ready_Probe_Queries": retrieval_ready_result.get("probe_queries", []),
                "Retrieval_Ready_Matched_Query": retrieval_ready_result.get("matched_query"),
                "Retrieval_Ready_Matched_Count": retrieval_ready_result.get("matched_count", 0),
            }
            if retrieval_ready_result.get("status") != "READY":
                session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"] = (time.time() - session_start_time) * 1000
                full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(session_item)
                raise RuntimeError(
                    "Memobase retrieval is not ready for the current session. "
                    f"status={retrieval_ready_result.get('status')}, "
                    f"attempt_count={retrieval_ready_result.get('attempt_count', 0)}, "
                    f"matched_query={retrieval_ready_result.get('matched_query')}"
                )
            if not isinstance(session_questions, list) or len(session_questions) == 0:
                session_item["Session_Memory_Metadata"]["Session_Retrieval_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Response_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Answered_Question_Count"] = 0
                session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"] = (time.time() - session_start_time) * 1000
                session_total_runtime_ms_sum += session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"]
                full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(session_item)
                continue
            answered_session_count += 1
            updated_session_item, answered_question_count, call_cost = Answer_Questions_For_One_Session(client, session_item, real_user_id, top_k, system_prompt, overwrite_existing_answers)
            total_answered_question_count += answered_question_count
            current_stage_total_cost["input_tokens"] += call_cost.get("input_tokens", 0) or 0
            current_stage_total_cost["output_tokens"] += call_cost.get("output_tokens", 0) or 0
            current_stage_total_cost["total_tokens"] += call_cost.get("total_tokens", 0) or 0
            current_stage_total_cost["total_cost_usd"] += call_cost.get("total_cost_usd", 0.0) or 0.0
            if current_stage_total_cost["model"] is None:
                current_stage_total_cost["model"] = call_cost.get("model")
            if call_cost.get("pricing_available") is True:
                current_stage_total_cost["pricing_available"] = True
            session_metadata = updated_session_item.get("Session_Memory_Metadata", {})
            persona_retrieval_time_ms += session_metadata.get("Session_Retrieval_Time_ms", 0.0) or 0.0
            persona_response_time_ms += session_metadata.get("Session_Response_Time_ms", 0.0) or 0.0
            session_metadata["Session_Total_Runtime_ms"] = (time.time() - session_start_time) * 1000
            updated_session_item["Session_Memory_Metadata"] = session_metadata
            session_total_runtime_ms_sum += session_metadata["Session_Total_Runtime_ms"]
            full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(updated_session_item)
        persona_total_runtime_ms = (time.time() - persona_start_time) * 1000
        runtime_summary = {
            "Persona_Add_Time_ms": persona_add_time_ms,
            "Persona_Retrieval_Time_ms": persona_retrieval_time_ms,
            "Persona_Response_Time_ms": persona_response_time_ms,
            "Persona_Total_Runtime_ms": persona_total_runtime_ms,
            "Average_Add_Time_Per_Session_ms": (persona_add_time_ms / len(full_session_chain)) if len(full_session_chain) > 0 else 0.0,
            "Average_Retrieval_Time_Per_Session_ms": (persona_retrieval_time_ms / answered_session_count) if answered_session_count > 0 else 0.0,
            "Average_Response_Time_Per_Session_ms": (persona_response_time_ms / answered_session_count) if answered_session_count > 0 else 0.0,
            "Average_Total_Runtime_Per_Session_ms": (session_total_runtime_ms_sum / len(full_session_chain)) if len(full_session_chain) > 0 else 0.0,
        }
        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(current_stage_total_cost, "memobase_answer_generation")
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_answered_question_count, answered_session_count, final_cost, real_user_id, runtime_summary, observable_token_cost_summary, backend_config
    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_Memobase_Eval failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Memobase_Eval(input_jsonl_path: str, output_jsonl_path: str, output_json_path: str, system_prompt: str, top_k: int, start_idx: int, end_idx: Optional[int], version: str, overwrite_existing_answers: bool):
    try:
        all_items = load_jsonl_items(input_jsonl_path)
        selected_items = all_items[start_idx:end_idx] if end_idx is not None else all_items[start_idx:]
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        output_items = []
        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing Memobase persona {absolute_idx + 1}/{len(all_items)}")
            updated_chain, total_answered_question_count, answered_session_count, final_cost, user_id, runtime_summary, observable_token_cost_summary, backend_config = Generate_Single_Persona_Memobase_Eval(persona_item=persona_item, system_prompt=system_prompt, top_k=top_k, version=version, overwrite_existing_answers=overwrite_existing_answers)
            result_item = Build_Compact_Memobase_Result_Item(persona_item=persona_item, updated_chain=updated_chain, total_answered_question_count=total_answered_question_count, answered_session_count=answered_session_count, final_cost=final_cost, user_id=user_id, eval_top_k=top_k, runtime_summary=runtime_summary, observable_token_cost_summary=observable_token_cost_summary, backend_config=backend_config)
            output_items.append(result_item)
            write_jsonl_items(output_jsonl_path, output_items)
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump(output_items, outfile, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Memobase evaluation completed for {len(output_items)} personas")
    except Exception as e:
        print(f"Error processing Memobase evaluation: {e}:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Memobase evaluation on persona dataset")
    parser.add_argument("--input_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "..", "Data", "Step4_4.jsonl"))
    parser.add_argument("--output_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memobase_results.jsonl"))
    parser.add_argument("--output_json_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memobase_results.json"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--overwrite_existing_answers", action="store_true")
    args = parser.parse_args()

    system_prompt = (
        "You are a helpful assistant. Answer the user's question only using the retrieved memory context when possible. "
        "If the retrieved memories do not provide enough evidence, say so briefly instead of guessing."
    )

    Generate_User_Memobase_Eval(
        input_jsonl_path=args.input_jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        output_json_path=args.output_json_path,
        system_prompt=system_prompt,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=args.version,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )
