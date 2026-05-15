import argparse
import copy
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    import jsonlines
except ImportError:
    jsonlines = None

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

MEMOS_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""
MAX_STORED_RETRIEVED_MEMORIES = 5
DEFAULT_MEMOS_INITIAL_READY_WAIT_SECONDS = int(os.getenv("MEMOS_INITIAL_READY_WAIT_SECONDS", "30"))
DEFAULT_MEMOS_READY_MAX_RETRIES = int(os.getenv("MEMOS_READY_MAX_RETRIES", "20"))
DEFAULT_MEMOS_READY_POLL_INTERVAL_SECONDS = int(os.getenv("MEMOS_READY_POLL_INTERVAL_SECONDS", "3"))


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


def Build_Memos_Headers(memos_key: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Token {memos_key}"}


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
                dialogue_turn = {"role": role, "content": str(content)}
                for time_key in ["chat_time", "timestamp", "created_at", "time"]:
                    raw_time_value = message_item.get(time_key)
                    if raw_time_value not in [None, ""]:
                        dialogue_turn[time_key] = raw_time_value
                flattened_dialogue.append(dialogue_turn)
    return flattened_dialogue


def Build_Memos_User_ID(persona_item: Dict[str, Any], version: str) -> str:
    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown_persona")
    persona_short = persona_id[-6:] if len(persona_id) >= 6 else persona_id
    unique_suffix = f"{time.time_ns()}_{uuid.uuid4().hex[:10]}"
    return f"mm_{persona_short}_{version}_{unique_suffix}"


def Build_Session_Conversation_ID(session_item: Dict[str, Any], fallback_index: int, user_id: str) -> str:
    session_id = session_item.get("Session_ID", fallback_index)
    return f"session_{session_id}_{user_id}"


def Build_Session_Chat_Time(session_item: Dict[str, Any], fallback_index: int) -> datetime:
    session_date = str(session_item.get("Date", "")).strip()
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"]:
        try:
            dt = datetime.strptime(session_date, fmt).replace(tzinfo=timezone.utc)
            return dt.replace(hour=12, minute=0, second=0) + timedelta(minutes=fallback_index)
        except Exception:
            continue
    return datetime.now(timezone.utc) + timedelta(minutes=fallback_index)


def _Format_Memos_Chat_Time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _Try_Parse_Message_Time(raw_value: Any) -> Optional[datetime]:
    if raw_value in [None, ""]:
        return None
    if isinstance(raw_value, datetime):
        return raw_value if raw_value.tzinfo is not None else raw_value.replace(tzinfo=timezone.utc)
    if isinstance(raw_value, (int, float)):
        numeric_value = float(raw_value)
        if numeric_value > 1e12:
            numeric_value = numeric_value / 1000.0
        try:
            return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
        except Exception:
            return None
    if not isinstance(raw_value, str):
        return None
    raw_text = raw_value.strip()
    if not raw_text:
        return None
    iso_candidate = raw_text.replace("Z", "+00:00") if raw_text.endswith("Z") else raw_text
    try:
        parsed_dt = datetime.fromisoformat(iso_candidate)
        return parsed_dt if parsed_dt.tzinfo is not None else parsed_dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    for fmt in [
        "%b %d, %Y, %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%dT%H:%M:%S",
    ]:
        try:
            return datetime.strptime(raw_text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def Build_Message_Chat_Time(
    message_item: Dict[str, Any],
    session_base_chat_time: datetime,
    message_index: int,
) -> str:
    for time_key in ["chat_time", "timestamp", "created_at", "time"]:
        parsed_dt = _Try_Parse_Message_Time(message_item.get(time_key))
        if parsed_dt is not None:
            return _Format_Memos_Chat_Time(parsed_dt)
    fallback_dt = session_base_chat_time + timedelta(seconds=message_index)
    return _Format_Memos_Chat_Time(fallback_dt)


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], user_id: str) -> str:
    lines = [f"Memories for user {user_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for idx, item in enumerate(retrieved_memories, start=1):
            memory_text = item.get("memory", "")
            score = item.get("score")
            if score is None:
                lines.append(f"{idx}. {memory_text}")
            else:
                lines.append(f"{idx}. {memory_text} (score={score})")
    return "\n".join(lines)


def Setup_Memos_Config() -> Tuple[str, str]:
    memos_url = os.getenv("MEMOS_ONLINE_URL")
    memos_key = os.getenv("MEMOS_KEY")
    if not memos_url:
        raise ValueError("MEMOS_ONLINE_URL is not set in the environment.")
    if not memos_key:
        raise ValueError("MEMOS_KEY is not set in the environment.")
    return memos_url.rstrip("/"), memos_key


def Reset_Memos_User_Memory(memos_url: str, memos_key: str, user_id: str):
    candidate_endpoints = ["delete/user", "delete/memory", "clear/user"]
    payload = {"user_id": user_id, "conversation_id": ""}
    for endpoint in candidate_endpoints:
        try:
            response = requests.post(
                f"{memos_url}/{endpoint}",
                data=json.dumps(payload, ensure_ascii=False),
                headers=Build_Memos_Headers(memos_key),
                timeout=30,
            )
            print(f"[DEBUG] MEMOS cleanup try {endpoint}: status={response.status_code}")
            if response.status_code == 200:
                return
        except Exception as e:
            print(f"[DEBUG] MEMOS cleanup try {endpoint} failed: {e}")
    print(f"[DEBUG] No supported MEMOS cleanup endpoint detected for user_id={user_id}")
def Add_Session_Dialogue_To_Memos(
    memos_url: str,
    memos_key: str,
    user_id: str,
    conversation_id: str,
    dialogue_messages: List[Dict[str, Any]],
    session_base_chat_time: datetime,
    batch_size: int = 8,
) -> Tuple[float, List[Dict[str, Any]]]:
    if len(dialogue_messages) == 0:
        return 0.0, []
    total_duration_ms = 0.0
    batch_results = []
    for batch_index, start_idx in enumerate(range(0, len(dialogue_messages), batch_size), start=1):
        batch_messages = []
        for offset, message_item in enumerate(dialogue_messages[start_idx:start_idx + batch_size]):
            message_index = start_idx + offset
            chat_time = Build_Message_Chat_Time(message_item, session_base_chat_time, message_index)
            batch_messages.append({
                "role": message_item.get("role"),
                "content": message_item.get("content"),
                "chat_time": chat_time,
            })
        payload = {
            "messages": batch_messages,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        start_time = time.time()
        response = requests.post(
            f"{memos_url}/add/message",
            data=json.dumps(payload, ensure_ascii=False),
            headers=Build_Memos_Headers(memos_key),
            timeout=120,
        )
        duration_ms = (time.time() - start_time) * 1000
        total_duration_ms += duration_ms
        response.raise_for_status()
        add_result = response.json()
        status_code = add_result.get("code", 0)
        if status_code != 0:
            raise RuntimeError(f"MEMOS add/message failed: {json.dumps(add_result, ensure_ascii=False)}")
        batch_results.append({
            "Batch_Index": batch_index,
            "Batch_Size": len(batch_messages),
            "Start_Message_Index": start_idx,
            "End_Message_Index": start_idx + len(batch_messages) - 1,
            "Add_Duration_ms": duration_ms,
            "Add_HTTP_Status": response.status_code,
            "Add_Result_Code": status_code,
        })
    return total_duration_ms, batch_results


def Build_Retrieval_Probe_Queries(dialogue_messages: List[Dict[str, Any]], session_questions: List[Dict[str, Any]]) -> List[str]:
    queries = []
    for message_item in reversed(dialogue_messages):
        content = str(message_item.get("content", "")).strip()
        if content and content not in queries:
            queries.append(content)
        if message_item.get("role") == "user" and content:
            break
    for question_item in session_questions[:2]:
        question_text = str(question_item.get("question", "")).strip()
        if question_text and question_text not in queries:
            queries.append(question_text)
    return queries[:3]


def _Execute_Memos_Search_Request(
    memos_url: str,
    memos_key: str,
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
) -> requests.Response:
    payload = {"query": query, "user_id": user_id, "conversation_id": ""}
    if session_id:
        payload["session_id"] = session_id
    return requests.post(
        f"{memos_url}/search/memory",
        data=json.dumps(payload, ensure_ascii=False),
        headers=Build_Memos_Headers(memos_key),
        timeout=120,
    )


def Search_Memos(
    memos_url: str,
    memos_key: str,
    query: str,
    user_id: str,
    top_k: int,
    session_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], float]:
    start_time = time.time()
    search_modes: List[Optional[str]] = [session_id] if session_id else [None]
    if session_id:
        search_modes.append(None)
    response_json = None
    last_error: Optional[Exception] = None
    for search_session_id in search_modes:
        try:
            response = _Execute_Memos_Search_Request(
                memos_url=memos_url,
                memos_key=memos_key,
                query=query,
                user_id=user_id,
                session_id=search_session_id,
            )
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("code") != 0:
                raise RuntimeError(f"MEMOS search/memory failed: {json.dumps(response_json, ensure_ascii=False)}")
            break
        except Exception as e:
            last_error = e
            if search_session_id is not None:
                print("[DEBUG] MEMOS search with session_id failed, retrying without session_id.")
                continue
            raise
    if response_json is None:
        raise last_error if last_error is not None else RuntimeError("MEMOS search/memory failed without a response.")
    duration_ms = (time.time() - start_time) * 1000
    data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
    retrieved_memories: List[Dict[str, Any]] = []
    for item in data.get("memory_detail_list", [])[:top_k]:
        if not isinstance(item, dict):
            continue
        retrieved_memories.append({"memory": item.get("memory_value", ""), "score": item.get("score"), "source": "memory_detail_list"})
    remaining = max(0, top_k - len(retrieved_memories))
    for item in data.get("preference_detail_list", [])[:remaining]:
        if not isinstance(item, dict):
            continue
        preference_type = item.get("preference_type", "")
        preference_value = item.get("preference", "")
        retrieved_memories.append({"memory": f"{preference_type}: {preference_value}", "score": item.get("score"), "source": "preference_detail_list"})
    preference_note = str(data.get("preference_note", "")).strip()
    if preference_note and len(retrieved_memories) < top_k:
        retrieved_memories.append({"memory": preference_note, "score": None, "source": "preference_note"})
    context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], user_id)
    return context_text, retrieved_memories[:top_k], duration_ms


def Wait_For_Memos_Retrieval_Ready(
    memos_url: str,
    memos_key: str,
    user_id: str,
    probe_queries: List[str],
    top_k: int,
    session_id: Optional[str] = None,
    max_retries: int = DEFAULT_MEMOS_READY_MAX_RETRIES,
    poll_interval_seconds: int = DEFAULT_MEMOS_READY_POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    wait_start_time = time.time()
    if len(probe_queries) == 0:
        return {
            "status": "SKIPPED",
            "attempt_count": 0,
            "probe_queries": [],
            "matched_query": None,
            "matched_count": 0,
            "Wait_Duration_ms": 0.0,
        }
    last_probe_result = {"status": "PENDING", "matched_query": None, "matched_count": 0}
    for attempt_idx in range(max_retries):
        for probe_query in probe_queries:
            try:
                _, retrieved_memories, _ = Search_Memos(
                    memos_url,
                    memos_key,
                    probe_query,
                    user_id,
                    max(1, min(top_k, 3)),
                    session_id=session_id,
                )
                if len(retrieved_memories) > 0:
                    return {
                        "status": "READY",
                        "attempt_count": attempt_idx + 1,
                        "probe_queries": probe_queries,
                        "matched_query": probe_query,
                        "matched_count": len(retrieved_memories),
                        "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
                    }
                last_probe_result = {"status": "WAITING", "matched_query": probe_query, "matched_count": 0}
            except Exception as e:
                last_probe_result = {"status": "SEARCH_ERROR", "matched_query": probe_query, "matched_count": 0, "error": str(e)}
        print(f"[DEBUG] MEMOS retrieval readiness poll {attempt_idx + 1}/{max_retries}: status={last_probe_result.get('status')}")
        time.sleep(poll_interval_seconds)
    last_probe_result.update(
        {
            "status": "TIMEOUT",
            "attempt_count": max_retries,
            "probe_queries": probe_queries,
            "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
        }
    )
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


def Answer_Questions_For_One_Session(
    memos_url: str,
    memos_key: str,
    session_item: Dict[str, Any],
    user_id: str,
    conversation_id: str,
    top_k: int,
    system_prompt: str,
    overwrite_existing_answers: bool,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "MEMOS retrieval and question answering"}
    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = "memos"
    session_memory_metadata["Top_K"] = top_k
    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue
        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue
        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        _, retrieved_memories, search_duration_ms = Search_Memos(
            memos_url,
            memos_key,
            question_text,
            user_id,
            search_top_k,
            session_id=conversation_id,
        )
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], user_id)
        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(system_prompt, answer_context_text, question_text)
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = "memos"
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


def Build_Compact_Memos_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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


def Build_Compact_Memos_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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
        compact_session["Session_Questions"] = [Build_Compact_Memos_Question(question_item, keep_top_k) for question_item in session_questions if isinstance(question_item, dict)]
    return compact_session


def Build_Compact_Memos_Result_Item(
    persona_item: Dict[str, Any],
    updated_chain: List[Dict[str, Any]],
    total_answered_question_count: int,
    answered_session_count: int,
    final_cost: Dict[str, Any],
    user_id: str,
    eval_top_k: int,
    runtime_summary: Dict[str, Any],
    observable_token_cost_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ID": persona_item.get("ID"),
        "Memory_System": "memos",
        "Memos_User_ID": user_id,
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "Memos_Runtime_Summary": runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [Build_Compact_Memos_Session(session_item, eval_top_k) for session_item in updated_chain],
    }


def Generate_Single_Persona_Memos_Eval(persona_item: Dict[str, Any], system_prompt: str, top_k: int, version: str, overwrite_existing_answers: bool):
    try:
        memos_url, memos_key = Setup_Memos_Config()
        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()
        user_id = Build_Memos_User_ID(persona_item, version)
        Reset_Memos_User_Memory(memos_url, memos_key, user_id)
        total_answered_question_count = 0
        answered_session_count = 0
        persona_add_time_ms = 0.0
        persona_retrieval_time_ms = 0.0
        persona_response_time_ms = 0.0
        session_total_runtime_ms_sum = 0.0
        current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "MEMOS retrieval and question answering"}
        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for MEMOS")
            session_start_time = time.time()
            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            conversation_id = Build_Session_Conversation_ID(session_item, current_idx, user_id)
            session_base_chat_time = Build_Session_Chat_Time(session_item, current_idx)
            add_duration_ms, add_batch_results = Add_Session_Dialogue_To_Memos(
                memos_url,
                memos_key,
                user_id,
                conversation_id,
                dialogue_messages,
                session_base_chat_time,
            )
            persona_add_time_ms += add_duration_ms
            session_questions = session_item.get("Session_Questions", [])
            probe_queries = Build_Retrieval_Probe_Queries(dialogue_messages, session_questions)
            initial_ready_wait_ms = 0.0
            if len(dialogue_messages) > 0 and DEFAULT_MEMOS_INITIAL_READY_WAIT_SECONDS > 0:
                print(
                    f"[DEBUG] MEMOS post-add fixed wait: "
                    f"{DEFAULT_MEMOS_INITIAL_READY_WAIT_SECONDS}s for conversation_id={conversation_id}"
                )
                initial_wait_start = time.time()
                time.sleep(DEFAULT_MEMOS_INITIAL_READY_WAIT_SECONDS)
                initial_ready_wait_ms = (time.time() - initial_wait_start) * 1000
                persona_add_time_ms += initial_ready_wait_ms
            retrieval_ready_result = Wait_For_Memos_Retrieval_Ready(
                memos_url,
                memos_key,
                user_id,
                probe_queries,
                top_k,
                session_id=conversation_id,
            )
            retrieval_ready_wait_ms = retrieval_ready_result.get("Wait_Duration_ms", 0.0) or 0.0
            persona_add_time_ms += retrieval_ready_wait_ms
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": "memos",
                "Top_K": top_k,
                "Dialogue_Added_To_Memory": len(dialogue_messages) > 0,
                "Dialogue_Message_Count": len(dialogue_messages),
                "Conversation_ID": conversation_id,
                "Add_Duration_ms": add_duration_ms,
                "Add_Batch_Count": len(add_batch_results),
                "Add_Batch_Size": 8,
                "Add_Batch_Results": add_batch_results,
                "Initial_Post_Add_Wait_Duration_ms": initial_ready_wait_ms,
                "Retrieval_Ready_Wait_Duration_ms": retrieval_ready_wait_ms,
                "Retrieval_Ready_Status": retrieval_ready_result.get("status"),
                "Retrieval_Ready_Attempt_Count": retrieval_ready_result.get("attempt_count", 0),
                "Retrieval_Ready_Probe_Queries": retrieval_ready_result.get("probe_queries", []),
                "Retrieval_Ready_Matched_Query": retrieval_ready_result.get("matched_query"),
                "Retrieval_Ready_Matched_Count": retrieval_ready_result.get("matched_count", 0),
            }
            if not isinstance(session_questions, list) or len(session_questions) == 0:
                session_item["Session_Memory_Metadata"]["Session_Retrieval_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Response_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Answered_Question_Count"] = 0
                session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"] = (time.time() - session_start_time) * 1000
                session_total_runtime_ms_sum += session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"]
                full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(session_item)
                continue
            answered_session_count += 1
            updated_session_item, answered_question_count, call_cost = Answer_Questions_For_One_Session(
                memos_url,
                memos_key,
                session_item,
                user_id,
                conversation_id,
                top_k,
                system_prompt,
                overwrite_existing_answers,
            )
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
        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(current_stage_total_cost, "memos_answer_generation")
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_answered_question_count, answered_session_count, final_cost, user_id, runtime_summary, observable_token_cost_summary
    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_Memos_Eval failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Memos_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    system_prompt: str,
    top_k: int,
    start_idx: int,
    end_idx: Optional[int],
    version: str,
    overwrite_existing_answers: bool,
):
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
            all_items = [json.loads(line) for line in infile if line.strip()]
        selected_items = all_items[start_idx:end_idx] if end_idx is not None else all_items[start_idx:]
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        output_items = []
        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing MEMOS persona {absolute_idx + 1}/{len(all_items)}")
            updated_chain, total_answered_question_count, answered_session_count, final_cost, user_id, runtime_summary, observable_token_cost_summary = Generate_Single_Persona_Memos_Eval(
                persona_item=persona_item,
                system_prompt=system_prompt,
                top_k=top_k,
                version=version,
                overwrite_existing_answers=overwrite_existing_answers,
            )
            result_item = Build_Compact_Memos_Result_Item(
                persona_item=persona_item,
                updated_chain=updated_chain,
                total_answered_question_count=total_answered_question_count,
                answered_session_count=answered_session_count,
                final_cost=final_cost,
                user_id=user_id,
                eval_top_k=top_k,
                runtime_summary=runtime_summary,
                observable_token_cost_summary=observable_token_cost_summary,
            )
            output_items.append(result_item)
            with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                for output_item in output_items:
                    outfile.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump(output_items, outfile, ensure_ascii=False, indent=2)
        print(f"[DEBUG] MEMOS evaluation completed for {len(output_items)} personas")
    except Exception as e:
        print(f"Error processing MEMOS evaluation: {e}:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MEMOS evaluation on persona dataset")
    parser.add_argument("--input_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Data", "Step4_4_no_interference.jsonl"))
    parser.add_argument("--output_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memos_results.jsonl"))
    parser.add_argument("--output_json_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memos_results.json"))
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

    Generate_User_Memos_Eval(
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

