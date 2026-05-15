import argparse
import copy
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    import jsonlines
except ImportError:
    jsonlines = None

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

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


MEMZERO_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "https://api.mem0.ai")
MAX_STORED_RETRIEVED_MEMORIES = 5
MEMZERO_SKIP_RETRIEVAL_READY_POLL = os.getenv("MEMZERO_SKIP_RETRIEVAL_READY_POLL", "1").strip().lower() not in {"0", "false", "no"}
MEMZERO_POST_ADD_BUFFER_SECONDS = max(0.0, float(os.getenv("MEMZERO_POST_ADD_BUFFER_SECONDS", "30")))


def Build_MemZero_Headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Token {api_key}"}


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


def Wait_For_MemZero_Event(api_key: str,
                           event_id: Optional[str],
                           max_retries: int = 120,
                           poll_interval_seconds: int = 3) -> Dict[str, Any]:
    """Wait for one Mem0 async event to finish.

    If Mem0 keeps the event in RUNNING/PENDING beyond the local wait budget,
    or the event endpoint temporarily returns transient HTTP/network errors,
    return a soft-timeout status instead of crashing the whole evaluation.
    """
    wait_start_time = time.time()
    if not event_id:
        return {
            "status": "UNKNOWN",
            "event_id": None,
            "message": "No event_id returned by Mem0 add.",
            "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
        }

    event_url = f"{MEM0_BASE_URL.rstrip('/')}/v1/event/{event_id}/"
    last_result = {"status": "PENDING", "event_id": event_id}
    transient_http_error_count = 0
    last_http_error = None

    for attempt_idx in range(max_retries):
        try:
            response = requests.get(
                event_url,
                headers=Build_MemZero_Headers(api_key),
                timeout=120
            )

            if response.status_code >= 400:
                transient_http_error_count += 1
                last_http_error = {
                    "status_code": response.status_code,
                    "reason": response.reason,
                    "text": response.text[:500],
                }
                print(
                    f"[DEBUG] Mem0 event poll {attempt_idx + 1}/{max_retries}: "
                    f"transient_http_status={response.status_code}"
                )
                time.sleep(poll_interval_seconds)
                continue

            response.raise_for_status()
            event_result = response.json()
            status = str(event_result.get("status", "")).upper()
            last_result = event_result

            print(f"[DEBUG] Mem0 event poll {attempt_idx + 1}/{max_retries}: status={status}")

            if status == "SUCCEEDED":
                event_result["Wait_Duration_ms"] = (time.time() - wait_start_time) * 1000
                return event_result
            if status == "FAILED":
                raise RuntimeError(f"Mem0 event failed: {json.dumps(event_result, ensure_ascii=False)}")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            transient_http_error_count += 1
            last_http_error = {
                "error_type": e.__class__.__name__,
                "message": str(e),
            }
            print(
                f"[DEBUG] Mem0 event poll {attempt_idx + 1}/{max_retries}: "
                f"transient_http_error={e.__class__.__name__}"
            )

        time.sleep(poll_interval_seconds)

    soft_timeout_result = copy.deepcopy(last_result)
    soft_timeout_result["status"] = f"TIMEOUT_{str(last_result.get('status', 'UNKNOWN')).upper()}"
    soft_timeout_result["event_id"] = event_id
    soft_timeout_result["message"] = (
        f"Mem0 event did not finish after {max_retries} polls. "
        "Continue with retrieval-readiness fallback."
    )
    if transient_http_error_count > 0:
        soft_timeout_result["transient_http_error_count"] = transient_http_error_count
        soft_timeout_result["last_http_error"] = last_http_error
    soft_timeout_result["Wait_Duration_ms"] = (time.time() - wait_start_time) * 1000
    print(
        f"[DEBUG] Mem0 event soft timeout: event_id={event_id}, "
        f"last_status={last_result.get('status')}, transient_http_error_count={transient_http_error_count}"
    )
    return soft_timeout_result


def load_jsonl_items(input_file: str) -> List[Dict[str, Any]]:
    """Load JSONL data with jsonlines or stdlib fallback."""
    items = []
    if jsonlines is not None:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                items.append(item)
        return items

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl_items(output_file: str, items: List[Dict[str, Any]]):
    """Write JSONL data with jsonlines or stdlib fallback."""
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
    """Parse dialogue_turn_12 -> 12 for sorting."""
    try:
        return int(str(key_name).split("_")[-1])
    except Exception:
        return 10**9


def Build_Session_Dialogue_List(session_dialogue: Any) -> List[Dict[str, Any]]:
    """Flatten Session_Dialogue dict into a chronological list of role/content turns."""
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
                flattened_dialogue.append({
                    "role": role,
                    "content": str(content)
                })

    return flattened_dialogue


def Build_MemZero_User_ID(persona_item: Dict[str, Any], version: str) -> str:
    """Build a unique user id for MemZero evaluation."""
    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown_persona")
    persona_short = persona_id[-6:] if len(persona_id) >= 6 else persona_id
    unique_suffix = f"{time.time_ns()}_{uuid.uuid4().hex[:10]}"
    return f"mz_{persona_short}_{version}_{unique_suffix}"


def Build_Session_Timestamp(session_item: Dict[str, Any], fallback_index: int) -> int:
    """Convert session date to unix timestamp."""
    session_date = str(session_item.get("Date", "")).strip()

    for fmt in ["%Y-%m-%d", "%Y/%m/%d"]:
        try:
            dt = datetime.strptime(session_date, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp()) + fallback_index
        except Exception:
            continue

    return int(time.time()) + fallback_index


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], user_id: str) -> str:
    """Convert MemZero search results into answer prompt context."""
    lines = [f"Memories for user {user_id}:"]

    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for idx, item in enumerate(retrieved_memories, start=1):
            memory_text = item.get("memory", "")
            created_at = item.get("created_at", "Unknown Time")
            score = item.get("score", None)
            if score is None:
                lines.append(f"{idx}. [{created_at}] {memory_text}")
            else:
                lines.append(f"{idx}. [{created_at}] {memory_text} (score={score})")

    return "\n".join(lines)


def Setup_MemZero_Client() -> Tuple[Any, str]:
    """Initialize MemZero client and return the API key for event polling."""
    if MemoryClient is None:
        raise ImportError("mem0 is not installed. Please install mem0 before running eval_memzero.")

    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY is not set in the environment.")

    return MemoryClient(api_key=api_key), api_key


def Reset_MemZero_User_Memory(client, user_id: str):
    """Clear old memory for the current user id if supported."""
    try:
        client.delete_all(user_id=user_id)
        print(f"[DEBUG] Cleared old memories for user_id={user_id}")
    except Exception as e:
        print(f"[DEBUG] Skip delete_all for user_id={user_id}: {e}")


def Add_Session_Dialogue_To_MemZero(client,
                                  api_key: str,
                                  user_id: str,
                                  dialogue_messages: List[Dict[str, Any]],
                                  timestamp: int,
                                  batch_size: int = 8,
                                  add_max_retries: int = 3,
                                  add_retry_seconds: int = 5) -> Tuple[float, List[Dict[str, Any]]]:
    """Add one session dialogue to MemZero in smaller batches and wait for each batch event."""
    if len(dialogue_messages) == 0:
        return 0.0, []

    total_duration_ms = 0.0
    batch_results = []

    for batch_index, start_idx in enumerate(range(0, len(dialogue_messages), batch_size), start=1):
        batch_messages = dialogue_messages[start_idx:start_idx + batch_size]
        add_result = None
        add_exception = None
        add_attempt_count = 0
        duration_ms = 0.0

        for attempt_idx in range(add_max_retries):
            add_attempt_count = attempt_idx + 1
            start_time = time.time()
            try:
                add_result = client.add(
                    batch_messages,
                    user_id=user_id,
                    version="v2",
                    output_format="v1.1",
                    timestamp=timestamp + start_idx,
                    enable_graph=False,
                    async_mode=False
                )
                duration_ms += (time.time() - start_time) * 1000
                add_exception = None
                break
            except Exception as e:
                duration_ms += (time.time() - start_time) * 1000
                add_exception = e
                error_text = str(e)
                print(
                    f"[DEBUG] Mem0 add batch {batch_index} attempt {attempt_idx + 1}/{add_max_retries} failed: {error_text}"
                )
                if attempt_idx < add_max_retries - 1:
                    time.sleep(add_retry_seconds)

        total_duration_ms += duration_ms

        if add_exception is not None or add_result is None:
            batch_results.append({
                "Batch_Index": batch_index,
                "Batch_Size": len(batch_messages),
                "Start_Message_Index": start_idx,
                "End_Message_Index": start_idx + len(batch_messages) - 1,
                "Add_Attempt_Count": add_attempt_count,
                "Add_Duration_ms": duration_ms,
                "Add_Event_ID": None,
                "Add_Event_Status": "ADD_FAILED",
                "Add_Result": None,
                "Event_Result": {
                    "status": "ADD_FAILED",
                    "message": str(add_exception),
                }
            })
            raise add_exception

        add_event_id = None
        try:
            add_event_id = add_result.get("results", [{}])[0].get("event_id")
        except Exception:
            add_event_id = None

        event_wait_result = Wait_For_MemZero_Event(
            api_key=api_key,
            event_id=add_event_id
        )
        event_wait_duration_ms = float(event_wait_result.get("Wait_Duration_ms", 0.0) or 0.0)
        total_duration_ms += event_wait_duration_ms

        batch_results.append({
            "Batch_Index": batch_index,
            "Batch_Size": len(batch_messages),
            "Start_Message_Index": start_idx,
            "End_Message_Index": start_idx + len(batch_messages) - 1,
            "Add_Attempt_Count": add_attempt_count,
            "Add_Duration_ms": duration_ms,
            "Add_Event_Wait_Duration_ms": event_wait_duration_ms,
            "Batch_Total_Add_Duration_ms": duration_ms + event_wait_duration_ms,
            "Add_Event_ID": add_event_id,
            "Add_Event_Status": event_wait_result.get("status"),
            "Add_Result": add_result,
            "Event_Result": event_wait_result
        })

    return total_duration_ms, batch_results


def Build_Retrieval_Probe_Queries(dialogue_messages: List[Dict[str, Any]],
                                  session_questions: List[Dict[str, Any]]) -> List[str]:
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


def Wait_For_MemZero_Retrieval_Ready(client,
                                     user_id: str,
                                     probe_queries: List[str],
                                     top_k: int,
                                     max_retries: int = 8,
                                     poll_interval_seconds: int = 2) -> Dict[str, Any]:
    wait_start_time = time.time()
    if len(probe_queries) == 0:
        return {
            "status": "SKIPPED",
            "attempt_count": 0,
            "probe_queries": [],
            "matched_query": None,
            "matched_count": 0,
            "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
        }

    last_probe_result = {"status": "PENDING", "matched_query": None, "matched_count": 0}
    for attempt_idx in range(max_retries):
        for probe_query in probe_queries:
            try:
                search_result = client.search(
                    probe_query,
                    top_k=max(1, min(top_k, 3)),
                    output_format="v1.1",
                    version="v2",
                    filters={"AND": [{"user_id": user_id}]}
                )
                results = search_result.get("results", []) if isinstance(search_result, dict) else []
                if len(results) > 0:
                    return {
                        "status": "READY",
                        "attempt_count": attempt_idx + 1,
                        "probe_queries": probe_queries,
                        "matched_query": probe_query,
                        "matched_count": len(results),
                        "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
                    }
                last_probe_result = {
                    "status": "WAITING",
                    "matched_query": probe_query,
                    "matched_count": 0,
                }
            except Exception as e:
                last_probe_result = {
                    "status": "SEARCH_ERROR",
                    "matched_query": probe_query,
                    "matched_count": 0,
                    "error": str(e),
                }

        print(
            f"[DEBUG] Mem0 retrieval readiness poll {attempt_idx + 1}/{max_retries}: "
            f"status={last_probe_result.get('status')}"
        )
        time.sleep(poll_interval_seconds)

    last_probe_result.update({
        "status": "TIMEOUT",
        "attempt_count": max_retries,
        "probe_queries": probe_queries,
        "Wait_Duration_ms": (time.time() - wait_start_time) * 1000,
    })
    return last_probe_result


def Build_MemZero_Retrieval_Ready_Result(client,
                                       user_id: str,
                                       top_k: int,
                                       dialogue_messages: List[Dict[str, Any]],
                                       probe_queries: List[str],
                                       add_batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if MEMZERO_SKIP_RETRIEVAL_READY_POLL:
        last_batch_result = add_batch_results[-1] if len(add_batch_results) > 0 else {}
        if len(dialogue_messages) == 0:
            return {
                "status": "SKIPPED_NO_DIALOGUE",
                "attempt_count": 0,
                "probe_queries": [],
                "matched_query": None,
                "matched_count": 0,
                "Wait_Duration_ms": 0.0,
            }
        return {
            "status": "SKIPPED_AFTER_SYNC_ADD",
            "attempt_count": 0,
            "probe_queries": probe_queries,
            "matched_query": None,
            "matched_count": 0,
            "Wait_Duration_ms": 0.0,
            "add_event_status": last_batch_result.get("Add_Event_Status"),
        }

    return Wait_For_MemZero_Retrieval_Ready(
        client=client,
        user_id=user_id,
        probe_queries=probe_queries,
        top_k=top_k,
    )


def Search_MemZero_For_Question(client, user_id: str, question_text: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    """Retrieve relevant memories for one question."""
    start_time = time.time()
    search_result = client.search(
        question_text,
        top_k=top_k,
        output_format="v1.1",
        version="v2",
        filters={"AND": [{"user_id": user_id}]}
    )

    retrieved_memories = []
    if isinstance(search_result, dict):
        for item in search_result.get("results", []):
            if not isinstance(item, dict):
                continue
            retrieved_memories.append({
                "memory": item.get("memory", ""),
                "created_at": item.get("created_at", "Unknown Time"),
                "score": item.get("score")
            })

    duration_ms = (time.time() - start_time) * 1000
    context_text = Build_Retrieved_Memory_Context(retrieved_memories, user_id)
    return context_text, retrieved_memories, duration_ms


def Generate_Answer_With_Retrieved_Memory(system_prompt: str, context_text: str, question_text: str) -> Tuple[str, Dict[str, Any], float]:
    """Use retrieved memory context to answer one question."""
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }

    if llm_request is None:
        raise ImportError("OpenAI dependencies are not available. Please install the required LLM dependencies.")

    user_prompt = (
        "Retrieved Memory Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Answer:"
    )

    start_time = time.time()
    answer_text, cost_info = llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        return_parsed_json=False,
        extract_json=False
    )
    duration_ms = (time.time() - start_time) * 1000

    if isinstance(answer_text, tuple):
        answer_text = answer_text[0]

    return str(answer_text).strip(), cost_info or zero_cost, duration_ms


def Answer_Questions_For_One_Session(client,
                                     session_item: Dict[str, Any],
                                     user_id: str,
                                     top_k: int,
                                     system_prompt: str,
                                     overwrite_existing_answers: bool) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """Answer all questions in one session using MemZero."""
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
        "note": "MemZero retrieval and question answering"
    }

    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = "memzero"
    session_memory_metadata["Top_K"] = top_k

    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue

        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue

        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        context_text, retrieved_memories, search_duration_ms = Search_MemZero_For_Question(
            client=client,
            user_id=user_id,
            question_text=question_text,
            top_k=search_top_k
        )
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], user_id)

        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(
            system_prompt=system_prompt,
            context_text=answer_context_text,
            question_text=question_text
        )

        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = "memzero"
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
    """Place Session_Memory_Metadata before Event_Types for readability."""
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


def Build_Compact_MemZero_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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
    if isinstance(retrieved_memories, list):
        compact_question["Retrieved_Memories"] = retrieved_memories[:max(keep_top_k, MAX_STORED_RETRIEVED_MEMORIES)]
    else:
        compact_question["Retrieved_Memories"] = []

    return compact_question



def Build_Compact_MemZero_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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
        compact_session["Session_Questions"] = [
            Build_Compact_MemZero_Question(question_item, keep_top_k)
            for question_item in session_questions
            if isinstance(question_item, dict)
        ]

    return compact_session



def Build_Compact_MemZero_Result_Item(
    persona_item: Dict[str, Any],
    updated_chain: List[Dict[str, Any]],
    total_answered_question_count: int,
    answered_session_count: int,
    final_cost: Dict[str, Any],
    memzero_user_id: str,
    eval_top_k: int,
    persona_runtime_summary: Dict[str, Any],
    observable_token_cost_summary: Dict[str, Any],
) -> Dict[str, Any]:
    compact_result = {
        "ID": persona_item.get("ID"),
        "Memory_System": "memzero",
        "MemZero_User_ID": memzero_user_id,
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "MemZero_Runtime_Summary": persona_runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [
            Build_Compact_MemZero_Session(session_item, eval_top_k)
            for session_item in updated_chain
        ],
    }
    return compact_result


def Generate_Single_Persona_MemZero_Eval(persona_item: Dict[str, Any],
                                         system_prompt: str,
                                         top_k: int,
                                         version: str,
                                         overwrite_existing_answers: bool):
    """Run MemZero evaluation for one persona."""
    try:
        client, mem0_api_key = Setup_MemZero_Client()
        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()

        user_id = Build_MemZero_User_ID(persona_item, version)
        Reset_MemZero_User_Memory(client, user_id)

        total_answered_question_count = 0
        answered_session_count = 0
        persona_add_time_ms = 0.0
        persona_retrieval_time_ms = 0.0
        persona_response_time_ms = 0.0
        session_total_runtime_ms_sum = 0.0
        current_stage_total_cost = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "model": None,
            "pricing_available": False,
            "note": "MemZero retrieval and question answering"
        }

        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for MemZero")
            session_start_time = time.time()

            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            timestamp = Build_Session_Timestamp(session_item, current_idx)
            add_duration_ms, add_batch_results = Add_Session_Dialogue_To_MemZero(
                client=client,
                api_key=mem0_api_key,
                user_id=user_id,
                dialogue_messages=dialogue_messages,
                timestamp=timestamp
            )
            if MEMZERO_POST_ADD_BUFFER_SECONDS > 0:
                time.sleep(MEMZERO_POST_ADD_BUFFER_SECONDS)
            last_batch_result = add_batch_results[-1] if len(add_batch_results) > 0 else {}
            session_questions = session_item.get("Session_Questions", [])
            probe_queries = Build_Retrieval_Probe_Queries(dialogue_messages, session_questions)
            retrieval_ready_result = Build_MemZero_Retrieval_Ready_Result(
                client=client,
                user_id=user_id,
                top_k=top_k,
                dialogue_messages=dialogue_messages,
                probe_queries=probe_queries,
                add_batch_results=add_batch_results,
            )
            retrieval_ready_wait_ms = float(retrieval_ready_result.get("Wait_Duration_ms", 0.0) or 0.0)
            add_duration_ms += retrieval_ready_wait_ms
            persona_add_time_ms += add_duration_ms
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": "memzero",
                "Top_K": top_k,
                "Dialogue_Added_To_Memory": len(dialogue_messages) > 0,
                "Dialogue_Message_Count": len(dialogue_messages),
                "Add_Duration_ms": add_duration_ms,
                "Post_Add_Buffer_Seconds": MEMZERO_POST_ADD_BUFFER_SECONDS,
                "Retrieval_Ready_Wait_Duration_ms": retrieval_ready_wait_ms,
                "Add_Batch_Count": len(add_batch_results),
                "Add_Batch_Size": 8,
                "Add_Event_ID": last_batch_result.get("Add_Event_ID"),
                "Add_Event_Status": last_batch_result.get("Add_Event_Status"),
                "Add_Batch_Results": add_batch_results,
                "Retrieval_Ready_Status": retrieval_ready_result.get("status"),
                "Retrieval_Ready_Attempt_Count": retrieval_ready_result.get("attempt_count", 0),
                "Retrieval_Ready_Probe_Queries": retrieval_ready_result.get("probe_queries", []),
                "Retrieval_Ready_Matched_Query": retrieval_ready_result.get("matched_query"),
                "Retrieval_Ready_Matched_Count": retrieval_ready_result.get("matched_count", 0),
                "Skip_Retrieval_Ready_Poll": MEMZERO_SKIP_RETRIEVAL_READY_POLL
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
                client=client,
                session_item=session_item,
                user_id=user_id,
                top_k=top_k,
                system_prompt=system_prompt,
                overwrite_existing_answers=overwrite_existing_answers
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
            print(
                f"[DEBUG] Session {session_item.get('Session_ID', current_idx)} completed - "
                f"questions answered: {answered_question_count}"
            )

        persona_total_runtime_ms = (time.time() - persona_start_time) * 1000
        persona_runtime_summary = {
            "Persona_Add_Time_ms": persona_add_time_ms,
            "Persona_Retrieval_Time_ms": persona_retrieval_time_ms,
            "Persona_Response_Time_ms": persona_response_time_ms,
            "Persona_Total_Runtime_ms": persona_total_runtime_ms,
            "Average_Add_Time_Per_Session_ms": (persona_add_time_ms / len(full_session_chain)) if len(full_session_chain) > 0 else 0.0,
            "Average_Retrieval_Time_Per_Session_ms": (persona_retrieval_time_ms / answered_session_count) if answered_session_count > 0 else 0.0,
            "Average_Response_Time_Per_Session_ms": (persona_response_time_ms / answered_session_count) if answered_session_count > 0 else 0.0,
            "Average_Total_Runtime_Per_Session_ms": (session_total_runtime_ms_sum / len(full_session_chain)) if len(full_session_chain) > 0 else 0.0
        }

        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(
            stage_cost=current_stage_total_cost,
            stage_name="memzero_answer_generation",
        )
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_answered_question_count, answered_session_count, final_cost, user_id, persona_runtime_summary, observable_token_cost_summary

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_MemZero_Eval failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_MemZero_Eval(
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
    """Batch entry for MemZero evaluation."""
    print(f"Processing file: {input_jsonl_path}")
    print(f"Output file: {output_jsonl_path}")

    try:
        print("[DEBUG] Using built-in MemZero answer prompt")
        print(f"[DEBUG] Top-K retrieval: {top_k}")

        all_personas = load_jsonl_items(input_jsonl_path)
        selected_items = all_personas[start_idx:end_idx] if end_idx is not None else all_personas[start_idx:]
        print(f"[DEBUG] Read {len(all_personas)} personas")

        all_results = []
        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing persona {absolute_idx + 1}/{len(all_personas)}")

            updated_chain, total_answered_question_count, answered_session_count, final_cost, memzero_user_id, persona_runtime_summary, observable_token_cost_summary = (
                Generate_Single_Persona_MemZero_Eval(
                    persona_item=persona_item,
                    system_prompt=system_prompt,
                    top_k=top_k,
                    version=version,
                    overwrite_existing_answers=overwrite_existing_answers
                )
            )

            result_item = Build_Compact_MemZero_Result_Item(
                persona_item=persona_item,
                updated_chain=updated_chain,
                total_answered_question_count=total_answered_question_count,
                answered_session_count=answered_session_count,
                final_cost=final_cost,
                memzero_user_id=memzero_user_id,
                eval_top_k=top_k,
                persona_runtime_summary=persona_runtime_summary,
                observable_token_cost_summary=observable_token_cost_summary,
            )
            all_results.append(result_item)

            print(
                f"[DEBUG] Persona {absolute_idx + 1} completed - "
                f"Answered sessions: {answered_session_count}, Answered questions: {total_answered_question_count}"
            )

        write_jsonl_items(output_jsonl_path, all_results)

        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            if len(all_results) == 1:
                json.dump(all_results[0], f, ensure_ascii=False, indent=4)
            else:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

        print("[DEBUG] Successfully processed MemZero evaluation")
        return True

    except Exception as e:
        print(f"Error processing MemZero evaluation: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MemZero evaluation on persona dataset")
    parser.add_argument("--input_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Data", "Step4_5_long.jsonl"))
    parser.add_argument("--output_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memzero_results.jsonl"))
    parser.add_argument("--output_json_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "memzero_results.json"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--overwrite_existing_answers", action="store_true")
    args = parser.parse_args()

    Generate_User_MemZero_Eval(
        input_jsonl_path=args.input_jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        output_json_path=args.output_json_path,
        system_prompt=MEMZERO_ANSWER_SYSTEM_PROMPT,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=args.version,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )



