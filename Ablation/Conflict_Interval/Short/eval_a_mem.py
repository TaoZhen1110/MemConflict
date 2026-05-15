import argparse
import copy
import json
import os
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import jsonlines
except ImportError:
    jsonlines = None

try:
    from agentic_memory.memory_system import AgenticMemorySystem
except ImportError:
    AgenticMemorySystem = None

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

A_MEM_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

MAX_STORED_RETRIEVED_MEMORIES = 5
A_MEM_ADD_MAX_RETRIES = 3
A_MEM_ADD_RETRY_SECONDS = 5


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
                flattened_dialogue.append({"role": role, "content": str(content)})
    return flattened_dialogue


def Build_A_Mem_User_ID(persona_item: Dict[str, Any], version: str) -> str:
    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or uuid.uuid4())
    normalized_persona_id = persona_id.replace("-", "_")
    return f"a_mem_{normalized_persona_id}_{version}_{int(time.time())}"


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
        if str(message_item.get("role", "")).lower() == "user" and content:
            break
    return queries[:3]


def Build_A_Mem_Note_Granularity() -> str:
    granularity = str(os.getenv("A_MEM_NOTE_GRANULARITY", "user_turn")).strip().lower()
    if granularity not in {"session", "user_turn"}:
        print(f"[DEBUG] Unsupported A_MEM_NOTE_GRANULARITY={granularity!r}; fallback to 'user_turn'.")
        return "user_turn"
    return granularity


def Build_A_Mem_Session_Notes(
    session_item: Dict[str, Any],
    dialogue_messages: List[Dict[str, Any]],
    granularity: str,
) -> List[Dict[str, Any]]:
    if len(dialogue_messages) == 0:
        return []

    session_date = str(session_item.get("Date", "")).strip()
    time_value = session_date.replace("-", "").replace("/", "")
    if len(time_value) == 8:
        time_value = f"{time_value}0000"
    if len(time_value) != 12 or not time_value.isdigit():
        time_value = None

    if granularity == "user_turn":
        notes = []
        user_turn_index = 0
        for message_index, message_item in enumerate(dialogue_messages):
            if str(message_item.get("role", "")).lower() != "user":
                continue
            user_turn_index += 1
            content = str(message_item.get("content", "")).strip()
            if not content:
                continue
            notes.append(
                {
                    "content": content,
                    "time": time_value,
                    "start_message_index": message_index,
                    "end_message_index": message_index,
                    "user_turn_index": user_turn_index,
                }
            )
        if len(notes) > 0:
            return notes

    transcript_lines = []
    for message_item in dialogue_messages:
        transcript_lines.append(f"{message_item.get('role')}: {message_item.get('content', '')}")
    return [{
        "content": "\n".join(transcript_lines),
        "time": time_value,
        "start_message_index": 0,
        "end_message_index": len(dialogue_messages) - 1,
        "user_turn_index": None,
    }]


def Patch_A_Mem_OpenAI_Base_URL(memory_system: Any, api_key: Optional[str]):
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url or not api_key:
        return
    try:
        from openai import OpenAI

        llm_obj = getattr(getattr(memory_system, "llm_controller", None), "llm", None)
        if llm_obj is not None and hasattr(llm_obj, "client"):
            llm_obj.client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"[DEBUG] Patched A-MEM OpenAI client with OPENAI_BASE_URL={base_url}")
    except Exception as e:
        print(f"[DEBUG] Failed to patch A-MEM OpenAI base URL: {e}")


def Setup_A_Mem_System() -> Any:
    if AgenticMemorySystem is None:
        raise ImportError("agentic_memory is not installed. Please install A-MEM before running eval_a_mem.")

    api_key = os.getenv("A_MEM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_backend = os.getenv("A_MEM_LLM_BACKEND", "openai")
    llm_model = os.getenv("A_MEM_LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("A_MEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if llm_backend == "openai" and not api_key:
        raise ValueError("A-MEM requires A_MEM_API_KEY or OPENAI_API_KEY when llm_backend=openai.")

    print(f"[DEBUG] A-MEM llm_backend={llm_backend}")
    print(f"[DEBUG] A-MEM llm_model={llm_model}")
    print(f"[DEBUG] A-MEM embedding_model={embedding_model}")

    memory_system = AgenticMemorySystem(
        model_name=embedding_model,
        llm_backend=llm_backend,
        llm_model=llm_model,
        api_key=api_key,
    )
    Patch_A_Mem_OpenAI_Base_URL(memory_system, api_key)
    return memory_system


def Add_Session_Dialogue_To_A_Mem(
    memory_system: Any,
    session_item: Dict[str, Any],
    dialogue_messages: List[Dict[str, Any]],
    granularity: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    note_items = Build_A_Mem_Session_Notes(session_item, dialogue_messages, granularity)
    if len(note_items) == 0:
        return 0.0, []

    total_duration_ms = 0.0
    add_results = []

    for note_index, note_item in enumerate(note_items, start=1):
        note_content = str(note_item.get("content", "")).strip()
        note_time = note_item.get("time")
        last_exception = None
        note_id = None
        duration_ms = 0.0
        add_attempt_count = 0

        for attempt_idx in range(A_MEM_ADD_MAX_RETRIES):
            add_attempt_count = attempt_idx + 1
            start_time = time.time()
            try:
                if note_time:
                    note_id = memory_system.add_note(note_content, time=note_time)
                else:
                    note_id = memory_system.add_note(note_content)
                duration_ms += (time.time() - start_time) * 1000
                last_exception = None
                break
            except Exception as e:
                duration_ms += (time.time() - start_time) * 1000
                last_exception = e
                print(
                    f"[DEBUG] A-MEM add note {note_index}/{len(note_items)} attempt "
                    f"{attempt_idx + 1}/{A_MEM_ADD_MAX_RETRIES} failed: {e}"
                )
                if attempt_idx < A_MEM_ADD_MAX_RETRIES - 1:
                    time.sleep(A_MEM_ADD_RETRY_SECONDS)

        total_duration_ms += duration_ms
        if last_exception is not None or note_id is None:
            raise last_exception

        add_results.append({
            "Batch_Index": note_index,
            "Batch_Size": 1,
            "Start_Message_Index": note_item.get("start_message_index"),
            "End_Message_Index": note_item.get("end_message_index"),
            "User_Turn_Index": note_item.get("user_turn_index"),
            "Add_Attempt_Count": add_attempt_count,
            "Add_Duration_ms": duration_ms,
            "Add_Note_ID": note_id,
            "Add_Note_Length_Chars": len(note_content),
        })

    return total_duration_ms, add_results


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], user_id: str) -> str:
    lines = [f"Memories for user {user_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for item in retrieved_memories:
            rank = item.get("rank")
            memory_text = item.get("memory", "")
            score = item.get("score")
            tags = item.get("tags", [])
            tag_text = f" tags={tags}" if tags not in [None, []] else ""
            if score is None:
                lines.append(f"{rank}. {memory_text}{tag_text}")
            else:
                lines.append(f"{rank}. {memory_text} (score={score}){tag_text}")
    return "\n".join(lines)


def Search_A_Mem(memory_system: Any, user_id: str, query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    _ = user_id
    start_time = time.time()
    raw_results = memory_system.search_agentic(query, k=top_k)
    duration_ms = (time.time() - start_time) * 1000

    retrieved_memories = []
    if isinstance(raw_results, list):
        for idx, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue
            retrieved_memories.append({
                "memory": str(item.get("content", "")).strip(),
                "context": item.get("context"),
                "keywords": item.get("keywords", []),
                "tags": item.get("tags", []),
                "timestamp": item.get("timestamp"),
                "category": item.get("category"),
                "is_neighbor": bool(item.get("is_neighbor", False)),
                "score": item.get("score"),
                "rank": idx,
            })

    context_text = Build_Retrieved_Memory_Context(retrieved_memories, user_id)
    return context_text, retrieved_memories, duration_ms


def Wait_For_A_Mem_Retrieval_Ready(memory_system: Any, user_id: str, probe_queries: List[str], top_k: int) -> Dict[str, Any]:
    if len(probe_queries) == 0:
        return {"status": "SKIPPED", "attempt_count": 0, "probe_queries": [], "matched_query": None, "matched_count": 0}

    for probe_query in probe_queries:
        try:
            _, retrieved_memories, _ = Search_A_Mem(memory_system, user_id, probe_query, max(top_k, 3))
            if len(retrieved_memories) > 0:
                return {
                    "status": "READY",
                    "attempt_count": 1,
                    "probe_queries": probe_queries,
                    "matched_query": probe_query,
                    "matched_count": len(retrieved_memories),
                }
        except Exception as e:
            print(f"[DEBUG] A-MEM retrieval probe failed for query={probe_query!r}: {e}")

    return {
        "status": "UNCONFIRMED",
        "attempt_count": 1,
        "probe_queries": probe_queries,
        "matched_query": None,
        "matched_count": 0,
    }


def Generate_Answer_With_Retrieved_Memory(system_prompt: str, context_text: str, question_text: str) -> Tuple[str, Dict[str, Any], float]:
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
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
        extract_json=False,
    )
    duration_ms = (time.time() - start_time) * 1000
    if isinstance(answer_text, tuple):
        answer_text = answer_text[0]
    return str(answer_text).strip(), cost_info or zero_cost, duration_ms


def Answer_Questions_For_One_Session(
    memory_system: Any,
    session_item: Dict[str, Any],
    user_id: str,
    top_k: int,
    system_prompt: str,
    overwrite_existing_answers: bool,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
        "note": "A-MEM retrieval and question answering",
    }
    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = "a_mem"
    session_memory_metadata["Top_K"] = top_k

    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue

        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue

        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        _, retrieved_memories, search_duration_ms = Search_A_Mem(memory_system, user_id, question_text, search_top_k)
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], user_id)
        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(
            system_prompt=system_prompt,
            context_text=answer_context_text,
            question_text=question_text,
        )

        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = "a_mem"
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


def Build_Compact_A_Mem_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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


def Build_Compact_A_Mem_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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
            Build_Compact_A_Mem_Question(question_item, keep_top_k)
            for question_item in session_questions
            if isinstance(question_item, dict)
        ]
    return compact_session


def Build_Compact_A_Mem_Result_Item(
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
        "Memory_System": "a_mem",
        "A_Mem_User_ID": user_id,
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "A_Mem_Runtime_Summary": runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [
            Build_Compact_A_Mem_Session(session_item, eval_top_k)
            for session_item in updated_chain
        ],
    }


def Generate_Single_Persona_A_Mem_Eval(
    persona_item: Dict[str, Any],
    system_prompt: str,
    top_k: int,
    version: str,
    overwrite_existing_answers: bool,
):
    try:
        memory_system = Setup_A_Mem_System()
        granularity = Build_A_Mem_Note_Granularity()
        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()
        user_id = Build_A_Mem_User_ID(persona_item, version)
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
            "note": "A-MEM retrieval and question answering",
        }

        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for A-MEM")
            session_start_time = time.time()
            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            add_duration_ms, add_batch_results = Add_Session_Dialogue_To_A_Mem(
                memory_system=memory_system,
                session_item=session_item,
                dialogue_messages=dialogue_messages,
                granularity=granularity,
            )
            persona_add_time_ms += add_duration_ms

            session_questions = session_item.get("Session_Questions", [])
            probe_queries = Build_Retrieval_Probe_Queries(dialogue_messages, session_questions)
            retrieval_ready_result = Wait_For_A_Mem_Retrieval_Ready(memory_system, user_id, probe_queries, top_k)
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": "a_mem",
                "Top_K": top_k,
                "Note_Granularity": granularity,
                "Dialogue_Added_To_Memory": len(dialogue_messages) > 0,
                "Dialogue_Message_Count": len(dialogue_messages),
                "Add_Status": "SUCCESS" if len(add_batch_results) > 0 else "SKIPPED",
                "Add_Duration_ms": add_duration_ms,
                "Add_Batch_Count": len(add_batch_results),
                "Add_Batch_Size": 1,
                "Add_Batch_Results": add_batch_results,
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
                memory_system=memory_system,
                session_item=session_item,
                user_id=user_id,
                top_k=top_k,
                system_prompt=system_prompt,
                overwrite_existing_answers=overwrite_existing_answers,
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
        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(current_stage_total_cost, "a_mem_answer_generation")
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_answered_question_count, answered_session_count, final_cost, user_id, runtime_summary, observable_token_cost_summary
    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_A_Mem_Eval failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_A_Mem_Eval(
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
        all_items = load_jsonl_items(input_jsonl_path)
        selected_items = all_items[start_idx:end_idx] if end_idx is not None else all_items[start_idx:]
        output_items = []

        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing A-MEM persona {absolute_idx + 1}/{len(all_items)}")
            (
                updated_chain,
                total_answered_question_count,
                answered_session_count,
                final_cost,
                user_id,
                runtime_summary,
                observable_token_cost_summary,
            ) = Generate_Single_Persona_A_Mem_Eval(
                persona_item=persona_item,
                system_prompt=system_prompt,
                top_k=top_k,
                version=version,
                overwrite_existing_answers=overwrite_existing_answers,
            )

            result_item = Build_Compact_A_Mem_Result_Item(
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
            write_jsonl_items(output_jsonl_path, output_items)

            output_dir = os.path.dirname(output_json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as outfile:
                json.dump(output_items, outfile, ensure_ascii=False, indent=2)

        print(f"[DEBUG] A-MEM evaluation completed for {len(output_items)} personas")
    except Exception as e:
        print(f"Error processing A-MEM evaluation: {e}:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A-MEM evaluation on persona dataset")
    parser.add_argument("--input_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "..", "Data", "Step4_4_short_interval.jsonl"))
    parser.add_argument("--output_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "a_mem_results.jsonl"))
    parser.add_argument("--output_json_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "a_mem_results.json"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--overwrite_existing_answers", action="store_true")
    args = parser.parse_args()

    Generate_User_A_Mem_Eval(
        input_jsonl_path=args.input_jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        output_json_path=args.output_json_path,
        system_prompt=A_MEM_ANSWER_SYSTEM_PROMPT,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=args.version,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )

