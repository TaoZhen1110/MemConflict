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
    from letta_client import Letta
except ImportError:
    Letta = None

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

LETTA_MODEL_CONFIG = os.getenv("LETTA_MODEL_CONFIG", "openai/gpt-4o-mini")
LETTA_EMBEDDING_CONFIG = os.getenv("LETTA_EMBEDDING_CONFIG", "openai/text-embedding-3-small")

LETTA_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

LETTA_ADD_MAX_RETRIES = 3
LETTA_ADD_RETRY_SECONDS = 5
LETTA_SEARCH_MAX_RETRIES = 3
LETTA_SEARCH_RETRY_SECONDS = 3
LETTA_POST_ADD_BUFFER_SECONDS = 1
MAX_STORED_RETRIEVED_MEMORIES = 5


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


def Build_Letta_User_ID(persona_item: Dict[str, Any], version: str) -> str:
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


def Setup_Letta_Client() -> Any:
    if Letta is None:
        raise ImportError("letta_client package is not available. Please install letta-client first.")
    api_key = os.getenv("LETTA_API_KEY")
    if not api_key:
        raise ValueError("LETTA_API_KEY is not set in the environment.")
    return Letta(api_key=api_key)


def chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def Setup_Letta_Agent(client: Any, user_id: str, persona_item: Dict[str, Any]) -> str:
    persona_hint = str(persona_item.get("persona_info", persona_item.get("persona", ""))).strip()
    agent = client.agents.create(
        model=LETTA_MODEL_CONFIG,
        embedding=LETTA_EMBEDDING_CONFIG,
        memory_blocks=[
            {"label": "human", "value": f"Identifier: {user_id}. Persona: {persona_hint}"},
            {"label": "persona", "value": "I am an empathetic AI evaluator assistant."},
        ],
    )
    return str(getattr(agent, "id", agent))


def Delete_Letta_Agent(client: Any, agent_id: Optional[str]):
    if not agent_id:
        return
    try:
        client.agents.delete(agent_id=agent_id)
    except Exception as e:
        print(f"[DEBUG] Letta agent cleanup failed: {e}")


def Add_Session_Dialogue_To_Letta(client: Any, agent_id: str, dialogue_messages: List[Dict[str, Any]], batch_size: int = 8) -> Tuple[float, List[Dict[str, Any]]]:
    if len(dialogue_messages) == 0:
        return 0.0, []
    total_duration_ms = 0.0
    batch_results = []
    for batch_index, start_idx in enumerate(range(0, len(dialogue_messages), batch_size), start=1):
        batch_dialogue = dialogue_messages[start_idx:start_idx + batch_size]
        batch_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in batch_dialogue)
        for segment_index, segment in enumerate(chunk_text(batch_text, max_chars=8000), start=1):
            start_time = time.time()
            client.agents.passages.create(agent_id=agent_id, text=segment)
            duration_ms = (time.time() - start_time) * 1000
            total_duration_ms += duration_ms
            batch_results.append({
                "Batch_Index": batch_index,
                "Segment_Index": segment_index,
                "Batch_Size": len(batch_dialogue),
                "Start_Message_Index": start_idx,
                "End_Message_Index": start_idx + len(batch_dialogue) - 1,
                "Add_Duration_ms": duration_ms,
            })
    return total_duration_ms, batch_results


def Search_Letta(client: Any, agent_id: str, query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    start_time = time.time()
    search_response = client.agents.passages.search(agent_id=agent_id, query=str(query))
    duration_ms = (time.time() - start_time) * 1000
    raw_results = getattr(search_response, "results", [])
    retrieved_memories = []
    for idx, item in enumerate(raw_results[:top_k], start=1):
        content = getattr(item, "content", getattr(item, "text", str(item)))
        retrieved_memories.append({"memory": str(content), "source": "passages.search", "rank": idx})
    lines = [f"Memories for agent {agent_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for item in retrieved_memories:
            lines.append(f"{item.get('rank')}. {item.get('memory')} [{item.get('source')}]")
    context_text = "\n".join(lines)
    return context_text, retrieved_memories, duration_ms


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], agent_id: str) -> str:
    lines = [f"Memories for agent {agent_id}:"]
    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for item in retrieved_memories:
            lines.append(f"{item.get('rank')}. {item.get('memory')} [{item.get('source')}]")
    return "\n".join(lines)


def Wait_For_Letta_Retrieval_Ready(client: Any, agent_id: str, probe_queries: List[str], top_k: int, max_retries: int = 10, poll_interval_seconds: int = 2) -> Dict[str, Any]:
    if len(probe_queries) == 0:
        return {"status": "SKIPPED", "attempt_count": 0, "probe_queries": [], "matched_query": None, "matched_count": 0}
    last_probe_result = {"status": "PENDING", "matched_query": None, "matched_count": 0}
    for attempt_idx in range(max_retries):
        for probe_query in probe_queries:
            try:
                _, retrieved_memories, _ = Search_Letta(client, agent_id, probe_query, max(1, min(top_k, 3)))
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
        print(f"[DEBUG] Letta retrieval readiness poll {attempt_idx + 1}/{max_retries}: status={last_probe_result.get('status')}")
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


def Answer_Questions_For_One_Session(client: Any, agent_id: str, session_item: Dict[str, Any], top_k: int, system_prompt: str, overwrite_existing_answers: bool) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "Letta retrieval and question answering"}
    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = "letta"
    session_memory_metadata["Top_K"] = top_k
    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue
        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue
        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        _, retrieved_memories, search_duration_ms = Search_Letta(client, agent_id, question_text, search_top_k)
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], agent_id)
        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(system_prompt, answer_context_text, question_text)
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = "letta"
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


def Build_Compact_Letta_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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


def Build_Compact_Letta_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
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
        compact_session["Session_Questions"] = [Build_Compact_Letta_Question(question_item, keep_top_k) for question_item in session_questions if isinstance(question_item, dict)]
    return compact_session


def Build_Compact_Letta_Result_Item(persona_item: Dict[str, Any], updated_chain: List[Dict[str, Any]], total_answered_question_count: int, answered_session_count: int, final_cost: Dict[str, Any], user_id: str, agent_id: str, eval_top_k: int, runtime_summary: Dict[str, Any], observable_token_cost_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ID": persona_item.get("ID"),
        "Memory_System": "letta",
        "Letta_User_ID": user_id,
        "Letta_Agent_ID": agent_id,
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "Letta_Runtime_Summary": runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [Build_Compact_Letta_Session(session_item, eval_top_k) for session_item in updated_chain],
    }


def Generate_Single_Persona_Letta_Eval(persona_item: Dict[str, Any], system_prompt: str, top_k: int, version: str, overwrite_existing_answers: bool):
    client = None
    agent_id = None
    try:
        client = Setup_Letta_Client()
        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()
        user_id = Build_Letta_User_ID(persona_item, version)
        agent_id = Setup_Letta_Agent(client, user_id, persona_item)
        total_answered_question_count = 0
        answered_session_count = 0
        persona_add_time_ms = 0.0
        persona_retrieval_time_ms = 0.0
        persona_response_time_ms = 0.0
        session_total_runtime_ms_sum = 0.0
        current_stage_total_cost = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "model": None, "pricing_available": False, "note": "Letta retrieval and question answering"}
        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for Letta")
            session_start_time = time.time()
            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            add_duration_ms, add_batch_results = Add_Session_Dialogue_To_Letta(client, agent_id, dialogue_messages)
            persona_add_time_ms += add_duration_ms
            if len(dialogue_messages) > 0:
                time.sleep(LETTA_POST_ADD_BUFFER_SECONDS)
            session_questions = session_item.get("Session_Questions", [])
            probe_queries = Build_Retrieval_Probe_Queries(dialogue_messages, session_questions)
            retrieval_ready_result = Wait_For_Letta_Retrieval_Ready(client, agent_id, probe_queries, top_k)
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": "letta",
                "Top_K": top_k,
                "Dialogue_Added_To_Memory": len(dialogue_messages) > 0,
                "Dialogue_Message_Count": len(dialogue_messages),
                "Agent_ID": agent_id,
                "Add_Duration_ms": add_duration_ms,
                "Add_Batch_Count": len(add_batch_results),
                "Add_Batch_Size": 8,
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
            updated_session_item, answered_question_count, call_cost = Answer_Questions_For_One_Session(client, agent_id, session_item, top_k, system_prompt, overwrite_existing_answers)
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
        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(current_stage_total_cost, "letta_answer_generation")
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_answered_question_count, answered_session_count, final_cost, user_id, agent_id, runtime_summary, observable_token_cost_summary
    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_Letta_Eval failed: {e}:{traceback.format_exc()}")
        raise
    finally:
        if client is not None and agent_id:
            Delete_Letta_Agent(client, agent_id)


def Generate_User_Letta_Eval(input_jsonl_path: str, output_jsonl_path: str, output_json_path: str, system_prompt: str, top_k: int, start_idx: int, end_idx: Optional[int], version: str, overwrite_existing_answers: bool):
    try:
        all_items = load_jsonl_items(input_jsonl_path)
        selected_items = all_items[start_idx:end_idx] if end_idx is not None else all_items[start_idx:]
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        output_items = []
        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing Letta persona {absolute_idx + 1}/{len(all_items)}")
            updated_chain, total_answered_question_count, answered_session_count, final_cost, user_id, agent_id, runtime_summary, observable_token_cost_summary = Generate_Single_Persona_Letta_Eval(persona_item=persona_item, system_prompt=system_prompt, top_k=top_k, version=version, overwrite_existing_answers=overwrite_existing_answers)
            result_item = Build_Compact_Letta_Result_Item(persona_item=persona_item, updated_chain=updated_chain, total_answered_question_count=total_answered_question_count, answered_session_count=answered_session_count, final_cost=final_cost, user_id=user_id, agent_id=agent_id, eval_top_k=top_k, runtime_summary=runtime_summary, observable_token_cost_summary=observable_token_cost_summary)
            output_items.append(result_item)
            with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                for output_item in output_items:
                    outfile.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump(output_items, outfile, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Letta evaluation completed for {len(output_items)} personas")
    except Exception as e:
        print(f"Error processing Letta evaluation: {e}:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Letta evaluation on persona dataset")
    parser.add_argument("--input_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "..", "Data", "Step4_4_long_interval.jsonl"))
    parser.add_argument("--output_jsonl_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "letta_results.jsonl"))
    parser.add_argument("--output_json_path", type=str, default=os.path.join(CURRENT_DIR, "Results", "letta_results.json"))
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

    Generate_User_Letta_Eval(
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

