import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(CURRENT_DIR, "llm_caller.log")

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in [None, ""]:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _get_env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.getenv(name)
    if value in [None, ""]:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _get_client() -> Any:
    if OpenAI is None:
        raise ImportError("openai is not installed in the current environment.")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def calculate_cumulative_cost(previous_cost: Optional[Dict[str, Any]], current_cost: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "current_stage": current_cost,
        "cumulative": {
            "input_tokens": current_cost.get("input_tokens", 0) or 0,
            "output_tokens": current_cost.get("output_tokens", 0) or 0,
            "total_tokens": current_cost.get("total_tokens", 0) or 0,
            "total_cost_usd": current_cost.get("total_cost_usd", 0.0) or 0.0,
        },
    }

    if previous_cost and isinstance(previous_cost, dict):
        if "cumulative" in previous_cost:
            prev_cumulative = previous_cost.get("cumulative") or {}
        elif "current_stage" in previous_cost:
            prev_cumulative = previous_cost.get("current_stage") or {}
        else:
            prev_cumulative = previous_cost

        result["cumulative"]["input_tokens"] += prev_cumulative.get("input_tokens", 0) or 0
        result["cumulative"]["output_tokens"] += prev_cumulative.get("output_tokens", 0) or 0
        result["cumulative"]["total_tokens"] += prev_cumulative.get("total_tokens", 0) or 0
        result["cumulative"]["total_cost_usd"] += prev_cumulative.get("total_cost_usd", 0.0) or 0.0

    result["cumulative"]["total_cost_usd"] = round(result["cumulative"]["total_cost_usd"], 6)
    return result


def _extract_json_from_content(content: str, markers: List[str]) -> Dict[str, Any]:
    json_content = content.strip()

    for marker in markers:
        if marker in json_content:
            parts = json_content.split(marker, 1)
            if len(parts) > 1:
                json_content = parts[1].strip()
                break

    if "```json" in json_content:
        start_idx = json_content.find("```json") + 7
        end_idx = json_content.rfind("```")
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()
    elif "```" in json_content:
        start_idx = json_content.find("```") + 3
        end_idx = json_content.rfind("```")
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()

    if "{" in json_content and "}" in json_content:
        start_idx = json_content.find("{")
        end_idx = json_content.rfind("}") + 1
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()

    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        logger.warning("Direct JSON parse failed, trying regex extraction.")

    match = re.search(r"({[\s\S]*})", json_content)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Failed to parse JSON from content: {json_content[:200]}...")


def _calculate_cost(model: str, usage: Optional[Any]) -> Dict[str, Any]:
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model": model,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "pricing_available": False,
            "note": "Usage information not available",
        }

    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    input_tokens = usage_dict.get("prompt_tokens", 0) or 0
    output_tokens = usage_dict.get("completion_tokens", 0) or 0

    cost_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model": model,
    }

    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        cost_info.update({
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing_available": True,
        })
    else:
        cost_info.update({
            "input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
            "pricing_available": False,
            "note": f"Pricing not available for model: {model}",
        })

    return cost_info


RETRY_TIMES = _get_env_int("RETRY_TIMES", 5)
WAIT_TIME_LOWER = _get_env_int("WAIT_TIME_LOWER", 1)
WAIT_TIME_UPPER = _get_env_int("WAIT_TIME_UPPER", 8)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_request(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: int = 300,
    return_parsed_json: bool = False,
    extract_json: bool = False,
    json_markers: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if return_parsed_json:
        extract_json = True

    final_model = model or os.getenv("OPENAI_MODEL")
    if not final_model:
        raise ValueError("OPENAI_MODEL is not set and no model parameter was provided.")

    client = _get_client()

    request_params: Dict[str, Any] = {
        "model": final_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": timeout,
    }

    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        env_max_tokens = os.getenv("OPENAI_MAX_TOKENS")
        if env_max_tokens not in [None, ""]:
            try:
                resolved_max_tokens = int(env_max_tokens)
            except Exception:
                resolved_max_tokens = None
    if resolved_max_tokens is not None:
        request_params["max_tokens"] = resolved_max_tokens

    resolved_temperature = temperature
    if resolved_temperature is None:
        resolved_temperature = _get_env_float("OPENAI_TEMPERATURE", None)
    if resolved_temperature is not None:
        request_params["temperature"] = resolved_temperature

    response = client.chat.completions.create(**request_params)
    content = (response.choices[0].message.content or "").strip()
    cost_info = _calculate_cost(final_model, response.usage)

    if not extract_json:
        return content, cost_info

    if json_markers is None:
        json_markers = [
            "Final JSON",
            "JSON Result",
            "Evaluation Result",
            "Result",
            "Corrected JSON",
            "Complete JSON",
        ]

    parsed_json = _extract_json_from_content(content, json_markers)
    if return_parsed_json:
        return parsed_json, cost_info
    return content, cost_info
