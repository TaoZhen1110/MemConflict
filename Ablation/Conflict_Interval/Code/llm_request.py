import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)
from openai import OpenAI

# 配置日志
log_file = 'MemConflict/llm_caller.log'
if os.path.exists(log_file):
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('')  # 清空文件内容

logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file,  # 输出到文件
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 从第一个代码示例中引入的部分 ---

# 模型定价字典（每百万 token 的费用，美元）
MODEL_PRICING = {
    'gpt-4o': {
        'input': 2.50,
        'output': 10.00
    },
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.60
    },
    'gpt-5-mini': {
        'input': 0.25,
        'output': 2.00
    }
}


def calculate_cumulative_cost(previous_cost: Optional[Dict], current_cost: Dict) -> Dict:
    """
    计算累计成本
    
    Args:
        previous_cost: 之前阶段的成本信息字典
        current_cost: 当前阶段的成本信息字典
        
    Returns:
        包含当前和累计成本的字典
    """
    result = {
        "current_stage": current_cost,
        "cumulative": {
            "input_tokens": current_cost.get("input_tokens", 0) or 0,
            "output_tokens": current_cost.get("output_tokens", 0) or 0,
            "total_tokens": current_cost.get("total_tokens", 0) or 0,
            "total_cost_usd": current_cost.get("total_cost_usd", 0) or 0
        }
    }
    
    # 如果有之前的成本信息，进行累加
    if previous_cost and isinstance(previous_cost, dict):
        # 检查是否有累计信息
        if "cumulative" in previous_cost:
            prev_cumulative = previous_cost["cumulative"]
        elif "current_stage" in previous_cost:
            # 如果之前只有当前阶段信息，将其作为累计基础
            prev_cumulative = previous_cost["current_stage"]
        else:
            # 直接使用之前的成本信息
            prev_cumulative = previous_cost
        
        # 累加 token 数量和费用
        if prev_cumulative:
            result["cumulative"]["input_tokens"] += prev_cumulative.get("input_tokens", 0) or 0
            result["cumulative"]["output_tokens"] += prev_cumulative.get("output_tokens", 0) or 0
            result["cumulative"]["total_tokens"] += prev_cumulative.get("total_tokens", 0) or 0
            result["cumulative"]["total_cost_usd"] += prev_cumulative.get("total_cost_usd", 0) or 0
    
    # 四舍五入费用到6位小数
    if result["cumulative"]["total_cost_usd"]:
        result["cumulative"]["total_cost_usd"] = round(result["cumulative"]["total_cost_usd"], 6)
    
    return result


def _extract_json_from_content(content: str, markers: List[str]) -> Dict:
    """从内容中提取 JSON，覆盖了多种复杂情况"""
    json_content = content
    
    # 1. 查找标记后的内容
    for marker in markers:
        if marker in json_content:
            parts = json_content.split(marker, 1)
            if len(parts) > 1:
                json_content = parts[1].strip()
                break
    
    # 2. 处理可能的 markdown 代码块
    if '```json' in json_content:
        start_idx = json_content.find('```json') + 7
        end_idx = json_content.rfind('```')
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()
    elif '```' in json_content:
        start_idx = json_content.find('```') + 3
        end_idx = json_content.rfind('```')
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()
    
    # 3. 尝试找到 JSON 对象的开始和结束
    if '{' in json_content and '}' in json_content:
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}') + 1
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()
    
    try:
        parsed_json = json.loads(json_content)
        logger.debug("Successfully parsed JSON")
        return parsed_json
    except json.JSONDecodeError as e:
        # 4. 如果直接解析失败，使用正则表达式尝试提取
        logger.warning(f"JSON parsing error: {e}, trying regex extraction")
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, json_content)
        if match:
            try:
                potential_json = match.group(1)
                parsed_json = json.loads(potential_json)
                logger.debug("Successfully extracted JSON through regex")
                return parsed_json
            except json.JSONDecodeError:
                logger.warning("Content extracted through regex is not valid JSON")
        
        # 如果所有尝试都失败，抛出异常，这将触发 tenacity 重试
        raise ValueError(f"Failed to parse JSON from content: {json_content[:200]}...")


def _calculate_cost(model: str, usage: Optional[Any]) -> Dict:
    """计算 token 消耗的费用，并返回标准格式的字典"""
    if usage is None:
        logger.warning("No usage information available in API response")
        return {
            "input_tokens": None, "output_tokens": None, "total_tokens": None, "model": model,
            "input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None,
            "pricing_available": False, "note": "Usage information not available"
        }
    
    # 将 Pydantic 模型转换为字典
    usage_dict = usage.model_dump()
    input_tokens = usage_dict.get('prompt_tokens', 0)
    output_tokens = usage_dict.get('completion_tokens', 0)
    
    cost_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model": model
    }
    
    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        cost_info.update({
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing_available": True
        })
    else:
        cost_info.update({
            "input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None,
            "pricing_available": False, "note": f"Pricing not available for model: {model}"
        })
    
    return cost_info

# --- 修改后的主调用函数 ---

load_dotenv()

# 重试配置
RETRY_TIMES = int(os.getenv('RETRY_TIMES'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER'))

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=os.getenv('OPENAI_BASE_URL'),
    api_key=os.getenv('OPENAI_API_KEY')
)


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
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    timeout: int = 300,
    return_parsed_json: bool = False,
    extract_json: bool = True,
    json_markers: Optional[List[str]] = None
) -> tuple:
    """
    调用 OpenAI ChatCompletion API，返回结果和成本信息。

    - 网络异常 / 429 / 5xx / JSON解析失败 → 自动重试
    
    参数:
        system_prompt: 系统提示
        user_prompt: 用户提示
        model: 模型名称，比如 "gpt-4o-mini"
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        timeout: 请求超时
        return_parsed_json: 如果为 True，元组的第一个元素返回解析后的 dict；否则返回 content 字符串
        extract_json: 是否尝试提取 JSON（False 时直接返回原始内容）
        json_markers: 用于定位 JSON 的文本标记列表，仅在 return_parsed_json=True 时生效
    
    返回:
        一个元组 (result, cost_info)，其中：
        - result: 如果 return_parsed_json=True，为解析后的 JSON 字典；否则为原始文本字符串。
        - cost_info: 包含详细 token 和费用信息的字典。
    """
    
    # 确定要使用的模型
    final_model = model or os.getenv('OPENAI_MODEL')
    if not final_model:
        raise ValueError("Model name must be provided either as a parameter or in the OPENAI_MODEL environment variable.")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Build request parameters dynamically based on what's available
    request_params = {
        "model": final_model,
        "messages": messages
    }
    
    # Add optional parameters only if they exist in environment or are explicitly provided
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    elif os.getenv('OPENAI_MAX_TOKENS'):
        request_params["max_tokens"] = int(os.getenv('OPENAI_MAX_TOKENS'))
    
    if temperature is not None:
        request_params["temperature"] = temperature
    elif os.getenv('OPENAI_TEMPERATURE'):
        request_params["temperature"] = float(os.getenv('OPENAI_TEMPERATURE'))
    
    if timeout is not None:
        request_params["timeout"] = timeout
    elif os.getenv('OPENAI_TIMEOUT'):
        request_params["timeout"] = int(os.getenv('OPENAI_TIMEOUT'))

        
    response = client.chat.completions.create(**request_params)

    # 1. 提取内容
    content = response.choices[0].message.content.strip()
    logger.debug(f"[DEBUG] API response content: {content}...")

    # 2. 计算成本
    cost_info = _calculate_cost(final_model, response.usage)

    if not extract_json:
        return content, cost_info
    
    # 定义默认的 JSON 标记
    if json_markers is None:
        json_markers = [
            "Corrected Profile", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result",
            "Dialogue Generation Result", "Generated Dialogue", "JSON Result",
            "Generation Result"
        ]
    # 解析 JSON，如果失败会抛出 ValueError，从而触发 tenacity 重试
    parsed_json = _extract_json_from_content(content, json_markers)
    
    # 3. 根据需要解析 JSON 或直接返回内容
    if return_parsed_json:
        return parsed_json, cost_info
    else:
        return content, cost_info


if __name__ == "__main__":
    # 使用前请确保设置了以下环境变量，或在代码中直接赋值
    # os.environ["OPENAI_BASE_URL"] = "YOUR_BASE_URL"
    # os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
    # os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    
    # --- 测试场景1: 获取纯文本和成本 ---
    print("\n--- Test 1: Get raw text and cost ---")
    try:
        user_prompt_1 = "你好，请用中文简单介绍一下自己。"
        text_result, cost_info_1 = llm_request(
            messages=[{'role': 'user', 'content': user_prompt_1}],
        )
        print("Text Result:", text_result)
        print("Cost Info:", json.dumps(cost_info_1, indent=2))
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # --- 测试场景2: 获取解析后的 JSON 和成本 ---
    print("\n--- Test 2: Get parsed JSON and cost ---")
    try:
        # 这个 prompt 故意让模型在 JSON 前后添加额外文本，以测试解析能力
        user_prompt_2 = """
        请生成一个 JSON 数据示例。
        
        下面是最终的JSON结果:
        ```json
        {
          "user_id": 12345,
          "username": "test_user",
          "is_active": true,
          "roles": ["editor", "viewer"]
        }
        ```
        这就是我生成的全部内容。
        """
        json_result, cost_info_2 = llm_request(
            messages=[{'role': 'user', 'content': user_prompt_2}],
            return_parsed_json=True
        )
        print("Parsed JSON Result:", json_result)
        print("Is a dictionary:", isinstance(json_result, dict))
        print("Cost Info:", json.dumps(cost_info_2, indent=2))
    except Exception as e:
        print(f"Test 2 failed: {e}")