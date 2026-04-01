"""
第25课 - 任务导向工具调用代理（框架版）

核心概念：
  - 函数→工具：给Python函数加名称+描述+参数，变成LLM可调用的工具
  - Agent循环：LLM决策→执行工具→观察结果→继续决策→...→最终回答
  - Tool Calling：LLM不执行操作，只输出调用指令

本课不使用LangGraph，手工实现工具调用Agent的完整循环。
"""

import os
import json
import re
import time
from typing import Dict, Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(messages: list) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not choices[0].get("message"):
                raise ValueError("API返回空响应")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


def call_llm_simple(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return call_llm(messages)


# ========== 工具函数（普通Python函数） ==========

def summarize(text: str) -> str:
    """对文本进行摘要"""
    return call_llm_simple(text, "你是摘要专家。用中文对以下文本进行简洁摘要，保留核心信息。")


def translate(text: str, target_lang: str = "英语") -> str:
    """翻译文本"""
    return call_llm_simple(text, f"你是翻译专家。把以下文本翻译成{target_lang}。只返回翻译结果。")


def word_count(text: str) -> str:
    """统计字数"""
    count = len(text)
    return f"文本共 {count} 个字符"


def extract_keywords(text: str) -> str:
    """提取关键词"""
    return call_llm_simple(text, "提取以下文本的5个核心关键词，用逗号分隔。")


# ========== 工具注册表 ==========

TOOLS = {
    "summarize": {
        "func": summarize,
        "description": "对文本进行摘要",
        "params": "text（要摘要的文本）",
    },
    "translate": {
        "func": translate,
        "description": "翻译文本到指定语言",
        "params": "text（要翻译的文本），target_lang（目标语言，默认英语）",
    },
    "word_count": {
        "func": word_count,
        "description": "统计文本字数",
        "params": "text（要统计的文本）",
    },
    "extract_keywords": {
        "func": extract_keywords,
        "description": "提取文本关键词",
        "params": "text（要提取关键词的文本）",
    },
}


def get_tools_description() -> str:
    """生成工具列表描述，供LLM参考"""
    desc = "可用工具：\n"
    for name, info in TOOLS.items():
        desc += f"  - {name}: {info['description']}，参数: {info['params']}\n"
    return desc


# ========== Agent 循环 ==========

def run_agent(task: str, max_steps: int = 5) -> str:
    """
    工具调用Agent的核心循环。

    流程：LLM决策 → 执行工具 → 观察结果 → LLM继续决策 → ... → 最终回答
    """

    tools_desc = get_tools_description()

    system = (
        "你是一个任务执行Agent。你可以调用工具来完成用户任务。\n\n"
        f"{tools_desc}\n"
        "执行流程：\n"
        "1. 分析任务需要调用哪些工具\n"
        "2. 每次输出一个工具调用，格式：\n"
        '   TOOL_CALL: {"tool": "工具名", "args": {"text": "参数值"}}\n'
        "3. 观察工具返回结果后，决定下一步\n"
        "4. 所有工具调用完成后，输出最终结果：\n"
        "   FINAL_ANSWER: 最终答案\n\n"
        "注意：每次只调用一个工具。"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"任务：{task}"},
    ]

    print()
    print('#' * 60)
    print(f'#  Agent 开始执行任务')
    print(f'#  任务: {task[:50]}...')
    print('#' * 60)

    for step in range(1, max_steps + 1):
        print()
        print(f'--- 第{step}步: LLM决策 ---')

        response = call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        # 检查是否是最终回答
        if "FINAL_ANSWER:" in response:
            final = response.split("FINAL_ANSWER:")[-1].strip()
            print(f'>>> 最终回答: {final[:200]}')
            return final

        # 解析工具调用
        tool_match = re.search(r'TOOL_CALL:\s*(\{.*?\})', response, re.DOTALL)
        if tool_match:
            try:
                tool_call = json.loads(tool_match.group(1))
                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("args", {})

                print(f'>>> 调用工具: {tool_name}({tool_args})')

                if tool_name in TOOLS:
                    # 执行工具
                    func = TOOLS[tool_name]["func"]
                    if tool_name == "translate" and "target_lang" in tool_args:
                        result = func(tool_args.get("text", ""), tool_args.get("target_lang", "英语"))
                    else:
                        result = func(tool_args.get("text", ""))

                    print(f'>>> 工具返回: {result[:150]}...')

                    # 把结果反馈给LLM
                    messages.append({"role": "user", "content": f"工具 {tool_name} 返回结果：{result}"})
                else:
                    messages.append({"role": "user", "content": f"错误：工具 {tool_name} 不存在"})

            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "错误：工具调用格式不正确，请重试"})
        else:
            # LLM没有调用工具也没有给最终答案，提醒它
            messages.append({"role": "user", "content": "请调用工具或给出最终答案（FINAL_ANSWER:）"})

    return "达到最大步数限制，未能完成任务。"


# ========== 运行 ==========

if __name__ == '__main__':
    print('第25课 - 任务导向工具调用代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print(get_tools_description())

    examples = [
        "请摘要以下文字并翻译成英语：人工智能正在改变软件开发的方式。AI编程助手已经成为许多开发者的日常工具，数据显示使用AI辅助编程的开发者效率提升了30%到50%。",
        "帮我提取关键词并统计字数：大语言模型的核心技术包括Transformer架构、注意力机制、预训练和微调，这些技术使得AI能够理解和生成人类语言。",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex[:60]}...')
    print()

    task = input('请输入任务（回车用示例1）: ').strip()
    if not task:
        task = examples[0]
        print(f'>>> 使用: {task[:60]}...')

    result = run_agent(task)

    print()
    print('=' * 60)
    print('【最终结果】')
    print('=' * 60)
    print(result)
