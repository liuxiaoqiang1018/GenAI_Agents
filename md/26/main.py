"""
第26课 - 搜索摘要工具（框架版）

核心概念：
  - 搜索作为数据源：先获取外部信息，再用LLM加工
  - 逐条摘要：每条搜索结果独立摘要，保留来源
  - Map-Reduce：搜索→逐条摘要(Map)→合并(Reduce)

说明：因无搜索API，用LLM模拟搜索结果。生产中替换为DuckDuckGo/Tavily等。
"""

import os
import json
import re
import time
from typing import List, Dict

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.5},
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


def extract_json(text: str):
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        return {}


# ========== 搜索函数 ==========

def search(query: str, num_results: int = 5) -> List[Dict]:
    """
    搜索函数。用LLM模拟搜索结果。
    生产环境中替换为:
      from langchain_community.tools import DuckDuckGoSearchResults
      results = DuckDuckGoSearchResults().invoke(query)
    """
    print(f'>>> 搜索: {query}')
    print(f'>>> （模拟搜索 — 生产中用 DuckDuckGo/Tavily API）')

    system = (f"你是一个搜索引擎模拟器。为以下查询生成{num_results}条模拟搜索结果。\n"
              f"每条结果包含真实可信的信息（基于你的知识）。\n"
              f"返回JSON：{{\"results\": [\n"
              f"  {{\"title\": \"标题\", \"snippet\": \"内容摘要(50-100字)\", \"source\": \"来源网站\"}}\n"
              f"]}}")

    result = extract_json(call_llm(query, system))
    results = result.get("results", [])

    if not results:
        results = [{"title": "默认结果", "snippet": f"关于{query}的信息", "source": "未知"}]

    return results


# ========== 摘要函数 ==========

def summarize_result(result: Dict, query: str) -> str:
    """对单条搜索结果进行摘要"""
    system = ("你是摘要专家。将以下搜索结果浓缩为1-2句话的精炼摘要。\n"
              "保留核心信息，去除冗余。用中文。")
    prompt = f"查询: {query}\n标题: {result['title']}\n内容: {result['snippet']}"
    return call_llm(prompt, system)


# ========== 合并函数 ==========

def combine_summaries(summaries: List[Dict], query: str) -> str:
    """合并所有摘要为最终报告"""
    report_parts = [f"# 搜索摘要报告：{query}\n"]

    for i, item in enumerate(summaries, 1):
        report_parts.append(f"**{i}. {item['title']}**（{item['source']}）")
        report_parts.append(f"  {item['summary']}\n")

    return "\n".join(report_parts)


# ========== 主流程 ==========

def search_and_summarize(query: str) -> str:
    """搜索→解析→逐条摘要→合并"""

    # 1. 搜索
    print()
    print('=' * 60)
    print('【1 - 搜索】')
    print('=' * 60)
    results = search(query)
    print(f'>>> 获得 {len(results)} 条结果')
    for r in results:
        print(f'    - {r["title"]}（{r.get("source", "")}）')

    # 2. 逐条摘要（Map）
    print()
    print('=' * 60)
    print(f'【2 - 逐条摘要】（Map: {len(results)} 条）')
    print('=' * 60)

    summaries = []
    for i, result in enumerate(results, 1):
        print(f'    [{i}/{len(results)}] 摘要: {result["title"][:30]}...')
        summary = summarize_result(result, query)
        summaries.append({
            "title": result["title"],
            "source": result.get("source", "未知"),
            "summary": summary,
        })
        print(f'    >>> {summary[:80]}...')

    # 3. 合并（Reduce）
    print()
    print('=' * 60)
    print('【3 - 合并报告】（Reduce）')
    print('=' * 60)

    report = combine_summaries(summaries, query)
    print(report)

    return report


# ========== 运行 ==========

if __name__ == '__main__':
    print('第26课 - 搜索摘要工具')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    examples = [
        "人工智能最新进展2024",
        "Python和Java的性能对比",
        "大语言模型的应用场景",
    ]

    print('示例查询:')
    for i, q in enumerate(examples, 1):
        print(f'  {i}. {q}')
    print()

    query = input('搜索查询（回车用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    result = search_and_summarize(query)

    print()
    print('#' * 60)
    print('#  搜索摘要完成')
    print('#' * 60)
