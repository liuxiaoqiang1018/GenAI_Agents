"""
第26课 - 搜索摘要工具内部机制

目的：让你看清搜索+摘要的本质：
  1. 搜索 = 调外部API获取数据（本课用LLM模拟）
  2. 解析 = 把搜索结果变成结构化数据
  3. 逐条摘要 = for result in results: call_llm("摘要: " + result)
  4. 合并 = 字符串拼接

这就是 RAG（检索增强生成）的简化版：
  - RAG: 向量数据库检索 → LLM生成
  - 本课: 搜索引擎检索 → LLM摘要
  原理完全一样，只是检索方式不同。
"""

import os
import json
import re
import time
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


# ================================================================
#  完整流程
# ================================================================

def search_and_summarize(query: str) -> str:
    """
    搜索摘要的完整流程。

    Java 类比：
        @Service
        public class SearchSummarizeService {
            @Autowired SearchEngine searchEngine;  // DuckDuckGo/Google
            @Autowired LLMService llm;

            public String searchAndSummarize(String query) {
                // 1. 搜索（外部API）
                List<SearchResult> results = searchEngine.search(query);

                // 2. 逐条摘要（Map）
                List<String> summaries = results.stream()
                    .map(r -> llm.summarize(r.getSnippet()))
                    .collect(toList());

                // 3. 合并（Reduce）
                return String.join("\n", summaries);
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：搜索（获取外部数据）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：搜索】')
    print('=' * 60)
    print(f'>>> 查询: {query}')
    print(f'>>> 生产中: results = DuckDuckGoSearchResults().invoke(query)')
    print(f'>>> 本课: 用LLM模拟搜索结果')

    system = ("模拟搜索引擎。生成5条搜索结果。\n"
              "返回JSON：{\"results\": [{\"title\": \"标题\", \"snippet\": \"内容(50-100字)\", \"source\": \"来源\"}]}")
    result = extract_json(call_llm(query, system))
    results = result.get("results", [{"title": "默认", "snippet": query, "source": "未知"}])
    total_llm_calls += 1

    print(f'>>> 获得 {len(results)} 条结果')

    # ==========================================
    # 第2步：逐条摘要（Map）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第2步：逐条摘要 — Map】')
    print(f'    for result in results: summary = call_llm("摘要: " + result)')
    print('=' * 60)

    summaries = []
    for i, r in enumerate(results, 1):
        print(f'\n    [{i}/{len(results)}] {r["title"]}')
        print(f'    原文: {r["snippet"][:60]}...')

        summary = call_llm(
            f"标题: {r['title']}\n内容: {r['snippet']}",
            "浓缩为1-2句精炼摘要。用中文。"
        )
        total_llm_calls += 1

        summaries.append({"title": r["title"], "source": r.get("source", ""), "summary": summary})
        print(f'    摘要: {summary[:80]}...')

    # ==========================================
    # 第3步：合并（Reduce）
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：合并 — Reduce】')
    print(f'    就是 "\\n".join(summaries)')
    print('=' * 60)

    report = f"# 搜索摘要：{query}\n\n"
    for i, s in enumerate(summaries, 1):
        report += f"**{i}. {s['title']}**（{s['source']}）\n"
        report += f"  {s['summary']}\n\n"

    print(report)
    print(f'>>> LLM调用: {total_llm_calls}次（搜索1 + 摘要{len(results)}）')

    return report


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第26课 - 搜索摘要工具（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 搜索 = 调外部API获取数据')
    print('  - 逐条摘要 = for + call_llm（Map）')
    print('  - 合并 = 字符串拼接（Reduce）')
    print('  - 这就是 RAG 的简化版')
    print()

    examples = [
        "人工智能最新进展2024",
        "大语言模型的应用场景",
    ]

    print('示例:')
    for i, q in enumerate(examples, 1):
        print(f'  {i}. {q}')
    print()

    query = input('搜索查询（回车用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    search_and_summarize(query)
