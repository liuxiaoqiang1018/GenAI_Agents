"""
第19课 - 多平台内容生成代理内部机制（不使用 LangGraph）

目的：让你看清线性+并行+汇聚的本质：
  1. 摘要 = call_llm(文章)
  2. 调研 = call_llm(摘要)
  3. 意图 = 用户选的平台列表
  4. 并行 = for platform in platforms: call_llm(摘要+调研, platform_prompt)
  5. 合并 = "\n".join(results)

对比 main.py（LangGraph 框架版），理解：
  - Send(4个平台) → 就是 for 循环（或 parallelStream）
  - operator.add → 就是 results.append()
  - 线性→并行→汇聚 → 就是 3步前处理 + 1个for循环 + 拼接
"""

import os
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
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
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


# ========== 平台prompt配置 ==========

PLATFORM_PROMPTS = {
    "微博": ("你是微博运营专家。生成一条微博：\n"
             "1. 正文不超过140字\n"
             "2. 加2-3个话题标签（#标签#）\n"
             "3. 简洁有力\n用中文。"),
    "公众号": ("你是公众号编辑。生成公众号文章：\n"
               "1. 吸引人的标题\n"
               "2. 分段+小标题\n"
               "3. 结尾引导互动\n用中文。"),
    "小红书": ("你是小红书博主。生成小红书笔记：\n"
               "1. 吸睛标题（可用emoji）\n"
               "2. 分享体验式写法\n"
               "3. 结尾加10个标签\n用中文。"),
    "知乎": ("你是知乎优质答主。生成知乎回答：\n"
             "1. 先给结论\n"
             "2. 分点论述有理有据\n"
             "3. 专业但不枯燥\n用中文。"),
}


# ========== 示例文章 ==========

EXAMPLE_ARTICLE = """
大语言模型（LLM）正在深刻改变软件开发的方式。GitHub Copilot、Cursor等AI编程助手已经成为
许多开发者的日常工具。最新数据显示，使用AI辅助编程的开发者效率提升了30%-50%。

然而，AI编程并不意味着"不需要程序员"。相反，AI更像是一个强大的副驾驶——它能快速生成代码、
解释错误、提供建议，但最终的架构设计、代码审查和质量把控仍然需要人类开发者来完成。

对于程序员来说，学会与AI协作而非被AI替代，是未来最重要的技能之一。
"""


# ================================================================
#  完整流程
# ================================================================

def generate_content(article: str, platforms: list) -> dict:
    """
    多平台内容生成的完整流程。

    Java 类比：
        @Service
        public class ContentIntelligenceService {
            public Map<String, String> generate(String article, List<String> platforms) {
                // 线性前处理
                String summary = summarizer.summarize(article);
                String research = researcher.research(summary);

                // 并行分发（就是 parallelStream + 不同的策略）
                Map<String, String> contents = platforms.parallelStream()
                    .collect(toMap(
                        p -> p,
                        p -> platformAdapters.get(p).generate(summary, research)
                    ));

                // 合并
                return contents;
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：文本摘要（线性）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：文本摘要】（LLM调用 #1）')
    print('=' * 60)

    summary = call_llm(article, "提取核心要点，200字以内摘要。用中文。")
    total_llm_calls += 1
    print(f'>>> 摘要: {summary[:150]}...')

    # ==========================================
    # 第2步：内容调研（线性）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：内容调研】（LLM调用 #2）')
    print('=' * 60)

    research = call_llm(
        f"文章摘要：{summary}\n补充行业趋势、热点和可引用案例。",
        "你是行业研究分析师。用中文。"
    )
    total_llm_calls += 1
    print(f'>>> 调研: {research[:150]}...')

    # ==========================================
    # 第3步：意图匹配（不调LLM）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第3步：意图匹配】目标平台: {platforms}')
    print('=' * 60)

    # ==========================================
    # 第4步：并行生成（Send的本质：for循环）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第4步：平台内容生成 — 并行】')
    print(f'    Send() 的本质：for platform in platforms:')
    print('=' * 60)

    contents = {}  # operator.add 的本质：往字典/列表里存

    for platform in platforms:
        if platform not in PLATFORM_PROMPTS:
            print(f'    ✗ {platform}: 不支持的平台，跳过')
            continue

        print(f'    [{platform}] 生成中...')

        content = call_llm(
            f"摘要：{summary}\n调研：{research}",
            PLATFORM_PROMPTS[platform]
        )
        total_llm_calls += 1

        contents[platform] = content
        print(f'    [{platform}] 完成（{len(content)}字）')

    # ==========================================
    # 第5步：合并（Reduce）
    # ==========================================
    print()
    print('=' * 60)
    print('【第5步：内容合并 — 汇聚】')
    print('=' * 60)

    final_parts = []
    for platform, content in contents.items():
        final_parts.append(f"【{platform}】\n{content}")

    final_output = "\n\n" + "=" * 40 + "\n\n".join(final_parts)
    print(f'>>> 合并了 {len(contents)} 个平台的内容')
    print(f'>>> LLM调用: {total_llm_calls}次（摘要1 + 调研1 + 平台{len(contents)}）')

    return {
        "summary": summary,
        "research": research,
        "contents": contents,
        "final_output": final_output,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第19课 - 多平台内容生成代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 线性前处理 = 2次LLM调用（摘要+调研）')
    print('  - 并行生成 = for platform in platforms: call_llm(...)')
    print('  - 汇聚合并 = 把所有结果拼在一起')
    print('  - 平台适配 = 同一数据，不同system prompt')
    print()

    # 获取文章
    print('请输入文章（回车使用默认示例）:')
    first_line = input().strip()
    if not first_line:
        article = EXAMPLE_ARTICLE
        print('>>> 使用默认示例')
    else:
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        article = '\n'.join(lines)

    # 选择平台
    print('\n可选: 微博、公众号、小红书、知乎')
    p_input = input('选择平台（逗号分隔，回车全部）: ').strip()
    if p_input:
        platforms = [p.strip() for p in p_input.split(',')]
    else:
        platforms = ["微博", "公众号", "小红书", "知乎"]
        print(f'>>> 全部: {platforms}')

    result = generate_content(article, platforms)

    print()
    print('#' * 60)
    print(f'#  生成完成！LLM调用 {result["total_llm_calls"]} 次')
    print('#' * 60)
    print(result["final_output"])
