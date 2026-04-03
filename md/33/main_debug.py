"""
第33课：新闻摘要 TL;DR 代理（透明调试版）

不使用任何框架，纯手写搜索-筛选-摘要流水线。
让你看清自适应搜索循环和多级流水线的内部机制。
"""

import os
import re
import sys
import json
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            return content
        except Exception as e:
            print(f"    !! LLM调用失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 模拟新闻数据 =====
def get_simulated_news(search_round: int) -> list:
    today = datetime.now().strftime("%Y-%m-%d")
    all_news = [
        {"title": "OpenAI发布GPT-5：多模态能力大幅提升", "url": "https://example.com/gpt5",
         "description": "OpenAI推出GPT-5，推理和多模态能力大幅突破。", "source": "科技日报",
         "text": "OpenAI今日正式发布了GPT-5大语言模型。该模型在多个基准测试中取得了显著提升，特别是在复杂推理任务上，准确率提高了30%。GPT-5支持原生多模态输入，包括文本、图像、音频和视频。在代码生成方面，GPT-5能够理解完整的项目上下文，生成更准确的代码。"},
        {"title": "谷歌Gemini 2.0发布：Agent能力成为核心亮点", "url": "https://example.com/gemini2",
         "description": "谷歌发布Gemini 2.0，首次将AI Agent能力作为核心特性。", "source": "AI头条",
         "text": "谷歌今日发布了Gemini 2.0系列模型，最大亮点是原生Agent能力。Gemini 2.0可以自主完成多步骤任务，包括网页浏览、代码执行和工具调用。谷歌还推出了Project Astra，一个基于Gemini 2.0的通用AI助手。"},
        {"title": "Anthropic发布Claude 4.6：百万级上下文窗口", "url": "https://example.com/claude46",
         "description": "Claude 4.6支持百万token上下文，Agent编码能力大幅增强。", "source": "AI研究周刊",
         "text": "Anthropic发布了Claude 4.6系列模型。最引人注目的是百万token的上下文窗口，使得Claude能够处理超长文档和大型代码库。Claude 4.6在Agent编码任务中表现出色，能够自主完成复杂的软件工程任务。"},
        {"title": "Meta开源Llama 4：性能首次接近GPT-4", "url": "https://example.com/llama4",
         "description": "Llama 4开源发布，性能接近商业模型水平。", "source": "开源中国",
         "text": "Meta今日开源了Llama 4系列大语言模型。Llama 4拥有最高4050亿参数，在多个基准测试中首次接近GPT-4的表现。开源社区反应热烈，认为将大大降低AI应用开发门槛。"},
        {"title": "AI Agent框架大战：LangGraph vs CrewAI vs AutoGen", "url": "https://example.com/frameworks",
         "description": "深度对比三大主流AI Agent框架。", "source": "技术评论",
         "text": "三大框架LangGraph、CrewAI和AutoGen展开激烈竞争。LangGraph以图结构灵活性见长，CrewAI以角色扮演为核心，AutoGen以群聊模式著称。分析师认为未来框架将趋向融合。"}
    ]
    if search_round <= 1:
        return all_news[:3]
    else:
        return all_news[3:]


def main():
    print("=" * 70)
    print("  新闻摘要 TL;DR 代理（透明调试版 - 无框架）")
    print("=" * 70)

    query = "今天最新的AI大模型新闻有哪些？"
    num_articles_tldr = 3
    max_searches = 3

    print(f"  查询: {query}")
    print(f"  目标摘要数: {num_articles_tldr}")
    print(f"  最大搜索次数: {max_searches}")

    # 状态
    potential_articles = []
    scraped_urls = []
    past_searches = []
    search_round = 0

    # ==============================================================
    # 自适应搜索循环（对应 LangGraph 的条件边循环）
    # ==============================================================
    while search_round < max_searches:

        # 阶段1：生成搜索关键词
        print(f"\n{'=' * 70}")
        print(f"【阶段1】生成搜索参数（第{search_round + 1}轮）")
        print(f"{'=' * 70}")

        search_prompt = (
            f"根据用户查询生成新闻搜索关键词。\n"
            f"用户查询: {query}\n"
            f"已搜索过: {past_searches}\n"
            f"剩余搜索次数: {max_searches - search_round}\n"
            f"请生成 1-3 个简洁的中文搜索关键词，用逗号分隔。"
            f"如果已搜索过，请尝试放宽或变换关键词。"
            f"只输出关键词，不要其他内容。"
        )

        print(f"\n  >>> 发送给 LLM <<<")
        print(f"  {'-' * 60}")
        print(f"  {search_prompt}")
        print(f"  {'-' * 60}")

        keywords = call_llm([{"role": "user", "content": search_prompt}], temperature=0.3)
        print(f"\n  >>> LLM 响应: {keywords}")

        past_searches.append(keywords)
        search_round += 1

        # 阶段2：检索文章
        print(f"\n{'=' * 70}")
        print(f"【阶段2】检索新闻文章（模拟数据）")
        print(f"{'=' * 70}")

        articles = get_simulated_news(search_round)
        new_articles = [a for a in articles if a["url"] not in scraped_urls]
        for a in new_articles:
            scraped_urls.append(a["url"])
        potential_articles.extend(new_articles)

        print(f"  本轮新增: {len(new_articles)} 篇")
        print(f"  累计候选: {len(potential_articles)} 篇")
        for a in new_articles:
            print(f"    - {a['title']}")

        # 路由决策（对应 articles_text_decision）
        print(f"\n{'=' * 70}")
        print(f"【路由决策】文章数量够不够？")
        print(f"{'=' * 70}")
        print(f"  候选: {len(potential_articles)}, 需要: {num_articles_tldr}, 剩余搜索: {max_searches - search_round}")

        if len(potential_articles) >= num_articles_tldr:
            print(f"  → 文章足够，退出搜索循环")
            break
        elif search_round >= max_searches:
            print(f"  → 搜索次数用尽，用已有文章继续")
            break
        else:
            print(f"  → 文章不够，放宽条件重搜（这就是 LangGraph 的自适应循环边）")

    # ==============================================================
    # 阶段3：LLM 选出最相关的 Top-N
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段3】LLM 选出 Top-{num_articles_tldr} 最相关文章")
    print(f"{'=' * 70}")

    articles_desc = "\n".join([
        f"[{i+1}] {a['title']} - {a['description']}"
        for i, a in enumerate(potential_articles)
    ])

    select_prompt = (
        f"根据用户查询选出最相关的 {num_articles_tldr} 篇文章。\n"
        f"用户查询: {query}\n\n"
        f"候选文章:\n{articles_desc}\n\n"
        f"只输出被选中文章的编号（如: 1,3,5），不要其他内容。"
    )

    print(f"\n  >>> 发送给 LLM <<<")
    print(f"  {'-' * 60}")
    print(f"  {select_prompt}")
    print(f"  {'-' * 60}")

    result = call_llm([{"role": "user", "content": select_prompt}], temperature=0)
    print(f"\n  >>> LLM 响应: {result}")

    numbers = re.findall(r'\d+', result)
    selected = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < len(potential_articles)]
    selected = selected[:num_articles_tldr]
    tldr_articles = [potential_articles[i] for i in selected]

    print(f"  选中 {len(tldr_articles)} 篇:")
    for a in tldr_articles:
        print(f"    - {a['title']}")

    # ==============================================================
    # 阶段4：LLM 逐篇生成摘要
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段4】LLM 逐篇生成 TL;DR 摘要")
    print(f"{'=' * 70}")

    for i, article in enumerate(tldr_articles):
        print(f"\n  --- 摘要第 {i+1} 篇: {article['title']} ---")

        summary_prompt = (
            f"请为以下新闻文章生成简洁的 TL;DR 摘要（3-5个要点）。\n\n"
            f"标题: {article['title']}\n"
            f"来源: {article.get('source', '未知')}\n"
            f"全文: {article['text']}\n\n"
            f"请用中文输出，每个要点以 * 开头。"
        )

        print(f"\n  >>> 发送给 LLM: 生成摘要...")

        summary = call_llm([{"role": "user", "content": summary_prompt}])
        tldr_articles[i]["summary"] = summary

        print(f"\n  >>> LLM 响应:")
        print(f"  {summary}")

    # ==============================================================
    # 阶段5：格式化输出
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段5】格式化最终结果")
    print(f"{'=' * 70}")

    output = f"搜索关键词: {', '.join(past_searches)}\n"
    output += f"共搜索 {search_round} 轮，候选 {len(potential_articles)} 篇，精选 {len(tldr_articles)} 篇\n"
    output += "=" * 50 + "\n\n"

    for article in tldr_articles:
        output += f"## {article['title']}\n"
        output += f"来源: {article.get('source', '未知')} | 链接: {article['url']}\n\n"
        output += f"{article.get('summary', '暂无摘要')}\n\n"
        output += "-" * 50 + "\n\n"

    print(output)

    # 调试总结
    print(f"{'=' * 70}")
    print(f"  调试总结：LangGraph 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. Graph（非 StateGraph）定义了 6 个节点的流水线")
    print(f"  2. 自适应循环: retrieve_articles → [够不够？] → 不够就回到 generate_params")
    print(f"     本质就是 while 循环 + if 判断")
    print(f"  3. 结构化输出: 用 Pydantic + JsonOutputParser 让 LLM 生成标准 API 参数")
    print(f"     本质就是在 Prompt 里描述格式 + 解析 JSON")
    print(f"  4. 去重机制: past_searches 记录已搜关键词，scraped_urls 记录已爬URL")
    print(f"     本质就是两个 list/set")
    print(f"  5. 多级流水线: 搜索→爬取→筛选→摘要→格式化")
    print(f"     每一级都是一个函数，LangGraph 用边把它们串起来")


if __name__ == "__main__":
    main()
