"""
第33课：新闻摘要 TL;DR 代理（LangGraph 框架版）

架构：搜索参数生成 → 新闻检索 → 文章筛选 → LLM摘要 → 格式化输出
      如果文章不够，自动放宽条件重新搜索（自适应循环）

核心模式：搜索-筛选-摘要流水线 + 自适应重试 + 结构化输出
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, Annotated, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

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
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 模拟新闻数据（不依赖 NewsAPI）=====
def get_simulated_news(query: str, search_round: int) -> list:
    """模拟 NewsAPI 返回的新闻文章"""
    today = datetime.now().strftime("%Y-%m-%d")

    all_news = [
        {
            "title": "OpenAI发布GPT-5：多模态能力大幅提升",
            "url": "https://example.com/gpt5-release",
            "description": "OpenAI宣布推出GPT-5模型，在推理、多模态和代码生成方面取得重大突破。",
            "source": "科技日报",
            "publishedAt": today,
            "text": "OpenAI今日正式发布了GPT-5大语言模型。该模型在多个基准测试中取得了显著提升，特别是在复杂推理任务上，准确率提高了30%。GPT-5支持原生多模态输入，包括文本、图像、音频和视频。在代码生成方面，GPT-5能够理解完整的项目上下文，生成更准确的代码。OpenAI CEO表示，GPT-5代表了向通用人工智能迈出的重要一步。业内专家认为，GPT-5的发布将推动AI在医疗、教育和科研领域的应用。"
        },
        {
            "title": "谷歌Gemini 2.0发布：Agent能力成为核心亮点",
            "url": "https://example.com/gemini-2",
            "description": "谷歌发布Gemini 2.0，首次将AI Agent能力作为核心特性推出。",
            "source": "人工智能头条",
            "publishedAt": today,
            "text": "谷歌今日发布了Gemini 2.0系列模型，最大亮点是原生Agent能力。Gemini 2.0可以自主完成多步骤任务，包括网页浏览、代码执行和工具调用。谷歌还推出了Project Astra，这是一个基于Gemini 2.0的通用AI助手，能够通过摄像头理解周围环境并提供帮助。在基准测试中，Gemini 2.0在数学推理和代码生成方面超过了GPT-4。谷歌计划将Agent能力整合到搜索、Gmail和Google Docs等产品中。"
        },
        {
            "title": "Anthropic发布Claude 4.6：百万级上下文窗口",
            "url": "https://example.com/claude-4-6",
            "description": "Anthropic推出Claude 4.6，支持百万token上下文窗口，Agent编码能力大幅增强。",
            "source": "AI研究周刊",
            "publishedAt": today,
            "text": "Anthropic发布了Claude 4.6系列模型，包括Opus、Sonnet和Haiku三个版本。最引人注目的是百万token的上下文窗口，使得Claude能够处理超长文档和大型代码库。Claude 4.6在Agent编码任务中表现出色，能够自主完成复杂的软件工程任务。Anthropic还推出了Claude Code CLI工具，允许开发者在终端中直接与Claude协作。安全性方面，Claude 4.6引入了更强的对齐机制，在保持能力的同时降低了有害输出的风险。"
        },
        {
            "title": "Meta开源Llama 4：开源模型首次接近商业模型水平",
            "url": "https://example.com/llama-4",
            "description": "Meta发布开源大模型Llama 4，性能首次接近GPT-4级别。",
            "source": "开源中国",
            "publishedAt": today,
            "text": "Meta今日开源了Llama 4系列大语言模型。Llama 4拥有最高4050亿参数，在多个基准测试中首次接近甚至超过GPT-4的表现。Meta表示，Llama 4的训练数据量是Llama 3的三倍，并引入了全新的混合专家(MoE)架构。开源社区对此反应热烈，认为这将大大降低AI应用的开发门槛。多家云服务商已宣布将支持Llama 4的部署。"
        },
        {
            "title": "AI Agent框架大战：LangGraph vs CrewAI vs AutoGen对比分析",
            "url": "https://example.com/agent-framework-comparison",
            "description": "深度对比三大主流AI Agent框架的优劣势和适用场景。",
            "source": "技术评论",
            "publishedAt": today,
            "text": "随着AI Agent成为行业热点，三大框架LangGraph、CrewAI和AutoGen展开了激烈竞争。LangGraph以其图结构的灵活性和状态管理能力见长，适合复杂工作流。CrewAI以角色扮演和任务委派为核心，适合多Agent协作场景。AutoGen则以群聊模式和代码执行能力著称。分析师认为，未来Agent框架将趋向融合，单一框架难以满足所有需求。开发者应根据具体场景选择合适的框架。"
        }
    ]

    # 第一轮返回前3条，第二轮返回后2条（模拟放宽搜索）
    if search_round <= 1:
        return all_news[:3]
    else:
        return all_news[3:]


# ===== 状态定义 =====
class GraphState(TypedDict):
    news_query: str                    # 用户查询
    num_searches_remaining: int        # 剩余搜索次数
    search_round: int                  # 当前搜索轮次
    past_searches: List[str]           # 已搜索过的关键词
    scraped_urls: List[str]            # 已爬取的URL
    num_articles_tldr: int             # 需要摘要的文章数
    potential_articles: List[dict]     # 候选文章（含全文）
    tldr_articles: List[dict]          # 被选中的文章（含摘要）
    formatted_results: str             # 最终格式化结果


# ===== 节点函数 =====

def generate_search_params(state: GraphState) -> GraphState:
    """节点1：LLM 生成搜索关键词"""
    print("\n" + "=" * 60)
    print("【节点1】生成搜索参数")
    print("=" * 60)

    past = state["past_searches"]
    remaining = state["num_searches_remaining"]

    prompt = (
        f"根据用户查询生成新闻搜索关键词。\n"
        f"用户查询: {state['news_query']}\n"
        f"已搜索过: {past}\n"
        f"剩余搜索次数: {remaining}\n"
        f"请生成 1-3 个简洁的中文搜索关键词，用逗号分隔。"
        f"如果已搜索过，请尝试放宽或变换关键词。"
        f"只输出关键词，不要其他内容。"
    )

    keywords = call_llm([{"role": "user", "content": prompt}], temperature=0.3)
    print(f"  生成的搜索关键词: {keywords}")

    state["past_searches"].append(keywords)
    state["num_searches_remaining"] -= 1
    state["search_round"] += 1

    return state


def retrieve_articles(state: GraphState) -> GraphState:
    """节点2：检索新闻文章（使用模拟数据）"""
    print("\n" + "=" * 60)
    print("【节点2】检索新闻文章")
    print("=" * 60)

    articles = get_simulated_news(state["news_query"], state["search_round"])

    # 去重
    new_articles = []
    for article in articles:
        if article["url"] not in state["scraped_urls"]:
            new_articles.append(article)
            state["scraped_urls"].append(article["url"])

    state["potential_articles"].extend(new_articles)
    print(f"  本轮新增: {len(new_articles)} 篇")
    print(f"  累计候选: {len(state['potential_articles'])} 篇")
    for a in new_articles:
        print(f"    - {a['title']} ({a['source']})")

    return state


def select_top_articles(state: GraphState) -> GraphState:
    """节点3：LLM 选出最相关的 Top-N 文章"""
    print("\n" + "=" * 60)
    print("【节点3】LLM 选出最相关文章")
    print("=" * 60)

    num = state["num_articles_tldr"]
    articles_desc = "\n".join([
        f"[{i+1}] {a['title']} - {a['description']}"
        for i, a in enumerate(state["potential_articles"])
    ])

    prompt = (
        f"根据用户查询选出最相关的 {num} 篇文章。\n"
        f"用户查询: {state['news_query']}\n\n"
        f"候选文章:\n{articles_desc}\n\n"
        f"只输出被选中文章的编号（如: 1,3,5），不要其他内容。"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0)

    # 解析编号
    numbers = re.findall(r'\d+', result)
    selected_indices = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < len(state["potential_articles"])]
    selected_indices = selected_indices[:num]

    state["tldr_articles"] = [state["potential_articles"][i] for i in selected_indices]
    print(f"  LLM 选择了 {len(state['tldr_articles'])} 篇:")
    for a in state["tldr_articles"]:
        print(f"    - {a['title']}")

    return state


def summarize_articles(state: GraphState) -> GraphState:
    """节点4：LLM 逐篇生成 TL;DR 摘要"""
    print("\n" + "=" * 60)
    print("【节点4】LLM 生成文章摘要")
    print("=" * 60)

    for i, article in enumerate(state["tldr_articles"]):
        print(f"\n  摘要第 {i+1} 篇: {article['title']}")

        prompt = (
            f"请为以下新闻文章生成简洁的 TL;DR 摘要（3-5个要点）。\n\n"
            f"标题: {article['title']}\n"
            f"来源: {article.get('source', '未知')}\n"
            f"全文: {article['text']}\n\n"
            f"请用中文输出，每个要点以 * 开头。"
        )

        summary = call_llm([{"role": "user", "content": prompt}])
        state["tldr_articles"][i]["summary"] = summary
        print(f"  {summary[:150]}...")

    return state


def format_results(state: GraphState) -> GraphState:
    """节点5：格式化最终输出"""
    print("\n" + "=" * 60)
    print("【节点5】格式化输出")
    print("=" * 60)

    search_terms = ", ".join(state["past_searches"])
    result = f"搜索关键词: {search_terms}\n"
    result += f"共搜索 {state['search_round']} 轮，候选 {len(state['potential_articles'])} 篇，精选 {len(state['tldr_articles'])} 篇\n"
    result += "=" * 50 + "\n\n"

    for article in state["tldr_articles"]:
        result += f"## {article['title']}\n"
        result += f"来源: {article.get('source', '未知')} | 链接: {article['url']}\n\n"
        result += f"{article.get('summary', '暂无摘要')}\n\n"
        result += "-" * 50 + "\n\n"

    state["formatted_results"] = result
    return state


# ===== 路由函数 =====
def articles_decision(state: GraphState) -> str:
    """自适应决策：文章够不够？"""
    total = len(state["potential_articles"])
    needed = state["num_articles_tldr"]
    remaining = state["num_searches_remaining"]

    print(f"\n  [路由决策] 候选文章: {total}, 需要: {needed}, 剩余搜索: {remaining}")

    if remaining <= 0:
        if total == 0:
            print(f"  → 搜索次数用尽且无文章 → 结束")
            return "END"
        else:
            print(f"  → 搜索次数用尽，用已有文章 → 选择Top-N")
            return "select_top_articles"
    else:
        if total < needed:
            print(f"  → 文章不够({total}<{needed})，放宽条件重搜")
            return "generate_search_params"
        else:
            print(f"  → 文章足够 → 选择Top-N")
            return "select_top_articles"


# ===== 构建工作流 =====
def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.set_entry_point("generate_search_params")

    workflow.add_node("generate_search_params", generate_search_params)
    workflow.add_node("retrieve_articles", retrieve_articles)
    workflow.add_node("select_top_articles", select_top_articles)
    workflow.add_node("summarize_articles", summarize_articles)
    workflow.add_node("format_results", format_results)

    workflow.add_edge("generate_search_params", "retrieve_articles")
    workflow.add_conditional_edges(
        "retrieve_articles",
        articles_decision,
        {
            "generate_search_params": "generate_search_params",
            "select_top_articles": "select_top_articles",
            "END": END
        }
    )
    workflow.add_edge("select_top_articles", "summarize_articles")
    workflow.add_edge("summarize_articles", "format_results")
    workflow.add_edge("format_results", END)

    return workflow.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  新闻摘要 TL;DR 代理（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()

    query = "今天最新的AI大模型新闻有哪些？"
    print(f"  查询: {query}")
    print(f"  摘要数量: 3 篇")

    initial_state = {
        "news_query": query,
        "num_searches_remaining": 3,
        "search_round": 0,
        "past_searches": [],
        "scraped_urls": [],
        "num_articles_tldr": 3,
        "potential_articles": [],
        "tldr_articles": [],
        "formatted_results": ""
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("【最终结果】")
    print("=" * 60)
    print(result["formatted_results"])


if __name__ == "__main__":
    main()
