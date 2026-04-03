"""
第35课：新闻记者 AI 助手（LangGraph 框架版）

架构：意图分类 �� 多路条件路由 → 5个分析模块 → Markdown报告
模块：摘要 / 事实核查 / 语气分析 / 引语提取 / 语法偏见审查

核心模式：意图分类 + 5路条件路由 + Map-Reduce摘要
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, List, Optional
from datetime import datetime
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
def call_llm(messages: list, temperature: float = 0.3) -> str:
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


# ===== 模拟文章 =====
SAMPLE_ARTICLE = """
人工智能正在重塑全球医疗健康产业。据世界卫生组织最新报告显示，2025年��球AI医疗市场规模已达450亿美元，预计到2030年将突破1500亿美元。

谷歌DeepMind研发的AlphaFold系列模型已成功预测超过2亿种蛋白质结构，被认为是生物学领域近年来最重要的突破之一。DeepMind CEO德米斯·哈萨比斯在接受采���时表示："AlphaFold的影响才刚刚开始，未来五年内，它将彻底改变药物研发的方式。"

然而，AI在医疗领域的应用也面临挑战。北京协和医院院长张抒扬指出："AI辅助诊断系统虽然在影像识别方面表现出色，准确率高达97%���但在复杂病例的综合判断上仍需要医生的经验和直觉。"她同时强调，患者数据隐私保护是AI医疗落地的关键前提。

值得注意的是，欧盟《人工���能法案》已将医疗AI系统列为"高风险"类别，要求开发者必须提供透明的算法解释和严格的临床验证数据。美国FDA也加快了AI医疗器械的审批流程，2024年已批准超过200款AI医疗软件。

中国在AI医疗领域也取得了显著进展。百度推出的灵医大模型已在全国超过500家医院部署，覆盖影像诊断、病历生成和药物推荐等场景。腾讯觅影系统在早期癌症筛查中的准确率已接近资深放射科医生水平。

业内专家普遍认为，AI不会取代医生，而是成为医生的"超级助手"。未来的医疗模式将是"AI + 医生"的深度协作，让优质医疗资源惠及更多人群。
"""


# ===== 状态定义 =====
class State(TypedDict):
    current_query: str
    article_text: str
    actions: List[str]
    summary_result: Optional[str]
    fact_check_result: Optional[str]
    tone_analysis_result: Optional[str]
    quote_extraction_result: Optional[str]
    grammar_bias_result: Optional[str]


# ===== 意图分类 =====
def categorize_user_input(state: State) -> dict:
    """意图分类节点：LLM 识别用户要做什么"""
    print("\n" + "=" * 60)
    print("【意图分类】LLM 识别用户需求")
    print("=" * 60)
    print(f"  用户���入: {state['current_query']}")

    prompt = (
        "根据用户输入，识别需要执行的分析操作。\n"
        "可选操作: summarization, fact-checking, tone-analysis, quote-extraction, grammar-and-bias-review\n"
        "如果用户要求'全部报告'或'完整分析'，返回所有操作。\n"
        "如果输入无关，返回 invalid。\n\n"
        f"用户输入: {state['current_query']}\n\n"
        "只返回操作列表，用逗号分隔，不要其他内容。"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    actions = [a.strip().lower() for a in result.split(",")]

    # 规范化
    valid_actions = {"summarization", "fact-checking", "tone-analysis", "quote-extraction", "grammar-and-bias-review"}
    actions = [a for a in actions if a in valid_actions]

    if not actions:
        # 默认全部
        if any(kw in state["current_query"] for kw in ["全部", "完整", "报告", "full", "all"]):
            actions = list(valid_actions)
        else:
            actions = ["invalid"]

    print(f"  识别的操作: {actions}")
    return {**state, "actions": actions}


# ===== 5个分析模块 =====

def summarization_node(state: State) -> dict:
    """摘要模块"""
    print("\n" + "=" * 60)
    print("【摘要模块】生成文章摘要")
    print("=" * 60)

    prompt = (
        "请用150-200字总结以下文章，关注主要事件、关键人物和重要数据。"
        "使用中立客观的新闻报道语气。\n\n"
        f"文章:\n{state['article_text']}"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  摘要: {result[:200]}...")
    return {**state, "summary_result": result}


def fact_checking_node(state: State) -> dict:
    """事实核查模块"""
    print("\n" + "=" * 60)
    print("【事实核查模块】检查事实准确性")
    print("=" * 60)

    prompt = (
        "对以下文章进行事实核查。逐条检查关键声明的准确性。\n"
        "对每个声明标注状态: 已确认/可疑/无法验证/模糊\n"
        "并给出简短解释。用中文回答。\n\n"
        f"文章:\n{state['article_text']}"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  核查结果: {result[:200]}...")
    return {**state, "fact_check_result": result}


def tone_analysis_node(state: State) -> dict:
    """语气分析模块"""
    print("\n" + "=" * 60)
    print("【语气分析模���】分析文章语气和立场")
    print("=" * 60)

    prompt = (
        "分析以下文章的语气和立场。判断它是中立、正面、批判还是有倾向性的？\n"
        "用文章中的具体例子支持你的分析。用中文回答。\n\n"
        f"文章:\n{state['article_text']}"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  语气分析: {result[:200]}...")
    return {**state, "tone_analysis_result": result}


def quote_extraction_node(state: State) -> dict:
    """引语提取模块"""
    print("\n" + "=" * 60)
    print("【引语提取模块】提取直接引用")
    print("=" * 60)

    prompt = (
        "识别以下文章中的直接引用（引号内的内容），标注说话人和上下文。\n"
        "如果没有引用，回复'文章中未发现直接引用'。用中文回答。\n\n"
        f"文章:\n{state['article_text']}"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  引语提取: {result[:200]}...")
    return {**state, "quote_extraction_result": result}


def grammar_bias_node(state: State) -> dict:
    """语法偏见审查模块"""
    print("\n" + "=" * 60)
    print("【语法偏见审查模块】检查语法和偏见")
    print("=" * 60)

    prompt = (
        "审查以下文章的语法、拼写、标点问题，以及是否存在偏见倾向。\n"
        "列出发现的问题和改进建议。用中文回答。\n\n"
        f"文章:\n{state['article_text']}"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  审查结果: {result[:200]}...")
    return {**state, "grammar_bias_result": result}


# ===== 路由 =====
def route_actions(state: State):
    """根据 actions 列表路由到对应模块"""
    actions = state.get("actions", [])
    if "invalid" in actions or not actions:
        return END
    # 返回第一个 action 对应的节点（简化处理，逐个执行）
    return actions[0]


# ===== 构建工作流 =====
def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("categorize", categorize_user_input)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("fact-checking", fact_checking_node)
    workflow.add_node("tone-analysis", tone_analysis_node)
    workflow.add_node("quote-extraction", quote_extraction_node)
    workflow.add_node("grammar-and-bias-review", grammar_bias_node)

    workflow.set_entry_point("categorize")

    workflow.add_conditional_edges(
        "categorize",
        route_actions,
        {
            "summarization": "summarization",
            "fact-checking": "fact-checking",
            "tone-analysis": "tone-analysis",
            "quote-extraction": "quote-extraction",
            "grammar-and-bias-review": "grammar-and-bias-review",
            END: END,
        }
    )

    # 每个模块完成后链接到下一个（实现全部报告的顺序执行）
    workflow.add_edge("summarization", "fact-checking")
    workflow.add_edge("fact-checking", "tone-analysis")
    workflow.add_edge("tone-analysis", "quote-extraction")
    workflow.add_edge("quote-extraction", "grammar-and-bias-review")
    workflow.add_edge("grammar-and-bias-review", END)

    return workflow.compile()


# ===== 格式化报告 =====
def format_report(state: dict) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    report = f"# 文章分析报告 ({date})\n\n"
    report += f"**用户查询**: {state.get('current_query', '')}\n"
    report += f"**执行操作**: {', '.join(state.get('actions', []))}\n\n"

    if state.get("summary_result"):
        report += f"## 文章摘要\n\n{state['summary_result']}\n\n"
    if state.get("fact_check_result"):
        report += f"## 事实核查\n\n{state['fact_check_result']}\n\n"
    if state.get("tone_analysis_result"):
        report += f"## 语气分析\n\n{state['tone_analysis_result']}\n\n"
    if state.get("quote_extraction_result"):
        report += f"## 引语提取\n\n{state['quote_extraction_result']}\n\n"
    if state.get("grammar_bias_result"):
        report += f"## 语法与偏见审查\n\n{state['grammar_bias_result']}\n\n"

    return report


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  新闻记者 AI 助手（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()

    # 测试：全部报告
    query = "请对这篇文章进行全部分析，生���完整报告"
    print(f"\n  用户查询: {query}")
    print(f"  文章长度: {len(SAMPLE_ARTICLE)} 字")

    result = app.invoke({
        "current_query": query,
        "article_text": SAMPLE_ARTICLE,
        "actions": [],
        "summary_result": None,
        "fact_check_result": None,
        "tone_analysis_result": None,
        "quote_extraction_result": None,
        "grammar_bias_result": None,
    })

    # 格式化并保存报告
    report = format_report(result)
    report_file = os.path.join(os.path.dirname(__file__), f"文章分析报告_{datetime.now().strftime('%Y-%m-%d')}.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  报告已保存: {report_file}")

    print("\n" + "=" * 60)
    print("【完整报告】")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
