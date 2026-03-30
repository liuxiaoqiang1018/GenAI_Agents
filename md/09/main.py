"""
第9课 - 智能客服路由系统（LangGraph 框架版）

核心概念：
  - 多条件路由：情感 × 类别 = 4条路径
  - 升级模式：消极情绪直接转人工，不走 LLM
  - 分析层 + 决策层 + 处理层 分离
"""

import os
import json
import re
from typing import TypedDict

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ========== State ==========

class State(TypedDict):
    query: str          # 用户工单
    category: str       # 分类：技术/账单/一般
    sentiment: str      # 情感：积极/中性/消极
    response: str       # 回复


# ========== 节点：分类 ==========

def categorize_node(state: State):
    """分类节点：把工单归类"""
    print()
    print('=' * 50)
    print('【分类节点】')
    print('=' * 50)

    system = "你是客服分类专家。将用户工单分类为以下之一：技术、账单、一般。只返回分类名称。"
    category = call_llm(state["query"], system)

    # 标准化
    for c in ["技术", "账单", "一般"]:
        if c in category:
            category = c
            break
    else:
        category = "一般"

    print(f'>>> 工单: {state["query"][:60]}...')
    print(f'>>> 分类: {category}')
    return {"category": category}


# ========== 节点：情感分析 ==========

def analyze_sentiment_node(state: State):
    """情感分析节点"""
    print()
    print('=' * 50)
    print('【情感分析节点】')
    print('=' * 50)

    system = ("分析用户工单的情绪态度（不是事情本身好坏）。\n"
               "- 积极：表达感谢、满意\n"
               "- 中性：正常描述问题、寻求帮助\n"
               "- 消极：愤怒、抱怨、威胁、投诉\n"
               "只回复：积极、中性、消极（三选一）。")
    sentiment = call_llm(state["query"], system)

    for s in ["积极", "消极", "中性"]:
        if s in sentiment:
            sentiment = s
            break
    else:
        sentiment = "中性"

    print(f'>>> 情感: {sentiment}')
    return {"sentiment": sentiment}


# ========== 路由函数 ==========

def route_query(state: State) -> str:
    """多条件路由：情感优先于类别"""
    sentiment = state["sentiment"]
    category = state["category"]

    print()
    print('=' * 50)
    print('【路由决策】')
    print('=' * 50)
    print(f'>>> 分类={category}, 情感={sentiment}')

    if sentiment == "消极":
        print(f'>>> → 升级人工（消极情绪优先）')
        return "escalate"
    elif category == "技术":
        print(f'>>> → 技术客服')
        return "handle_technical"
    elif category == "账单":
        print(f'>>> → 账单客服')
        return "handle_billing"
    else:
        print(f'>>> → 一般客服')
        return "handle_general"


# ========== 节点：处理 ==========

def handle_technical_node(state: State):
    """技术客服"""
    print()
    print('=' * 50)
    print('【技术客服节点】')
    print('=' * 50)

    system = "你是技术客服专家。用专业但友好的语气回复用户的技术问题。用中文回答。"
    response = call_llm(state["query"], system)
    print(f'>>> 回复: {response[:200]}...')
    return {"response": response}


def handle_billing_node(state: State):
    """账单客服"""
    print()
    print('=' * 50)
    print('【账单客服节点】')
    print('=' * 50)

    system = "你是账单客服专家。用耐心、清晰的语气回复用户的账单问题。用中文回答。"
    response = call_llm(state["query"], system)
    print(f'>>> 回复: {response[:200]}...')
    return {"response": response}


def handle_general_node(state: State):
    """一般客服"""
    print()
    print('=' * 50)
    print('【一般客服节点】')
    print('=' * 50)

    system = "你是客服代表。用友好的语气回复用户的咨询。用中文回答。"
    response = call_llm(state["query"], system)
    print(f'>>> 回复: {response[:200]}...')
    return {"response": response}


def escalate_node(state: State):
    """升级人工——不调用 LLM，直接返回固定文案"""
    print()
    print('=' * 50)
    print('【升级人工节点】（不调用LLM）')
    print('=' * 50)

    response = ("非常抱歉给您带来不好的体验。您的问题已被标记为紧急工单，"
                "我们的资深客服专员将在15分钟内与您联系。请保持电话畅通。"
                f"\n\n工单摘要：{state['query'][:100]}"
                f"\n分类：{state['category']}"
                f"\n情感：{state['sentiment']}")

    print(f'>>> 升级回复: {response[:200]}')
    return {"response": response}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    # 添加节点
    workflow.add_node("categorize", categorize_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("handle_technical", handle_technical_node)
    workflow.add_node("handle_billing", handle_billing_node)
    workflow.add_node("handle_general", handle_general_node)
    workflow.add_node("escalate", escalate_node)

    # 固定边
    workflow.set_entry_point("categorize")
    workflow.add_edge("categorize", "analyze_sentiment")

    # 条件路由：4条路径
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            "handle_technical": "handle_technical",
            "handle_billing": "handle_billing",
            "handle_general": "handle_general",
            "escalate": "escalate",
        }
    )

    # 所有处理节点 → 结束
    workflow.add_edge("handle_technical", END)
    workflow.add_edge("handle_billing", END)
    workflow.add_edge("handle_general", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
        print('\n>>> 粘贴到 https://mermaid.live 查看可视化')
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 运行 ==========

def run_query(app, query: str):
    print()
    print('#' * 60)
    print(f'#  用户工单: {query}')
    print('#' * 60)

    result = app.invoke({"query": query})

    print()
    print('=' * 60)
    print('【处理结果】')
    print('=' * 60)
    print(f'  分类: {result["category"]}')
    print(f'  情感: {result["sentiment"]}')
    print(f'  回复: {result["response"]}')
    print()

    return result


if __name__ == '__main__':
    print('第9课 - 智能客服路由系统')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 示例工单（覆盖4条路径）
    examples = [
        "我的网络连接一直断断续续，怎么回事？",           # 技术 + 中性 → 技术客服
        "我被多扣了两个月的费用，请帮我查一下",            # 账单 + 中性 → 账单客服
        "请问你们的营业时间是什么时候？",                  # 一般 + 中性 → 一般客服
        "你们这破服务什么玩意！收我钱也不解决问题！我要投诉！", # 消极 → 升级人工
    ]

    print('--- 示例工单（覆盖4条路径）---')
    for q in examples:
        run_query(app, q)

    # 交互模式
    print('\n输入工单，输入 /quit 退出\n')
    while True:
        user_input = input('客户: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        run_query(app, user_input)
