"""
第11课 - 旅行规划器（LangGraph 框架版）

核心概念：
  - 交互式输入：前两个节点收集用户输入，不调用 LLM
  - 状态累积：每个节点填充一个字段，逐步丰富上下文
  - 最后一步生成：只有最终节点调用 LLM 生成行程
  - 模式：收集信息 → 生成结果
"""

import os
from typing import TypedDict, List

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
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ========== State ==========

class PlannerState(TypedDict):
    city: str             # 目标城市
    interests: List[str]  # 用户兴趣列表
    itinerary: str        # 生成的行程


# ========== 节点：输入城市 ==========

def input_city_node(state: PlannerState):
    """收集城市信息（用户交互，不调用LLM）"""
    print()
    print('=' * 50)
    print('【1 - 输入城市】')
    print('=' * 50)

    city = input('请输入你想去的城市: ').strip()
    if not city:
        city = "北京"
        print(f'>>> 未输入，默认: {city}')
    else:
        print(f'>>> 目标城市: {city}')

    return {"city": city}


# ========== 节点：输入兴趣 ==========

def input_interests_node(state: PlannerState):
    """收集兴趣信息（用户交互，不调用LLM）"""
    print()
    print('=' * 50)
    print('【2 - 输入兴趣】')
    print('=' * 50)

    print(f'你要去 {state["city"]}，请输入你的兴趣（逗号分隔）:')
    print('例如: 美食,历史,购物,自然风光')
    raw = input('你的兴趣: ').strip()

    if not raw:
        interests = ["美食", "历史"]
        print(f'>>> 未输入，默认: {interests}')
    else:
        interests = [i.strip() for i in raw.split(',') if i.strip()]
        print(f'>>> 兴趣列表: {interests}')

    return {"interests": interests}


# ========== 节点：生成行程 ==========

def create_itinerary_node(state: PlannerState):
    """调用 LLM 生成行程"""
    print()
    print('=' * 50)
    print('【3 - 生成行程】（调用LLM）')
    print('=' * 50)

    city = state["city"]
    interests = "、".join(state["interests"])

    print(f'>>> 城市: {city}')
    print(f'>>> 兴趣: {interests}')
    print(f'>>> 正在生成行程...')

    system = ("你是一个专业的旅行规划师。根据用户提供的城市和兴趣，"
              "生成一份详细的一日游行程。用中文回答。"
              "包含时间安排、具体地点、交通建议和小贴士。")
    prompt = f"请为我规划一天的{city}之旅。我的兴趣是：{interests}。"

    itinerary = call_llm(prompt, system)

    print()
    print(itinerary)

    return {"itinerary": itinerary}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(PlannerState)

    workflow.add_node("input_city", input_city_node)
    workflow.add_node("input_interests", input_interests_node)
    workflow.add_node("create_itinerary", create_itinerary_node)

    workflow.set_entry_point("input_city")

    # 纯线性：城市 → 兴趣 → 生成行程
    workflow.add_edge("input_city", "input_interests")
    workflow.add_edge("input_interests", "create_itinerary")
    workflow.add_edge("create_itinerary", END)

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

if __name__ == '__main__':
    print('第11课 - 旅行规划器')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    while True:
        print()
        print('#' * 60)
        print('#  开始规划旅行')
        print('#' * 60)

        result = app.invoke({
            "city": "",
            "interests": [],
            "itinerary": "",
        })

        print()
        print('=' * 60)
        print('【规划完成】')
        print('=' * 60)
        print(f'  城市: {result["city"]}')
        print(f'  兴趣: {", ".join(result["interests"])}')
        print()

        again = input('\n再规划一次？(y/n): ').strip().lower()
        if again != 'y':
            print('再见，祝旅途愉快！')
            break
