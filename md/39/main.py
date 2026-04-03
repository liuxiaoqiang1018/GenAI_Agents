"""
第39课：汽车购买助手代理（LangGraph 框架版）

架构：多轮对话收集需求 → LLM构建搜索条件 → 搜索车辆 → 用户选择 → 获取详情
核心模式：交互式多轮对话 + LLM参数构建 + 三路循环（选车/改条件/结束）
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, List, Dict, Optional
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
def call_llm(messages: list, temperature: float = 0.5) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"无choices: {str(data)[:200]}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("LLM返回空内容")
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# ===== 模拟车辆数据 =====
MOCK_LISTINGS = [
    {"id": "1", "title": "比亚迪宋PLUS DM-i 2024款", "price": "15.98万", "mileage": "0.5万公里",
     "year": "2024", "fuel": "插电混动", "location": "北京",
     "detail": "1.5L插混系统，纯电续航110km，综合油耗3.8L/100km。全景天窗、360全景影像、DiLink智能座舱。空间宽敞，适合家庭出行。口碑：油耗低、空间大、智能化好。常见问题：悬挂偏硬、高速NVH一般。"},
    {"id": "2", "title": "特斯拉Model Y 2024款", "price": "24.99万", "mileage": "0.3万公里",
     "year": "2024", "fuel": "纯电", "location": "上海",
     "detail": "后驱标准版，续航554km，0-100km/h 6.9秒。Autopilot辅助驾驶、15英寸中控屏。OTA持续升级。口碑：科技感强、加速快、保值率高。常见问题：做工粗糙、悬挂偏硬、冬季续航衰减。"},
    {"id": "3", "title": "丰田RAV4荣放 2024款", "price": "17.58万", "mileage": "1.2万公里",
     "year": "2024", "fuel": "汽油", "location": "广州",
     "detail": "2.0L自然吸气，CVT变速箱，四驱可选。Toyota Safety Sense智行安全。皮实耐用，保值率高。口碑：可靠性强、保养便宜、通过性好。常见问题：动力一般、内饰设计老气、车机不够智能。"},
    {"id": "4", "title": "小鹏G6 2024款", "price": "20.99万", "mileage": "0.8万公里",
     "year": "2024", "fuel": "纯电", "location": "深圳",
     "detail": "后驱580标准版，续航580km，800V超快充。XNGP智驾系统、全景天幕。口碑：智驾领先、充电快、性价比高。常见问题：品牌认知度低、售后网点少。"},
    {"id": "5", "title": "本田CR-V 2024款 e:PHEV", "price": "22.59万", "mileage": "0.6万公里",
     "year": "2024", "fuel": "插电混动", "location": "成都",
     "detail": "2.0L插混系统，纯电续航82km。Honda SENSING 360安全系统。做工扎实，驾驶质感好。口碑：品质可靠、驾控好、保值率高。常见问题：价格偏高、纯电续航短。"}
]


# ===== 状态 =====
class State(TypedDict):
    user_needs: str          # 用户需求总结
    listings: List[dict]     # 搜索到的车辆列表
    selected_car: dict       # 用户选中的车辆
    next_node: str           # 路由目标
    conversation: List[str]  # 对话历史摘要


# ===== 节点函数 =====

def ask_user_needs(state: State) -> dict:
    """多轮对话收集用户需求"""
    print("\n" + "=" * 60)
    print("【询问需求】汽车购买助手")
    print("=" * 60)

    existing_needs = state.get("user_needs", "")

    if existing_needs:
        system_msg = (
            f"你是一个汽车购买助手。已知用户需求：\n{existing_needs}\n\n"
            f"用户想修改或补充需求，请询问需要调整什么。用中文，简洁友好。"
        )
    else:
        system_msg = (
            "你是一个汽车购买助手。请热情地介绍自己，然后询问用户的购车需求，"
            "包括：用途（通勤/家庭/越野）、预算、偏好的车型（轿车/SUV/MPV）、"
            "动力类型（燃油/混动/纯电）、其他特殊要求。用中文，简洁友好。"
        )

    greeting = call_llm([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "开始对话"}
    ])
    print(f"\n  助手: {greeting}")

    user_input = input("\n  你的回答: ").strip()
    if not user_input:
        user_input = "预算20万以内，家庭用SUV，偏好混动或纯电，有智能驾驶更好"
        print(f"  (使用默认输入: {user_input})")

    # LLM 总结需求并判断下一步
    classify_prompt = (
        f"根据用户回答总结购车需求，并判断下一步。\n\n"
        f"{'已有需求: ' + existing_needs if existing_needs else ''}\n"
        f"用户最新回答: {user_input}\n\n"
        f"请输出两行:\n"
        f"第一行: 需求总结（简洁的要点列表）\n"
        f"第二行: 下一步（只写 search 或 more_info）\n"
        f"  - search: 需求已明确，可以开始搜索\n"
        f"  - more_info: 需要更多信息"
    )

    result = call_llm([{"role": "user", "content": classify_prompt}], temperature=0.1)
    lines = result.strip().split("\n")

    needs = lines[0] if lines else user_input
    next_step = "search_listings"
    for line in lines:
        if "more_info" in line.lower():
            next_step = "ask_user_needs"
            break
        if "search" in line.lower():
            next_step = "search_listings"
            break

    print(f"\n  需求总结: {needs}")
    print(f"  下一步: {next_step}")

    return {**state, "user_needs": needs, "next_node": next_step}


def search_listings(state: State) -> dict:
    """根据需求搜索车辆列表"""
    print("\n" + "=" * 60)
    print("【搜索车辆】根据需求匹配")
    print("=" * 60)

    # LLM 从模拟数据中筛选匹配的车辆
    listings_json = json.dumps([{k: v for k, v in car.items() if k != "detail"} for car in MOCK_LISTINGS], ensure_ascii=False)

    prompt = (
        f"根据用户需求从以下车辆中选出最匹配的3辆，返回它们的id列表。\n\n"
        f"用户需求: {state['user_needs']}\n\n"
        f"车辆列表:\n{listings_json}\n\n"
        f"只输出id，用逗号分隔，如: 1,3,5"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0.1)
    ids = re.findall(r'\d+', result)
    matched = [car for car in MOCK_LISTINGS if car["id"] in ids]

    if not matched:
        matched = MOCK_LISTINGS[:3]

    print(f"  匹配到 {len(matched)} 辆车:")
    for i, car in enumerate(matched, 1):
        print(f"    {i}. {car['title']} - {car['price']} ({car['fuel']})")

    # 用户选择
    print(f"\n  请选择操作:")
    print(f"    输入编号(1-{len(matched)}) 查看详情")
    print(f"    输入 r 修改条件重搜")
    print(f"    输入 q 结束")

    choice = input("\n  你的选择: ").strip()
    if not choice:
        choice = "1"
        print(f"  (默认选择: {choice})")

    if choice.lower() == "q":
        return {**state, "listings": matched, "next_node": "end"}
    elif choice.lower() == "r":
        return {**state, "listings": matched, "next_node": "ask_user_needs"}
    else:
        idx = int(choice) - 1 if choice.isdigit() and 0 < int(choice) <= len(matched) else 0
        selected = matched[idx]
        print(f"  选中: {selected['title']}")
        return {**state, "listings": matched, "selected_car": selected, "next_node": "fetch_detail"}


def fetch_detail(state: State) -> dict:
    """获取选中车辆的详细信息"""
    print("\n" + "=" * 60)
    print("【车辆详情】获取详细信息和口碑")
    print("=" * 60)

    car = state["selected_car"]
    print(f"  车辆: {car['title']}")

    # LLM 生成详细分析
    prompt = (
        f"你是一位专业的汽车顾问。请根据以下信息为用户提供购买建议。\n\n"
        f"车辆: {car['title']}\n"
        f"价格: {car['price']}\n"
        f"详情: {car.get('detail', '暂无')}\n\n"
        f"用户需求: {state['user_needs']}\n\n"
        f"请提供:\n1. 车辆亮点\n2. 潜在问题\n3. 是否符合用户需求\n4. 购买建议\n"
        f"用中文回答。"
    )

    analysis = call_llm([{"role": "user", "content": prompt}])
    print(f"\n  分析报告:\n{analysis}")

    # 继续选择
    print(f"\n  输入 r 重新搜索，输入 q 结束")
    choice = input("  你的选择: ").strip()
    next_node = "ask_user_needs" if choice.lower() == "r" else "end"

    return {**state, "next_node": next_node}


# ===== 路由 =====
def route_after_needs(state: State) -> str:
    return state.get("next_node", "search_listings")

def route_after_search(state: State) -> str:
    node = state.get("next_node", "end")
    if node == "end":
        return END
    return node

def route_after_detail(state: State) -> str:
    node = state.get("next_node", "end")
    if node == "end":
        return END
    return node


# ===== 构建工作流 =====
def build_workflow():
    builder = StateGraph(State)

    builder.add_node("ask_user_needs", ask_user_needs)
    builder.add_node("search_listings", search_listings)
    builder.add_node("fetch_detail", fetch_detail)

    builder.set_entry_point("ask_user_needs")

    builder.add_conditional_edges("ask_user_needs", route_after_needs,
                                  {"search_listings": "search_listings", "ask_user_needs": "ask_user_needs"})
    builder.add_conditional_edges("search_listings", route_after_search,
                                  {"fetch_detail": "fetch_detail", "ask_user_needs": "ask_user_needs", END: END})
    builder.add_conditional_edges("fetch_detail", route_after_detail,
                                  {"ask_user_needs": "ask_user_needs", END: END})

    return builder.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  汽车购买助手（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()
    result = app.invoke({
        "user_needs": "",
        "listings": [],
        "selected_car": {},
        "next_node": "",
        "conversation": []
    })

    print(f"\n{'=' * 60}")
    print("  对话结束，感谢使用汽车购买助手！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
