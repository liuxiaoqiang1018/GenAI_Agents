"""
第39课：汽车购买助手代理（透明调试版）

不使用任何框架，纯手写交互式对话+搜索+详情流程。
展示多轮对话需求收集和条件循环的内部机制。
"""

import os
import re
import sys
import json
import time
import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


def call_llm(messages: list, temperature: float = 0.5) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"无choices")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    !! LLM失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# 模拟车辆
CARS = [
    {"id": "1", "title": "比亚迪宋PLUS DM-i", "price": "15.98万", "fuel": "插混",
     "detail": "纯电续航110km，油耗3.8L，空间大，智能座舱。优点：省油空间大。缺点：悬挂硬。"},
    {"id": "2", "title": "特斯拉Model Y", "price": "24.99万", "fuel": "纯电",
     "detail": "续航554km，Autopilot，科技感强。优点：加速快保值高。缺点：做工粗糙冬季续航差。"},
    {"id": "3", "title": "丰田RAV4荣放", "price": "17.58万", "fuel": "汽油",
     "detail": "2.0L CVT，皮实耐用。优点：可靠保养便宜。缺点：动力一般内饰老气。"},
    {"id": "4", "title": "小鹏G6", "price": "20.99万", "fuel": "纯电",
     "detail": "续航580km，800V快充，XNGP智驾。优点：智驾领先充电快。缺点：品牌认知度低。"},
]


def main():
    print("=" * 70)
    print("  汽车购买助手（透明调试版 - 无框架）")
    print("=" * 70)

    user_needs = ""

    while True:
        # ==========================================================
        # 阶段1：询问需求（对应 ask_user_needs 节点）
        # ==========================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段1】询问用户需求")
        print(f"{'=' * 70}")

        if user_needs:
            system_msg = f"你是汽车购买助手。已知需求：\n{user_needs}\n用户想调整，请询问要改什么。中文简洁。"
        else:
            system_msg = "你是汽车购买助手。请介绍自己，询问用途、预算、车型偏好、动力类型。中文简洁。"

        greeting = call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "开始对话"}
        ])
        print(f"\n  助手: {greeting}")

        user_input = input("\n  你的回答（回车用默认值）: ").strip()
        if not user_input:
            user_input = "预算20万以内，家庭SUV，混动或纯电，要有智能驾驶"
            print(f"  (默认: {user_input})")

        # LLM 总结
        print(f"\n  >>> LLM 总结需求...")
        summary = call_llm([{"role": "user", "content": (
            f"总结用户购车需求为简洁要点:\n"
            f"{'已有: ' + user_needs + chr(10) if user_needs else ''}"
            f"新输入: {user_input}\n只输出需求要点。"
        )}], temperature=0.1)
        user_needs = summary
        print(f"  需求总结: {user_needs}")

        # ==========================================================
        # 路由决策（对应条件边）
        # ==========================================================
        print(f"\n{'=' * 70}")
        print(f"【路由】需求是否充分？")
        print(f"{'=' * 70}")
        print(f"  规则: 有预算+车型偏好 → 搜索; 否则 → 继续问")
        print(f"  结果: → 进入搜索（简化处理，默认充分）")

        # ==========================================================
        # 阶段2：搜索车辆（对应 search_listings 节点）
        # ==========================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段2】搜索匹配车辆")
        print(f"{'=' * 70}")
        print(f"  原教程: Playwright 爬取 Autotrader 网站")
        print(f"  本课: 用模拟数据匹配")

        # LLM 筛选
        cars_json = json.dumps([{"id": c["id"], "title": c["title"], "price": c["price"], "fuel": c["fuel"]} for c in CARS], ensure_ascii=False)
        match_result = call_llm([{"role": "user", "content": (
            f"从以下车辆中选出最匹配用户需求的3辆，返回id用逗号分隔:\n"
            f"需求: {user_needs}\n车辆:\n{cars_json}\n只输出id。"
        )}], temperature=0.1)

        ids = re.findall(r'\d+', match_result)
        matched = [c for c in CARS if c["id"] in ids][:3]
        if not matched:
            matched = CARS[:3]

        print(f"\n  匹配结果:")
        for i, car in enumerate(matched, 1):
            print(f"    {i}. {car['title']} - {car['price']} ({car['fuel']})")

        print(f"\n  输入编号查看详情，r 修改条件，q 结束")
        choice = input("  选择: ").strip() or "1"

        if choice.lower() == "q":
            break
        elif choice.lower() == "r":
            continue

        # ==========================================================
        # 阶段3：车辆详情（对应 fetch_additional_info 节点）
        # ==========================================================
        idx = int(choice) - 1 if choice.isdigit() and 0 < int(choice) <= len(matched) else 0
        car = matched[idx]

        print(f"\n{'=' * 70}")
        print(f"【阶段3】车辆详情 + 购买建议")
        print(f"{'=' * 70}")
        print(f"  车辆: {car['title']}")
        print(f"  原教程: 爬取详情页 + DuckDuckGo搜索口碑")

        analysis = call_llm([{"role": "user", "content": (
            f"你是汽车顾问。分析这辆车:\n{car['title']} {car['price']}\n{car['detail']}\n"
            f"用户需求: {user_needs}\n请提供亮点、问题、是否匹配、购买建议。中文。"
        )}])
        print(f"\n  分析:\n{analysis}")

        print(f"\n  输入 r 重新搜索，q 结束")
        choice = input("  选择: ").strip()
        if choice.lower() != "r":
            break

    print(f"\n{'=' * 70}")
    print(f"  对话结束，感谢使用！")
    print(f"{'=' * 70}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结")
    print(f"{'=' * 70}")
    print(f"  1. 核心是一个 while True 循环，内含三个阶段:")
    print(f"     询问需求 → 搜索列表 → 查看详情")
    print(f"  2. 每个阶段后都有用户选择，决定进入哪个分支:")
    print(f"     - 继续问 / 搜索 / 选车 / 改条件 / 结束")
    print(f"  3. LangGraph 的条件边就是这些 if-elif-else 分支")
    print(f"  4. 原教程的亮点是 Playwright 爬虫爬真实网站")
    print(f"     + DuckDuckGo 搜索车型口碑")
    print(f"  5. 本质: while 循环 + input() + LLM 调用 + 条件分支")


if __name__ == "__main__":
    main()
