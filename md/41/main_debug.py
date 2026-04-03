"""
第41课：杂货管理代理系统（透明调试版）

展示 CrewAI 的 Agent/Task/Crew 模式的内部机制：
- Agent = system prompt（角色+目标+背景+性格）
- Task = user prompt（描述+期望输出+上下文）
- Crew = 顺序执行所有 Task，上一个的输出作为下一个的 context
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
                raise ValueError("无choices")
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


RECEIPT = """
# 永辉超市购物小票 (2026-04-03)
| 商品 | 数量 | 单位 | 金额 |
|------|------|------|------|
| 鸡蛋 | 10 | 个 | 15.0 |
| 牛奶 | 2 | 盒(1L) | 15.0 |
| 西兰花 | 1 | 颗 | 6.8 |
| 猪肉里脊 | 1 | 份(500g) | 25.0 |
| 番茄 | 4 | 个 | 12.0 |
| 大米 | 1 | 袋(5kg) | 35.0 |
| 豆腐 | 2 | 块 | 7.0 |
| 青椒 | 3 | 个 | 6.0 |
"""


def main():
    print("=" * 70)
    print("  杂货管理代理系统（透明调试版 - 无框架）")
    print("=" * 70)

    # ==============================================================
    # Task 1: 小票解析（Agent 1 执行）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Task 1】小票解析")
    print(f"{'=' * 70}")

    # CrewAI 的本质: system = Agent(role+goal+backstory), user = Task(description)
    agent1_system = (
        "角色: 购物小票解析专家\n"
        "目标: 从购物小票中精确提取商品名称、数量、单位\n"
        "性格: 精确、细致、高效"
    )
    task1_user = (
        f"分析以下购物小票，提取商品信息。输出JSON格式。\n\n"
        f"小票:\n{RECEIPT}\n\n"
        '格式: {{"items": [{{"item_name": "商品", "count": 数量, "unit": "单位"}}], "date_of_purchase": "YYYY-MM-DD"}}\n'
        "只输出JSON。"
    )

    print(f"\n  [Agent system prompt]:")
    print(f"    {agent1_system}")
    print(f"\n  [Task description] (部分):")
    print(f"    {task1_user[:150]}...")
    print(f"\n  >>> 调用 LLM...")

    task1_output = call_llm([
        {"role": "system", "content": agent1_system},
        {"role": "user", "content": task1_user}
    ], temperature=0.1)
    task1_output = re.sub(r'^```(?:json)?\s*', '', task1_output)
    task1_output = re.sub(r'\s*```$', '', task1_output)

    print(f"\n  >>> Task 1 输出:\n  {task1_output[:300]}...")

    # ==============================================================
    # Task 2: 保质期估算（context = Task 1 输出）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Task 2】保质期估算（context = Task 1 的输出）")
    print(f"{'=' * 70}")
    print(f"  这就是 CrewAI 的 context 机制: Task2.context=[task1]")
    print(f"  本质: 把 Task 1 的输出拼进 Task 2 的 Prompt 里")

    agent2_system = (
        "角色: 食品保质期专家\n"
        "目标: 估算每种食品冷藏保存的保质期，计算过期日期\n"
        "性格: 严谨、可靠"
    )
    task2_user = (
        f"根据以下商品列表估算保质期，加上购买日期得出过期日期。\n\n"
        f"[context from Task 1]:\n{task1_output}\n\n"
        '输出JSON，增加 expiration_date 字段。只输出JSON。'
    )

    print(f"\n  >>> 调用 LLM（context 已注入）...")

    task2_output = call_llm([
        {"role": "system", "content": agent2_system},
        {"role": "user", "content": task2_user}
    ], temperature=0.2)
    task2_output = re.sub(r'^```(?:json)?\s*', '', task2_output)
    task2_output = re.sub(r'\s*```$', '', task2_output)

    print(f"\n  >>> Task 2 输出:\n  {task2_output[:400]}...")

    # ==============================================================
    # Task 3: 库存跟踪（context = Task 2 + human_input）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Task 3】库存跟踪（context = Task 2 + human_input）")
    print(f"{'=' * 70}")
    print(f"  human_input=True: CrewAI 在此暂停，等待用户输入消耗情况")

    print(f"\n  当前库存:\n  {task2_output[:300]}...")
    print(f"\n  请告诉我已消耗的食材:")
    consumed = input("  输入（回车用默认）: ").strip()
    if not consumed:
        consumed = "鸡蛋用了3个，牛奶喝了1盒，番茄用了2个"
        print(f"  (默认: {consumed})")

    agent3_system = "角色: 库存跟踪专家\n目标: 根据消耗更新库存\n性格: 细心、响应"
    task3_user = (
        f"根据消耗更新库存。\n\n"
        f"[context from Task 2]:\n{task2_output}\n\n"
        f"[human_input]: {consumed}\n\n"
        "减去已消耗的，输出更新后的JSON。只输出JSON。"
    )

    print(f"\n  >>> 调用 LLM...")

    task3_output = call_llm([
        {"role": "system", "content": agent3_system},
        {"role": "user", "content": task3_user}
    ], temperature=0.1)
    task3_output = re.sub(r'^```(?:json)?\s*', '', task3_output)
    task3_output = re.sub(r'\s*```$', '', task3_output)

    print(f"\n  >>> Task 3 输出（更新后库存）:\n  {task3_output[:400]}...")

    # ==============================================================
    # Task 4: 食谱推荐（context = Task 3）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Task 4】食谱推荐（context = Task 3 的输出）")
    print(f"{'=' * 70}")

    agent4_system = "角色: 家庭食谱推荐专家\n目标: 用剩余食材推荐实用食谱\n性格: 创意、实用"
    task4_user = (
        f"根据剩余食材推荐2-3个家常菜。只用有库存的食材。\n\n"
        f"[context from Task 3]:\n{task3_output}\n\n"
        "每个食谱含菜名、食材用量、做法步骤。中文。"
    )

    print(f"\n  >>> 调用 LLM...")

    task4_output = call_llm([
        {"role": "system", "content": agent4_system},
        {"role": "user", "content": task4_user}
    ])

    print(f"\n  >>> Task 4 输出（食谱推荐）:\n  {task4_output}")

    # 保存
    recipe_file = os.path.join(os.path.dirname(__file__), "recipe_debug.md")
    with open(recipe_file, "w", encoding="utf-8") as f:
        f.write(f"# 食谱推荐\n\n{task4_output}")
    print(f"\n  已保存: {recipe_file}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：CrewAI 的 Agent/Task/Crew 模式本质")
    print(f"{'=' * 70}")
    print(f"  1. Agent = system prompt（role + goal + backstory + personality）")
    print(f"     就是给 LLM 设定角色和人设")
    print(f"  2. Task = user prompt（description + expected_output）")
    print(f"     就是给 LLM 具体的任务指令和输出格式要求")
    print(f"  3. context=[prev_task] = 把上一个 Task 的输出拼进当前 Prompt")
    print(f"     就是函数调用的返回值传参")
    print(f"  4. human_input=True = 运行中暂停等待 input()")
    print(f"  5. Crew.kickoff() = 按顺序执行所有 Task")
    print(f"     就是 for task in tasks: task.execute()")
    print(f"  6. 和 LangGraph 的区别:")
    print(f"     - CrewAI: 角色扮演，Agent 有人设和性格")
    print(f"     - LangGraph: 图结构，节点+边+路由")
    print(f"     - CrewAI 更像'团队协作'，LangGraph 更像'工作流引擎'")


if __name__ == "__main__":
    main()
