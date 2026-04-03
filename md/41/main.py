"""
第41课：杂货管理代理系统（模拟 CrewAI 版）

架构：4 Agent 接力 — 小票解析 → 保质期估算 → 库存跟踪 → 食谱推荐
核心模式：Agent/Task/Crew + context传递 + human_input + JSON结构化输出
"""

import os
import re
import sys
import json
import time
import httpx
from datetime import datetime, timedelta
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
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# ===== 模拟购物小票 =====
RECEIPT_MARKDOWN = """
# 永辉超市购物小票
日期: 2026-04-03

| 商品 | 数量 | 单位 | 单价 | 金额 |
|------|------|------|------|------|
| 鸡蛋 | 10 | 个 | 1.5 | 15.0 |
| 牛奶(伊利纯牛奶) | 2 | 盒(1L) | 7.5 | 15.0 |
| 西兰花 | 1 | 颗(约500g) | 6.8 | 6.8 |
| 猪肉(里脊) | 1 | 份(500g) | 25.0 | 25.0 |
| 番茄 | 4 | 个 | 3.0 | 12.0 |
| 米饭(东北大米) | 1 | 袋(5kg) | 35.0 | 35.0 |
| 豆腐 | 2 | 块 | 3.5 | 7.0 |
| 青椒 | 3 | 个 | 2.0 | 6.0 |

合计: 121.8 元
"""


# ===== 4个 Agent 的执行函数 =====

def receipt_interpreter(receipt: str) -> str:
    """Agent 1: 小票解析者 — 从购物小票提取结构化商品信息"""
    print("\n" + "=" * 60)
    print("【Agent 1】小票解析者 — 提取商品信息")
    print("  角色: 精确、细致的数据提取专家")
    print("=" * 60)

    prompt = (
        "你是一个购物小票解析专家。请从以下购物小票中提取所有商品信息。\n\n"
        f"购物小票:\n{receipt}\n\n"
        "请用JSON格式输出，格式如下:\n"
        '{"items": [{"item_name": "商品名", "count": 数量, "unit": "单位"}], '
        '"date_of_purchase": "YYYY-MM-DD"}\n'
        "只输出JSON，不要其他内容。"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0.1)
    result = re.sub(r'^```(?:json)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    print(f"  提取结果: {result[:300]}...")
    return result


def expiration_estimator(items_json: str) -> str:
    """Agent 2: 保质期专家 — 估算每个商品的过期日期"""
    print("\n" + "=" * 60)
    print("【Agent 2】保质期专家 — 估算过期日期")
    print("  角色: 严谨、可靠的食品安全专家")
    print("  原教程: 搜索 stilltasty.com 查询保质期")
    print("=" * 60)

    prompt = (
        "你是一个食品保质期专家。根据以下商品列表，估算每个商品冷藏保存的保质期天数，"
        "然后加上购买日期得出过期日期。\n\n"
        f"商品信息:\n{items_json}\n\n"
        "请用JSON格式输出，在每个商品中增加 expiration_date 字段:\n"
        '{"items": [{"item_name": "商品名", "count": 数量, "unit": "单位", '
        '"expiration_date": "YYYY-MM-DD"}]}\n'
        "只输出JSON。"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0.2)
    result = re.sub(r'^```(?:json)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    print(f"  保质期估算: {result[:400]}...")
    return result


def grocery_tracker(items_with_expiry: str) -> str:
    """Agent 3: 库存跟踪者 — 根据消耗更新库存（含人工输入）"""
    print("\n" + "=" * 60)
    print("【Agent 3】库存跟踪者 — 更新库存")
    print("  角色: 细心、响应的库存管理员")
    print("  特点: human_input=True，需要用户告知消耗情况")
    print("=" * 60)

    print(f"\n  当前库存:\n{items_with_expiry[:400]}...")

    # 人工输入（human_input）
    print(f"\n  请告诉我您已经消耗了哪些食材：")
    print(f"  （示例: 鸡蛋用了3个，牛奶喝了1盒，番茄用了2个）")
    user_input = input("  你的输入（回车用默认值）: ").strip()
    if not user_input:
        user_input = "鸡蛋用了3个，牛奶喝了1盒，番茄用了2个"
        print(f"  (默认: {user_input})")

    prompt = (
        "你是库存跟踪专家。根据用户的消耗输入，更新库存数量。\n\n"
        f"当前库存:\n{items_with_expiry}\n\n"
        f"用户消耗: {user_input}\n\n"
        "请减去已消耗的数量，输出更新后的库存JSON。保留过期日期。\n"
        '{"items": [{"item_name": "商品名", "count": 剩余数量, "unit": "单位", '
        '"expiration_date": "YYYY-MM-DD"}]}\n'
        "只输出JSON。"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0.1)
    result = re.sub(r'^```(?:json)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    print(f"  更新后库存: {result[:400]}...")

    # 保存文件
    tracker_file = os.path.join(os.path.dirname(__file__), "grocery_tracker.json")
    with open(tracker_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  已保存: {tracker_file}")

    return result


def recipe_recommender(remaining_items: str) -> str:
    """Agent 4: 食谱推荐者 — 根据剩余食材推荐食谱"""
    print("\n" + "=" * 60)
    print("【Agent 4】食谱推荐者 — 推荐食谱")
    print("  角色: 创意、实用的家庭厨师")
    print("  原教程: 搜索 americastestkitchen.com")
    print("=" * 60)

    prompt = (
        "你是一个家庭食谱推荐专家。根据以下剩余食材，推荐2-3个简单实用的家常菜食谱。\n"
        "只使用库存中有的食材（count > 0），不要使用已用完的。\n"
        "如果食材不够做某道菜，建议需要额外购买什么。\n\n"
        f"剩余食材:\n{remaining_items}\n\n"
        "请用中文输出，每个食谱包含：菜名、所需食材和用量、做法步骤。"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  推荐食谱:\n{result[:500]}...")

    # 保存文件
    recipe_file = os.path.join(os.path.dirname(__file__), "recipe_recommendation.md")
    with open(recipe_file, "w", encoding="utf-8") as f:
        f.write(f"# 食谱推荐\n\n{result}")
    print(f"  已保存: {recipe_file}")

    return result


# ===== 主函数（模拟 Crew.kickoff()）=====
def main():
    print("=" * 60)
    print("  杂货管理代理系统（CrewAI 模式）")
    print("=" * 60)

    # Task 1: 解析小票
    items_json = receipt_interpreter(RECEIPT_MARKDOWN)

    # Task 2: 估算保质期（context = Task 1 的输出）
    items_with_expiry = expiration_estimator(items_json)

    # Task 3: 库存跟踪（context = Task 2 的输出 + human_input）
    remaining = grocery_tracker(items_with_expiry)

    # Task 4: 食谱推荐（context = Task 3 的输出）
    recipes = recipe_recommender(remaining)

    print(f"\n{'=' * 60}")
    print("【完成】所有 Agent 任务执行完毕")
    print(f"{'=' * 60}")
    print(f"  库存文件: grocery_tracker.json")
    print(f"  食谱文件: recipe_recommendation.md")


if __name__ == "__main__":
    main()
