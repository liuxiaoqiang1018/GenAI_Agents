"""
第38课：智能购物推荐代理 ShopGenie（透明调试版）

不使用任何框架，展示结构化提取和产品对比推荐的内部机制。
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
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            return re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
        except Exception as e:
            print(f"    !! LLM失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# 模拟数据
PRODUCTS = [
    {"name": "华为 Mate 70 Pro",
     "content": "华为Mate 70 Pro搭载麒麟9100处理器，5500mAh大电池，支持100W有线快充。5000万像素XMAGE主摄。6.82英寸OLED屏幕，120Hz。256GB起步。优点：信号强、影像好、续航长。缺点：价格偏高、应用生态有差距。评分4.6/5。售价5499元起。"},
    {"name": "小米 15 Pro",
     "content": "小米15 Pro搭载骁龙8至尊版，6000mAh电池，90W快充。5000万像素徕卡三摄。6.73英寸2K AMOLED屏幕，峰值3200nit。256GB起步。优点：性价比高、性能强。缺点：机身偏重、发热控制一般。评分4.7/5。售价4299元起。"},
    {"name": "iPhone 16 Pro Max",
     "content": "iPhone 16 Pro Max搭载A18 Pro芯片，4685mAh电池。4800万像素主摄+5倍变焦。6.9英寸Super Retina XDR。256GB起步。优点：生态完善、视频强、流畅。缺点：价格昂贵、快充慢。评分4.5/5。售价9999元起。"}
]


def main():
    query = "5000元以下最好的手机推荐"

    print("=" * 70)
    print("  智能购物推荐代理 ShopGenie（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  查询: {query}")

    # ==============================================================
    # 阶段1：搜索产品
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段1】搜索产品信息（模拟 Tavily 搜索）")
    print(f"{'=' * 70}")
    print(f"  原教程用 Tavily API 搜索 + WebBaseLoader 爬取博客全文")
    print(f"  本课用模拟数据替代")

    for p in PRODUCTS:
        print(f"  - {p['name']}")

    # ==============================================================
    # 阶段2：LLM 结构化提取
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段2】LLM 结构化提取产品信息")
    print(f"{'=' * 70}")
    print(f"  这是本课的核心: 用 Pydantic Schema 约束 LLM 输出 JSON")
    print(f"  原教程用 SmartphoneReview(BaseModel) + JsonOutputParser")
    print(f"  本质: 在 Prompt 中描述 JSON 格式，让 LLM 按格式输出")

    structured = []
    for product in PRODUCTS:
        print(f"\n  --- 提取: {product['name']} ---")

        prompt = (
            f"请从以下产品评测中提取结构化信息，用JSON格式输出。\n\n"
            f"产品评测:\n{product['content']}\n\n"
            f"输出格式:\n"
            f'{{"name": "产品名", "price": "价格", "processor": "处理器", "battery": "电池", '
            f'"camera": "相机", "display": "屏幕", "storage": "存储", '
            f'"pros": ["优点1"], "cons": ["缺点1"], "score": 4.5}}\n'
            f"只输出JSON。"
        )

        print(f"  >>> Prompt: 提取 {product['name']} 的结构化信息...")
        result = call_llm([{"role": "user", "content": prompt}], temperature=0.1)
        result = re.sub(r'^```(?:json)?\s*', '', result)
        result = re.sub(r'\s*```$', '', result)

        try:
            data = json.loads(result)
            structured.append(data)
            print(f"  >>> 提取结果:")
            print(f"      名称: {data.get('name')}")
            print(f"      价格: {data.get('price')}")
            print(f"      处理器: {data.get('processor')}")
            print(f"      评分: {data.get('score')}/5")
            print(f"      优点: {data.get('pros')}")
            print(f"      缺点: {data.get('cons')}")
        except json.JSONDecodeError:
            print(f"  >>> JSON解析失败: {result[:100]}")

    # ==============================================================
    # 阶段3：LLM 对比推荐
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段3】LLM 对比分析 + 推荐最佳产品")
    print(f"{'=' * 70}")
    print(f"  原教程用 ProductComparison(BaseModel) 做嵌套结构化输出")
    print(f"  包含 SpecsComparison + RatingsComparison + BestProduct")

    products_json = json.dumps(structured, ensure_ascii=False, indent=2)
    compare_prompt = (
        f"你是专业手机评测专家。对比以下产品并推荐最佳选择。\n\n"
        f"用户需求: {query}\n\n"
        f"产品信息:\n{products_json}\n\n"
        f"请从性能、拍照、续航、性价比维度对比，推荐一款最佳产品。\n"
        f"用中文，使用表格格式。"
    )

    print(f"\n  >>> 发送给 LLM: 对比 {len(structured)} 款产品...")
    comparison = call_llm([{"role": "user", "content": compare_prompt}])
    print(f"\n  >>> LLM 对比结果:")
    print(f"  {comparison}")

    # ==============================================================
    # 阶段4：搜索评测视频（模拟）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段4】搜索评测视频（模拟 YouTube API）")
    print(f"{'=' * 70}")
    print(f"  原教程用 YouTube Data API v3 搜索评测视频")
    youtube_link = "https://www.youtube.com/results?search_query=手机评测+2025"
    print(f"  评测视频: {youtube_link}")

    # ==============================================================
    # 阶段5：生成报告
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段5】生成推荐报告")
    print(f"{'=' * 70}")

    report = f"# 购物推荐报告\n\n**查询**: {query}\n\n## 产品对比\n\n{comparison}\n\n## 评测视频\n\n[点击观看]({youtube_link})\n"

    filepath = os.path.join(os.path.dirname(__file__), "购物推荐报告_debug.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  已保存: {filepath}")

    print(f"\n{report}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：LangGraph 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. 6个节点线性串联: 搜索→提取→对比→YouTube→展示→邮件")
    print(f"  2. 核心创新: 复杂 Pydantic 嵌套结构化输出")
    print(f"     SmartphoneReview → ProductComparison → BestProduct")
    print(f"     让 LLM 输出的非结构化文本变成程序可处理的 JSON")
    print(f"  3. 多源整合: Web搜索(Tavily) + YouTube API + 邮件(SMTP)")
    print(f"  4. 本质: 3次 LLM 调用(提取/对比/邮件) + 2次 API 调用(搜索/视频)")
    print(f"     框架在此只是把它们串成流水线")


if __name__ == "__main__":
    main()
