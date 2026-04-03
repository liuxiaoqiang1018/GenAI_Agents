"""
第38课：智能购物推荐代理 ShopGenie（LangGraph 框架版）

架构：搜索产品信息 → 结构化提取 → LLM对比推荐 → 搜索评测视频 → 展示结果
核心模式：线性流水线 + 复杂Pydantic结构化输出 + 多源信息整合
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

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


# ===== Pydantic 数据模型 =====
class ProductInfo(BaseModel):
    """产品结构化信息"""
    name: str = Field(description="产品名称")
    price: str = Field(description="价格")
    processor: str = Field(default="未知", description="处理器")
    battery: str = Field(default="未知", description="电池")
    camera: str = Field(default="未知", description="相机")
    display: str = Field(default="未知", description="屏幕")
    storage: str = Field(default="未知", description="存储")
    pros: List[str] = Field(default=[], description="优点")
    cons: List[str] = Field(default=[], description="缺点")
    score: float = Field(default=0.0, description="评分(满分5)")


# ===== 模拟数据 =====
def get_simulated_products() -> list:
    """模拟搜索到的产品博客数据"""
    return [
        {
            "name": "华为 Mate 70 Pro",
            "content": "华为Mate 70 Pro搭载麒麟9100处理器，5500mAh大电池，支持100W有线快充。5000万像素XMAGE主摄，影像表现出色。6.82英寸OLED屏幕，120Hz刷新率。256GB起步存储。优点：信号强、影像好、续航长。缺点：价格偏高、应用生态仍有差距。评分：4.6/5。售价5499元起。"
        },
        {
            "name": "小米 15 Pro",
            "content": "小米15 Pro搭载骁龙8至尊版处理器，性能强悍。6000mAh硅碳负极电池，支持90W快充。5000万像素徕卡三摄，人像拍摄尤其出色。6.73英寸2K AMOLED屏幕，峰值亮度3200nit。256GB起步。优点：性价比高、性能强、屏幕亮。缺点：机身偏重、发热控制一般。评分：4.7/5。售价4299元起。"
        },
        {
            "name": "iPhone 16 Pro Max",
            "content": "iPhone 16 Pro Max搭载A18 Pro芯片，配备4685mAh电池，支持MagSafe快充。4800万像素主摄+5倍光学变焦长焦。6.9英寸Super Retina XDR屏幕，ProMotion 120Hz。256GB起步。优点：生态完善、视频拍摄强、流畅度高。缺点：价格昂贵、快充速度慢、信号一般。评分：4.5/5。售价9999元起。"
        }
    ]


# ===== 状态 =====
class State(TypedDict):
    query: str
    products_raw: list
    products_structured: list
    comparison: str
    best_product: str
    youtube_link: str
    final_report: str


# ===== 节点函数 =====

def search_products(state: State) -> dict:
    """节点1：搜索产品信息（模拟）"""
    print("\n" + "=" * 60)
    print("【节点1】搜索产品信息")
    print("=" * 60)

    products = get_simulated_products()
    print(f"  搜索到 {len(products)} 款产品:")
    for p in products:
        print(f"    - {p['name']}")

    return {**state, "products_raw": products}


def extract_structured_info(state: State) -> dict:
    """节点2：LLM 结构化提取产品信息"""
    print("\n" + "=" * 60)
    print("【节点2】LLM 结构化提取产品信息")
    print("=" * 60)

    structured_products = []
    for product in state["products_raw"]:
        prompt = (
            f"请从以下产品评测中提取结构化信息，用JSON格式输出。\n\n"
            f"产品评测:\n{product['content']}\n\n"
            f"输出格式:\n"
            f'{{"name": "产品名", "price": "价格", "processor": "处理器", "battery": "电池", '
            f'"camera": "相机", "display": "屏幕", "storage": "存储", '
            f'"pros": ["优点1", "优点2"], "cons": ["缺点1", "缺点2"], "score": 4.5}}\n'
            f"只输出JSON，不要其他内容。"
        )

        result = call_llm([{"role": "user", "content": prompt}], temperature=0.1)
        # 清理JSON
        result = re.sub(r'^```(?:json)?\s*', '', result)
        result = re.sub(r'\s*```$', '', result)

        try:
            data = json.loads(result)
            structured_products.append(data)
            print(f"  {data.get('name', '未知')}: {data.get('score', 0)}/5 - {data.get('price', '未知')}")
        except json.JSONDecodeError:
            print(f"  JSON解析失败: {result[:100]}...")
            structured_products.append({"name": product["name"], "content": product["content"]})

    return {**state, "products_structured": structured_products}


def compare_products(state: State) -> dict:
    """节点3：LLM 对比分析并推荐最佳产品"""
    print("\n" + "=" * 60)
    print("【节点3】LLM 对比分析 + 推荐最佳产品")
    print("=" * 60)

    products_json = json.dumps(state["products_structured"], ensure_ascii=False, indent=2)
    prompt = (
        f"你是一位专业的手机评测专家。请对比以下产品并推荐最佳选择。\n\n"
        f"用户需求: {state['query']}\n\n"
        f"产品信息:\n{products_json}\n\n"
        f"请从性能、拍照、续航、性价比等维度对比，最后推荐一款最佳产品并说明理由。\n"
        f"用中文输出，使用表格和列表格式。"
    )

    comparison = call_llm([{"role": "user", "content": prompt}])
    print(f"  对比分析: {comparison[:300]}...")

    # 提取推荐产品
    best_prompt = (
        f"根据以下对比分析，用一句话说出推荐的最佳产品名称和核心理由:\n{comparison[:500]}\n"
        f"格式: 产品名: 理由"
    )
    best = call_llm([{"role": "user", "content": best_prompt}], temperature=0.1)
    print(f"  最佳推荐: {best}")

    return {**state, "comparison": comparison, "best_product": best}


def search_youtube(state: State) -> dict:
    """节点4：搜索 YouTube 评测视频（模拟）"""
    print("\n" + "=" * 60)
    print("【节点4】搜索评测视频（模拟）")
    print("=" * 60)

    # 从推荐产品中提取名称
    product_name = state["best_product"].split(":")[0].strip() if ":" in state["best_product"] else "手机评测"
    youtube_link = f"https://www.youtube.com/results?search_query={product_name}+评测"
    print(f"  评测视频搜索链接: {youtube_link}")

    return {**state, "youtube_link": youtube_link}


def generate_report(state: State) -> dict:
    """节点5：生成最终推荐报告"""
    print("\n" + "=" * 60)
    print("【节点5】生成推荐报告")
    print("=" * 60)

    report = f"# 购物推荐报告\n\n"
    report += f"**查询**: {state['query']}\n\n"
    report += f"## 产品对比\n\n{state['comparison']}\n\n"
    report += f"## 最佳推荐\n\n{state['best_product']}\n\n"
    report += f"## 评测视频\n\n[点击观看评测]({state['youtube_link']})\n"

    # 保存
    filepath = os.path.join(os.path.dirname(__file__), "购物推荐报告.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: {filepath}")

    return {**state, "final_report": report}


# ===== 构建工作流 =====
def build_workflow():
    builder = StateGraph(State)

    builder.add_node("search", search_products)
    builder.add_node("extract", extract_structured_info)
    builder.add_node("compare", compare_products)
    builder.add_node("youtube", search_youtube)
    builder.add_node("report", generate_report)

    builder.add_edge(START, "search")
    builder.add_edge("search", "extract")
    builder.add_edge("extract", "compare")
    builder.add_edge("compare", "youtube")
    builder.add_edge("youtube", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  智能购物推荐代理 ShopGenie（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()

    result = app.invoke({
        "query": "5000元以下最好的手机推荐",
        "products_raw": [],
        "products_structured": [],
        "comparison": "",
        "best_product": "",
        "youtube_link": "",
        "final_report": ""
    })

    print(f"\n{'=' * 60}")
    print(f"【最终推荐报告】")
    print(f"{'=' * 60}")
    print(result["final_report"])


if __name__ == "__main__":
    main()
