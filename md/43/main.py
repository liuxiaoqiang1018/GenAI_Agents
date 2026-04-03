"""
第43课：EU 绿色合规 FAQ 机器人（RAG 管道版）

架构：知识库分块 → 向量检索 → LLM评分过滤 → LLM摘要回答 → 质量评估
核心模式：RAG 全链路 + LLM 双重过滤 + 查询改写 + 黄金QA评估
"""

import os
import re
import sys
import math
import time
import httpx
from collections import Counter
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
def call_llm(messages: list, temperature: float = 0.3) -> str:
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


# ===== 模拟知识库（EU 绿色协议文档块）=====
KNOWLEDGE_BASE = [
    {"id": 1, "content": "欧盟绿色协议（European Green Deal, EGD）是欧盟到2050年实现温室气体净零排放的战略，同时追求可持续经济增长。涵盖农业、能源、制造业等多个领域。"},
    {"id": 2, "content": "碳边境调节机制（CBAM）是一项防止碳泄漏的政策工具，对来自气候政策不严格国家的特定商品进口征收碳成本。初始覆盖水泥、钢铁、化肥、电力和铝等高排放行业。"},
    {"id": 3, "content": "循环经济行动计划（CEAP）旨在通过促进材料的再利用、修复和回收来消除废物。重点领域包括包装、纺织品和电子产品行业。"},
    {"id": 4, "content": "从农场到餐桌战略（Farm to Fork, F2F）是绿色协议的一部分，目标是使欧盟食品系统公平、健康和环保。计划减少50%的农药使用，减少营养流失，促进有机农业发展。"},
    {"id": 5, "content": "欧盟2030年生物多样性战略专注于逆转生物多样性丧失。目标包括恢复退化生态系统、减少50%农药使用、确保25%农田为有机农田。"},
    {"id": 6, "content": "净零工业法案（NZIA）旨在提升欧盟净零技术制造能力，如太阳能板、电池和电解槽。目标是到2030年在国内制造至少40%的战略净零技术。"},
    {"id": 7, "content": "绿色协议工业计划旨在通过简化法规、增加资金、培养技能和促进贸易来增强欧洲的净零工业基础。重点制造电池、氢能系统和风力涡轮机等关键技术。"},
    {"id": 8, "content": "CBAM对中小企业出口商的影响：出口到欧盟的企业需要报告产品的碳排放量。未能遵守可能面临贸易壁垒和额外成本。中小企业需要投资碳排放追踪和报告系统。"},
]

# ===== 黄金QA对（用于评估）=====
GOLD_QA = [
    {"query": "什么是欧盟绿色协议？", "answer": "欧盟绿色协议是欧盟到2050年实现温室气体净零排放的战略，涵盖农业、能源和制造业等领域，追求可持续经济增长。"},
    {"query": "什么是碳边境调节机制？", "answer": "CBAM是对来自气候政策不严格国家的特定商品进口征收碳成本的政策工具，覆盖水泥、钢铁、化肥、电力和铝等行业。"},
    {"query": "循环经济行动计划的目标是什么？", "answer": "CEAP旨在通过促进再利用、修复和回收来消除废物，重点领域包括包装、纺织品和电子产品。"},
]


# ===== 简易 TF-IDF 相似度（替代 FAISS 向量检索）=====
def compute_similarity(query: str, text: str) -> float:
    """简易余弦相似度"""
    q_tokens = Counter(query.lower())
    t_tokens = Counter(text.lower())
    all_tokens = set(q_tokens.keys()) | set(t_tokens.keys())
    q_vec = [q_tokens.get(t, 0) for t in all_tokens]
    t_vec = [t_tokens.get(t, 0) for t in all_tokens]
    dot = sum(a * b for a, b in zip(q_vec, t_vec))
    norm_q = math.sqrt(sum(a * a for a in q_vec))
    norm_t = math.sqrt(sum(a * a for a in t_vec))
    if norm_q == 0 or norm_t == 0:
        return 0.0
    return dot / (norm_q * norm_t)


def retrieve_chunks(query: str, top_k: int = 3) -> list:
    """向量检索模拟：按相似度排序返回 Top-K"""
    scored = [(chunk, compute_similarity(query, chunk["content"])) for chunk in KNOWLEDGE_BASE]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(chunk, score) for chunk, score in scored[:top_k]]


# ===== RAG 管道 =====

def process_query(query: str):
    """完整 RAG 管道"""

    # 步骤1: 查询改写
    print("\n" + "=" * 60)
    print("【步骤1】查询改写")
    print("=" * 60)

    rephrase_result = call_llm([
        {"role": "user", "content": f"请改写以下问题，使其更清晰、更适合检索:\n{query}\n只输出改写后的问题。"}
    ])
    print(f"  原始查询: {query}")
    print(f"  改写后: {rephrase_result}")

    # 步骤2: 向量检索
    print("\n" + "=" * 60)
    print("【步骤2】向量检索 Top-3")
    print("=" * 60)

    retrieved = retrieve_chunks(rephrase_result, top_k=3)
    print(f"  检索到 {len(retrieved)} 个文档块:")
    for chunk, score in retrieved:
        print(f"    [相似度 {score:.3f}] {chunk['content'][:80]}...")

    # 步骤3: LLM 评分过滤
    print("\n" + "=" * 60)
    print("【步骤3】LLM 评分过滤（相关性判断）")
    print("=" * 60)

    relevant_chunks = []
    for chunk, score in retrieved:
        grade = call_llm([
            {"role": "system", "content": "你是相关性评估专家。判断文档块是否与查询相关，只回答 yes 或 no。"},
            {"role": "user", "content": f"查询: {rephrase_result}\n文档块: {chunk['content']}\n相关吗？"}
        ], temperature=0)

        is_relevant = "yes" in grade.lower()
        print(f"    块{chunk['id']}: {'相关' if is_relevant else '不相关'} (LLM判定: {grade})")
        if is_relevant:
            relevant_chunks.append(chunk)

    if not relevant_chunks:
        print("  无相关块，使用所有检索结果")
        relevant_chunks = [chunk for chunk, _ in retrieved]

    # 步骤4: LLM 摘要生成
    print("\n" + "=" * 60)
    print("【步骤4】LLM 生成回答")
    print("=" * 60)

    context = "\n\n".join([c["content"] for c in relevant_chunks])
    answer = call_llm([
        {"role": "system", "content": "你是EU绿色合规专家。根据提供的文档内容简洁准确地回答问题，不超过两句话。"},
        {"role": "user", "content": f"问题: {query}\n\n参考文档:\n{context}"}
    ])
    print(f"  回答: {answer}")

    # 步骤5: 质量评估
    print("\n" + "=" * 60)
    print("【步骤5】质量评估（与黄金QA对比）")
    print("=" * 60)

    best_score = 0.0
    best_gold = None
    for gold in GOLD_QA:
        sim = compute_similarity(answer, gold["answer"])
        if sim > best_score:
            best_score = sim
            best_gold = gold

    if best_gold:
        print(f"  最匹配的黄金QA: {best_gold['query']}")
        print(f"  相似度: {best_score:.3f}")
        if best_score >= 0.5:
            print(f"  评估: 通过（相似度 >= 0.5）")
        else:
            print(f"  评估: 需要改进（相似度 < 0.5）")

    return answer


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  EU 绿色合规 FAQ 机器人（RAG 管道版）")
    print("=" * 60)
    print(f"  知识库: {len(KNOWLEDGE_BASE)} 个文档块")
    print(f"  黄金QA: {len(GOLD_QA)} 对")

    test_queries = [
        "什么是欧盟绿色协议？",
        "碳边境调节机制对中小企业有什么影响？",
        "循环经济行动计划关注哪些领域？",
    ]

    for q in test_queries:
        print(f"\n{'*' * 60}")
        print(f"  用户问题: {q}")
        print(f"{'*' * 60}")
        process_query(q)

    print(f"\n{'=' * 60}")
    print("  所有查询处理完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
