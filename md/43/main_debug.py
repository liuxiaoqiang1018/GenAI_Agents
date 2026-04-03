"""
第43课：EU 绿色合规 FAQ 机器人（透明调试版）

展示 RAG 管道每一步的 Prompt 构建和 LLM 调用过程。
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

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


def call_llm(messages: list, temperature: float = 0.3) -> str:
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


# 知识库
KB = [
    {"id": 1, "content": "欧盟绿色协议（EGD）是欧盟到2050年实现温室气体净零排放的战略，涵盖农业、能源、制造业等领域。"},
    {"id": 2, "content": "碳边境调节机制（CBAM）对来自气候政策不严格国家的商品进口征收碳成本。覆盖水泥、钢铁、化肥、电力和铝。"},
    {"id": 3, "content": "循环经济行动计划（CEAP）促进材料再利用、修复和回收，重点领域包括包装、纺织品和电子产品。"},
    {"id": 4, "content": "从农场到餐桌战略（F2F）目标是减少50%农药使用，减少营养流失，促进有机农业。"},
    {"id": 5, "content": "CBAM对中小企业：出口到欧盟需报告碳排放量，未合规可能面临贸易壁垒。需投资碳追踪系统。"},
]

GOLD_QA = [
    {"query": "什么是欧盟绿色协议？", "answer": "欧盟绿色协议是到2050年实现温室气体净零排放的战略，涵盖农业、能源和制造业等领域。"},
]


def similarity(a: str, b: str) -> float:
    ta, tb = Counter(a.lower()), Counter(b.lower())
    all_t = set(ta) | set(tb)
    va = [ta.get(t, 0) for t in all_t]
    vb = [tb.get(t, 0) for t in all_t]
    dot = sum(x * y for x, y in zip(va, vb))
    na, nb = math.sqrt(sum(x*x for x in va)), math.sqrt(sum(x*x for x in vb))
    return dot / (na * nb) if na and nb else 0.0


def main():
    query = "什么是欧盟绿色协议？"

    print("=" * 70)
    print("  EU 绿色合规 FAQ 机器人（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  用户问题: {query}")
    print(f"  知识库: {len(KB)} 个文档块")

    # ==============================================================
    # 步骤1：查询改写
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤1】查询改写（Rephrase）")
    print(f"{'=' * 70}")
    print(f"  原教程: LLM 改写查询，提高检索命中率")

    rephrase_prompt = f"请改写以下问题使其更适合检索:\n{query}\n只输出改写后的问题。"
    print(f"\n  >>> Prompt: {rephrase_prompt}")

    rephrased = call_llm([{"role": "user", "content": rephrase_prompt}])
    print(f"  >>> 改写后: {rephrased}")

    # ==============================================================
    # 步骤2：向量检索
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤2】向量检索（模拟 FAISS）")
    print(f"{'=' * 70}")
    print(f"  原教程: FAISS + OpenAI Embedding + SemanticChunker")
    print(f"  本课: 简易字符级余弦相似度模拟")

    scored = [(chunk, similarity(rephrased, chunk["content"])) for chunk in KB]
    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = scored[:3]

    print(f"\n  检索结果（按相似度排序）:")
    for chunk, score in top3:
        print(f"    [块{chunk['id']} 相似度={score:.3f}] {chunk['content'][:60]}...")

    # ==============================================================
    # 步骤3：LLM 评分过滤
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤3】LLM 评分过滤（双重过滤）")
    print(f"{'=' * 70}")
    print(f"  原教程: RetrieverAgent 用 LLM 给每个块打 yes/no 分")
    print(f"  目的: 向量检索可能返回语义不相关的块，LLM 再筛一轮")

    relevant = []
    for chunk, score in top3:
        grade_prompt = f"查询: {rephrased}\n文档块: {chunk['content']}\n这个文档块与查询相关吗？只回答yes或no。"

        print(f"\n  >>> 块{chunk['id']} 评分...")
        print(f"  >>> Prompt: {grade_prompt[:100]}...")

        grade = call_llm([
            {"role": "system", "content": "你是相关性评估专家。只回答yes或no。"},
            {"role": "user", "content": grade_prompt}
        ], temperature=0)

        is_rel = "yes" in grade.lower()
        print(f"  >>> LLM判定: {grade} → {'相关' if is_rel else '不相关'}")
        if is_rel:
            relevant.append(chunk)

    if not relevant:
        relevant = [c for c, _ in top3]
        print(f"  无相关块，回退使用所有检索结果")

    print(f"\n  过滤后: {len(relevant)} 个相关块")

    # ==============================================================
    # 步骤4：LLM 摘要生成
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤4】LLM 生成回答（SummarizerAgent）")
    print(f"{'=' * 70}")

    context = "\n\n".join([c["content"] for c in relevant])
    answer_prompt = f"问题: {query}\n\n参考文档:\n{context}"

    print(f"\n  >>> system: EU绿色合规专家，简洁回答，不超两句")
    print(f"  >>> user: {answer_prompt[:150]}...")

    answer = call_llm([
        {"role": "system", "content": "你是EU绿色合规专家。根据文档简洁准确回答，不超两句话。"},
        {"role": "user", "content": answer_prompt}
    ])

    print(f"\n  >>> 回答: {answer}")

    # ==============================================================
    # 步骤5：质量评估
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤5】质量评估（EvaluationAgent）")
    print(f"{'=' * 70}")
    print(f"  原教程: 余弦相似度 + F1 分数对比黄金QA")

    gold = GOLD_QA[0]
    sim = similarity(answer, gold["answer"])
    print(f"  黄金答案: {gold['answer']}")
    print(f"  生成答案: {answer}")
    print(f"  相似度: {sim:.3f}")
    print(f"  评估: {'通过' if sim >= 0.3 else '需改进'}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：RAG 管道的本质")
    print(f"{'=' * 70}")
    print(f"  1. RAG = 检索(Retrieval) + 生成(Generation)")
    print(f"     检索: 从知识库找相关内容")
    print(f"     生成: 把检索结果塞进 Prompt 让 LLM 回答")
    print(f"  2. 本课的增强:")
    print(f"     - 查询改写: LLM 优化查询提高检索率")
    print(f"     - LLM评分过滤: 向量检索后再用 LLM 过滤不相关的")
    print(f"     - 黄金QA评估: 自动评估回答质量")
    print(f"  3. 本质: 相似度搜索 + Prompt拼接 + LLM调用")
    print(f"     FAISS/向量库只是加速了相似度搜索的速度")
    print(f"  4. Java类比: Elasticsearch + Spring AI 做相同的事")


if __name__ == "__main__":
    main()
