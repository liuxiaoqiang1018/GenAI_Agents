# 第43课：EU 绿色合规 FAQ 机器人（EU Green Compliance FAQ Bot）

## 课程概述

本课构建一个**基于 RAG 的法规问答系统**：将 EU 绿色协议文档分块存入向量库 → 用户提问后检索相关块 → LLM 评分过滤不相关块 → LLM 生成摘要回答 → 评估Agent用黄金QA对评估回答质量。

**核心模式**：RAG（语义分块 + 向量检索 + LLM评分过滤 + 摘要生成）+ 质量评估

## 架构流程

```
[离线] EU法规文档 → 语义分块(SemanticChunker) → Embedding → 存入FAISS向量库

[在线] 用户提问 → 查询改写(Rephrase) → 向量检索Top-K
                                           ↓
                                  [检索Agent] LLM逐块评分(相关/不相关)
                                           ↓ 过滤后的块
                                  [摘要Agent] LLM生成简洁回答
                                           ↓
                                  [评估Agent] 与黄金QA对比(余弦相似度+F1)
                                           ↓
                                  输出回答 + 评估分数
```

## 三个 Agent 角色

| Agent | 职责 | 核心能力 |
|---|---|---|
| **RetrieverAgent（检索者）** | 向量检索 + LLM 相关性评分过滤 | FAISS检索 + LLM二分类(yes/no) |
| **SummarizerAgent（摘要者）** | 基于查询上下文生成简洁回答 | 上下文感知摘要 |
| **EvaluationAgent（评估者）** | 用黄金QA计算回答质量 | 余弦相似度 + F1分数 |

## 关键概念标注

| 概念 | 在代码中的位置 | Java 类比 |
|---|---|---|
| **语义分块** | `SemanticChunker` — 按语义断点而非固定长度切分 | 智能分段器 |
| **向量存储** | `FAISS.from_documents()` — 向量化存储和检索 | Elasticsearch 向量索引 |
| **LLM 评分过滤** | `_get_relevance_score()` — LLM 判断块是否相关 | 智能过滤器 |
| **查询改写** | `rephrase_query()` — LLM 改写查询提高检索率 | 查询扩展 |
| **黄金QA评估** | `gold_qa_dict` — 预设的标准问答对 | 单元测试断言 |
| **余弦相似度** | `_cosine_similarity()` — 词频向量余弦距离 | 文本相似度 |

## 本课特点

1. **完整 RAG 管道**：分块→嵌入→检索→过滤→摘要→评估，全链路
2. **LLM 评分过滤**：不是直接用检索结果，而是 LLM 再评一次相关性（双重过滤）
3. **查询改写**：提问前先 LLM 改写查询，提高检索命中率
4. **质量评估闭环**：有黄金QA对做自动评估，不是生成就完事

## 环境准备

```bash
python -m venv venv
venv\Scripts\activate

pip install langgraph langchain langchain_openai httpx python-dotenv
```

注意：原教程用 FAISS 向量库 + OpenAI Embedding，本课用简易 TF-IDF 相似度模拟向量检索。

## 运行方式

```bash
cd md/43
python main.py        # 完整 RAG 管道（模拟向量检索 + LLM 摘要）
python main_debug.py  # 透明版（展示 RAG 每一步的 Prompt 和过程）
```
