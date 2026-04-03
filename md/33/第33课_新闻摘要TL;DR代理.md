# 第33课：新闻摘要 TL;DR 代理（News TL;DR）

## 课程概述

本课构建一个**自动新闻搜索与摘要系统**：根据用户查询生成 NewsAPI 搜索参数 → 检索新闻元数据 → 爬取文章全文 → LLM 选出最相关文章 → 并行摘要 → 格式化输出。如果文章数量不够，自动放宽搜索条件重新搜索（自适应循环）。

**核心模式**：搜索-爬取-筛选-摘要流水线 + 自适应重试循环 + 结构化输出(Pydantic)

## 架构流程

```
用户查询 → LLM生成搜索参数(Pydantic结构化) → NewsAPI检索元数据
    → 爬取文章全文 → [文章数量够？]
      ├─ 不够 → 放宽搜索条件 → 重新搜索（循环）
      └─ 够了 → LLM选出Top-N最相关文章
                    → LLM逐篇生成摘要
                    → 格式化输出 → 结束
```

## 关键概念标注

| 概念 | 在代码中的位置 | Java 类比 |
|---|---|---|
| **结构化输出** | `NewsApiParams(BaseModel)` + `JsonOutputParser` | POJO + Jackson |
| **API 参数生成** | `generate_newsapi_params()` — LLM 根据查询生成搜索参数 | 智能表单填充 |
| **网页爬取** | `retrieve_articles_text()` — BeautifulSoup 抓取全文 | Jsoup |
| **自适应循环** | `articles_text_decision()` — 文章不够就放宽条件重搜 | while + 退避策略 |
| **Top-N 选择** | `select_top_urls()` — LLM 从候选中选最相关的 | 智能排序/过滤 |
| **并行摘要** | `summarize_articles_parallel()` — 逐篇 LLM 生成 TL;DR | 批量处理 |
| **历史搜索去重** | `past_searches` + `scraped_urls` | 去重集合 |

## 本课特点

1. **自适应搜索循环**：不是简单搜一次，搜不够就自动放宽条件重搜（类似第7课自我评估循环）
2. **LLM 生成 API 参数**：用 Pydantic 结构化输出让 LLM 生成标准 API 调用参数
3. **网页爬取集成**：第一次在 Agent 中集成 BeautifulSoup 做网页抓取
4. **多级流水线**：搜索→爬取→筛选→摘要，4级处理流水线
5. **去重机制**：记录已搜过的参数和已爬过的URL，避免重复

## 环境准备

```bash
python -m venv venv
venv\Scripts\activate

pip install langgraph langchain langchain_openai httpx python-dotenv pydantic
```

注意：原教程使用 NewsAPI（需注册 key），本课 main.py 使用模拟新闻数据，无需 NewsAPI key。

## 运行方式

```bash
cd md/33
python main.py        # 框架版（LangGraph + 模拟新闻数据）
python main_debug.py  # 透明版（纯手写流水线）
```
