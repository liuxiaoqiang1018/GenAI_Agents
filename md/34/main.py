"""
第34课：AI 周报生成代理（LangGraph 框架版）

架构：搜索者 → 摘要者 → 发布者（纯线性三步流水线）
核心模式：三 Agent 分工 + 线性流水线
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
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


# ===== 数据模型 =====
class Article(BaseModel):
    title: str
    url: str
    content: str
    source: str = ""


class GraphState(TypedDict):
    articles: Optional[List[dict]]
    summaries: Optional[List[dict]]
    report: Optional[str]


# ===== 模拟新闻数据（替代 Tavily API）=====
def get_simulated_articles() -> List[dict]:
    return [
        {
            "title": "OpenAI推出GPT-5：推理能力大幅跃升",
            "url": "https://example.com/gpt5",
            "source": "科技日报",
            "content": "OpenAI正式发布GPT-5大语言模型，在复杂推理任务上准确率提升30%，支持原生多模态输入（文本、图像、音频、视频），代码生成能力显著增强，能理解完整项目上下文。业内认为这是向通用人工智能迈出的重要一步。"
        },
        {
            "title": "谷歌DeepMind发布AlphaFold 3：预测所有生物分子结构",
            "url": "https://example.com/alphafold3",
            "source": "自然杂志",
            "content": "谷歌DeepMind联合Isomorphic Labs发布AlphaFold 3，能够预测蛋白质、DNA、RNA及药物分子的三维结构，准确度超过以往方法50%以上。这将加速药物研发和生物学基础研究。免费服务已向全球科研人员开放。"
        },
        {
            "title": "欧盟AI法案正式生效：全球首部综合性AI监管法规",
            "url": "https://example.com/eu-ai-act",
            "source": "路透社",
            "content": "欧盟《人工智能法案》正式生效，成为全球首部综合性AI监管法规。该法案将AI系统分为四个风险等级，对高风险AI应用（如招聘、信贷评估）提出严格合规要求，禁止社会评分和实时人脸识别等应用。违规企业面临最高3500万欧元罚款。"
        },
        {
            "title": "Anthropic发布Claude模型系统卡：透明度新标杆",
            "url": "https://example.com/claude-system-card",
            "source": "AI安全周刊",
            "content": "Anthropic发布了最新Claude模型的系统卡，详细记录了模型的能力边界、安全测试结果和已知限制。这是行业内最详尽的模型透明度报告之一，包括红队测试结果、偏见评估和部署建议。业内专家认为这为AI透明度设立了新标准。"
        },
        {
            "title": "开源AI生态爆发：Hugging Face用户突破100万",
            "url": "https://example.com/hf-milestone",
            "source": "开源中国",
            "content": "AI开源平台Hugging Face注册用户突破100万，托管模型超50万个。CEO表示开源AI正在民主化人工智能技术，降低企业和个人开发者的准入门槛。平台上最受欢迎的模型类别包括文本生成、图像处理和语音识别。社区驱动的模型评测排行榜成为行业基准。"
        }
    ]


# ===== 三个 Agent =====

class NewsSearcher:
    """搜索者 Agent：检索最新 AI/ML 新闻"""

    def search(self) -> List[dict]:
        print("  检索 AI/ML 最新新闻...")
        articles = get_simulated_articles()
        for a in articles:
            print(f"    - {a['title']} ({a['source']})")
        return articles


class Summarizer:
    """摘要者 Agent：逐篇生成通俗摘要"""

    def __init__(self):
        self.system_prompt = (
            "你是一位AI领域专家，擅长将复杂技术话题用通俗易懂的语言解释给普通读者。"
            "请用2-3句话总结这篇文章，重点突出关键信息，用简单的语言解释专业术语。"
        )

    def summarize(self, article: dict) -> str:
        response = call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"标题: {article['title']}\n\n内���: {article['content']}"}
        ], temperature=0.3)
        return response


class Publisher:
    """发布者 Agent：编排 Markdown 报告"""

    def create_report(self, summaries: List[dict]) -> str:
        summaries_text = "\n\n".join([
            f"标题: {item['title']}\n摘要: {item['summary']}\n来源: {item['url']}"
            for item in summaries
        ])

        prompt = (
            "请根据以下新闻摘要生成一份 AI/ML 周报。\n"
            "格式要求:\n"
            "1. 一段简短的导语（介绍本周亮点）\n"
            "2. 逐条新闻及摘要\n"
            "3. 总结展望\n"
            "语言通俗易懂，面向普通读者。用中文输出。"
        )

        response = call_llm([
            {"role": "system", "content": prompt},
            {"role": "user", "content": summaries_text}
        ])

        # 保存报告文件
        current_date = datetime.now().strftime("%Y-%m-%d")
        report_file = os.path.join(os.path.dirname(__file__), f"ai_周报_{current_date}.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# AI/ML 周报 ({current_date})\n\n{response}")
        print(f"  报告已保存: {report_file}")

        return response


# ===== 节点函数 =====

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("【搜索者 Agent】检索新闻")
    print("=" * 60)
    searcher = NewsSearcher()
    articles = searcher.search()
    return {**state, "articles": [a if isinstance(a, dict) else a.dict() for a in articles]}


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("【摘要者 Agent】逐篇生成摘要")
    print("=" * 60)
    summarizer = Summarizer()
    summaries = []

    for article in state["articles"]:
        print(f"\n  摘要: {article['title']}")
        summary = summarizer.summarize(article)
        print(f"  → {summary[:100]}...")
        summaries.append({
            "title": article["title"],
            "summary": summary,
            "url": article["url"]
        })

    return {**state, "summaries": summaries}


def publish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("【发布者 Agent】编排 Markdown 报告")
    print("=" * 60)
    publisher = Publisher()
    report = publisher.create_report(state["summaries"])
    return {**state, "report": report}


# ===== 构建工作流 =====
def create_workflow():
    workflow = StateGraph(state_schema=GraphState)

    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("publish", publish_node)

    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "publish")
    workflow.set_entry_point("search")

    return workflow.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  AI 周报生成代理（LangGraph 框架版）")
    print("=" * 60)

    workflow = create_workflow()
    final_state = workflow.invoke({
        "articles": None,
        "summaries": None,
        "report": None
    })

    print("\n" + "=" * 60)
    print("【最终报告】")
    print("=" * 60)
    print(final_state["report"])


if __name__ == "__main__":
    main()
