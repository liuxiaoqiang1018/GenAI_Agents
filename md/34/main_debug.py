"""
第34课：AI 周报生成代理（透明调试版）

不使用任何框架，纯手写三步流水线。
让你看清线性多 Agent 流水线的内部机制。
"""

import os
import re
import sys
import json
import time
import httpx
from datetime import datetime
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
            print(f"    !! LLM调用失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 模拟新闻数据 =====
def get_articles():
    return [
        {"title": "OpenAI推出GPT-5：推理能力大幅跃升", "url": "https://example.com/gpt5", "source": "科技日报",
         "content": "OpenAI正式发布GPT-5大语言模型，在复杂推理任务上准确率提升30%，支持原生多模态输入，代码生成能力显著增强。"},
        {"title": "谷歌DeepMind发布AlphaFold 3：预测所有生物分子结构", "url": "https://example.com/alphafold3", "source": "自然杂志",
         "content": "AlphaFold 3能够预测蛋白质、DNA、RNA及药物分子的三维结构，准确度超过以往方法50%以上，将加速药物研发。"},
        {"title": "欧盟AI法案正式生效：全球首部综合性AI监管法规", "url": "https://example.com/eu-ai-act", "source": "路透社",
         "content": "欧盟《人工智能法案》正式生效，将AI系统分为四个风险等级，对高风险AI应用提出严格合规要求，违规最高罚3500万欧元。"},
        {"title": "Anthropic发布Claude模型系统卡：透明度新标杆", "url": "https://example.com/claude-card", "source": "AI安全周刊",
         "content": "Anthropic发布了最详尽的模型透明度报告，包括红队测试结果、偏见评估和部署建议，为AI透明度设立了新标准。"},
        {"title": "开源AI生态爆发：Hugging Face用户突破100万", "url": "https://example.com/hf", "source": "开源中国",
         "content": "Hugging Face注册用户突破100万，托管模型超50万个，开源AI正在民主化人工智能技术。"}
    ]


def main():
    print("=" * 70)
    print("  AI 周报生成代理（透明调试版 - 无框架）")
    print("=" * 70)

    # ==============================================================
    # 阶段1：搜索者 Agent — 检索新闻
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段1】搜索者 Agent — 检索 AI/ML 新闻")
    print(f"{'=' * 70}")
    print(f"  原教程使用 Tavily API 进行网络搜索")
    print(f"  本课使用模拟数据替代（无需 API key）")

    articles = get_articles()
    print(f"\n  检索到 {len(articles)} 篇文章:")
    for a in articles:
        print(f"    - {a['title']} ({a['source']})")

    # State 更新: articles 被填充
    print(f"\n  [State 更新] articles = [{len(articles)} 篇文章]")

    # ==============================================================
    # 阶段2：摘要者 Agent — 逐篇生成通俗摘要
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段2】摘要者 Agent — 逐篇生成通俗摘要")
    print(f"{'=' * 70}")

    system_prompt = (
        "你是一位AI领域专家，擅长将复杂技术话题用通俗易懂的语言解释给普通读者。"
        "请用2-3句话总结这篇文章，重点突出关键信息，用简单的语言解释专业术语。"
    )

    summaries = []
    for i, article in enumerate(articles, 1):
        print(f"\n  --- 第 {i} 篇: {article['title']} ---")

        print(f"\n  >>> 发送给 LLM <<<")
        print(f"  [system]: {system_prompt[:60]}...")
        print(f"  [user]: 标题: {article['title']} | 内容: {article['content'][:80]}...")

        summary = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"标题: {article['title']}\n\n内容: {article['content']}"}
        ], temperature=0.3)

        print(f"\n  >>> LLM 响应: {summary}")

        summaries.append({
            "title": article["title"],
            "summary": summary,
            "url": article["url"]
        })

    # State 更新: summaries 被填充
    print(f"\n  [State 更新] summaries = [{len(summaries)} 条摘要]")

    # ==============================================================
    # 阶段3：发布者 Agent — 编排 Markdown 报告
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段3】发布者 Agent — 编排 Markdown 报告")
    print(f"{'=' * 70}")

    summaries_text = "\n\n".join([
        f"标题: {item['title']}\n摘要: {item['summary']}\n来源: {item['url']}"
        for item in summaries
    ])

    report_prompt = (
        "请根据以下新闻摘要生成一份 AI/ML 周报。\n"
        "格式要求:\n"
        "1. 一段简短的导语（介绍本周亮点）\n"
        "2. 逐条新闻及摘要\n"
        "3. 总结展望\n"
        "语言通俗易懂，面向普通读者。用中文输出。"
    )

    print(f"\n  >>> 发送给 LLM: 编排报告 <<<")
    print(f"  [system]: {report_prompt[:60]}...")
    print(f"  [user]: {len(summaries)} 条摘要内容...")

    report = call_llm([
        {"role": "system", "content": report_prompt},
        {"role": "user", "content": summaries_text}
    ])

    # 保存文件
    current_date = datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.join(os.path.dirname(__file__), f"ai_周报_{current_date}_debug.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# AI/ML 周报 ({current_date})\n\n{report}")
    print(f"\n  报告已保存: {report_file}")

    # State 更新: report 被填充
    print(f"\n  [State 更新] report = [Markdown 报告]")

    # ==============================================================
    # 最终报告
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【最终报告】")
    print(f"{'=' * 70}")
    print(report)

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：LangGraph 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. StateGraph 定义了 GraphState: articles → summaries → report")
    print(f"  2. 三个节点线性串联: search → summarize → publish")
    print(f"  3. 没有条件路由，没有循环，是最简洁的 LangGraph 用法")
    print(f"  4. 三个 Agent 类各司其职:")
    print(f"     - NewsSearcher: 负责检索（原教程用 Tavily API）")
    print(f"     - Summarizer: 负责逐篇摘要（LLM 调用）")
    print(f"     - Publisher: 负责编排报告（LLM 调用 + 文件保存）")
    print(f"  5. 本质: 三个函数顺序调用 + 共享 dict 传递数据")
    print(f"     LangGraph 在这里的价值是：")
    print(f"     - 可视化流程图")
    print(f"     - 统一的状态管理")
    print(f"     - 方便后续扩展（加条件路由、循环等）")


if __name__ == "__main__":
    main()
