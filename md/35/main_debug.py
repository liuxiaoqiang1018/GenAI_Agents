"""
第35课：新闻记者 AI 助手（透明调试版）

不使用任何框架，纯手写意图分类 + 5模块分析流程。
让你看清多路条件路由和多模块分析的内部机制。
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
def call_llm(messages: list, temperature: float = 0.3) -> str:
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


# ===== 模拟文章 =====
SAMPLE_ARTICLE = """
人工智能正在重塑全球医疗健康产业。据世界卫生组织最新报告显示，2025年全球AI医疗市场规模已达450亿美元，预计到2030年将突破1500亿美元。

谷歌DeepMind研发的AlphaFold系列模型已成功预测超过2亿种蛋白质结构，被认为是生物学领域近年来最重要的突破之一。DeepMind CEO德米斯·哈萨比斯在接受采访时表示："AlphaFold的影响才刚刚开始，未来五年内，它将彻底改变药物研发的方式。"

然而，AI在医疗领域的应用也面临挑战。北京协和医院院长张抒扬指出："AI辅助诊断系统虽然在影像识别方面表现出色，准确率高达97%，但在复杂病例的综合判断上仍需要医生的经验和直觉。"她同时强调，患者数据隐私保护是AI医疗落地的关键前提。

值得注意的是，欧盟《人工智能法案》已将医疗AI系统列为"高风险"类别，要求开发者必须提供透明的算法解释和严格的临床验证数据。美国FDA也加快了AI医疗器械的审批流程，2024年已批准超过200款AI医疗软件。

中国在AI医疗领域也取得了显著进展。百度推出的灵医大模型已在全国超过500家医院部署，覆盖影像诊断、病历生成和药物推荐等场景。腾讯觅影系统在早期癌症筛查中的准确率已接近资深放射科医生水平。

业内专家普遍认为，AI不会取代医生，而是成为医生的"超级助手"。未来的医疗模式将是"AI + 医生"的深度协作，让优质医疗资源惠及更多人群。
"""


# ===== 5个分析模块的 Prompt =====
ANALYSIS_MODULES = {
    "summarization": {
        "name": "摘要",
        "prompt": "请用150-200字总结以下文章，关注主要事件、关键人物和重要数据。使用中立客观的新闻报道语气。\n\n文章:\n{article}"
    },
    "fact-checking": {
        "name": "事实核查",
        "prompt": "对以下文章进行事实核查。逐条检查关键声明的准确性。\n对每个声明标注状态: 已确认/可疑/无法验证/模糊\n并给出简短解释。用中文回答。\n\n文章:\n{article}"
    },
    "tone-analysis": {
        "name": "语气分析",
        "prompt": "分析以下文章的语气和立场。判断它是中立、正面、批判还是有倾向性的？\n用文章中的具体例子支持你的分析。用中文回答。\n\n文章:\n{article}"
    },
    "quote-extraction": {
        "name": "引语提取",
        "prompt": "识别以下文章中的直接引用（引号内的内容），标注说话人和上下文。\n如果没有引用，回复'文章中未发现直接引用'。用中文回答。\n\n文章:\n{article}"
    },
    "grammar-and-bias-review": {
        "name": "语法偏见审查",
        "prompt": "审查以下文章的语法、拼写、标点问题，以及是否存在偏见倾向。\n列出发现的问题和改进建议。用中文回答。\n\n文章:\n{article}"
    }
}


def main():
    print("=" * 70)
    print("  新闻记者 AI 助手（透明调试版 - 无框架）")
    print("=" * 70)

    query = "请对这篇文章进行全部分析，生成完整报告"
    print(f"  用户查询: {query}")
    print(f"  文章长度: {len(SAMPLE_ARTICLE)} 字")

    # ==============================================================
    # 阶段1：意图分类（对应 categorize_user_input 节点）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段1】意图分类 — LLM 识别用户需求")
    print(f"{'=' * 70}")

    classify_prompt = (
        "根据用户输入，识别需要执行的分析操作。\n"
        "可选操作: summarization, fact-checking, tone-analysis, quote-extraction, grammar-and-bias-review\n"
        "如果用户要求'全部报告'或'完整分析'，返回所有操作。\n"
        "如果输入无关，返回 invalid。\n\n"
        f"用户输入: {query}\n\n"
        "只返回操作列表，用逗号分隔，不要其他内容。"
    )

    print(f"\n  >>> 发送给 LLM <<<")
    print(f"  {'-' * 60}")
    print(f"  {classify_prompt}")
    print(f"  {'-' * 60}")

    result = call_llm([{"role": "user", "content": classify_prompt}])
    print(f"\n  >>> LLM 响应: {result}")

    actions = [a.strip().lower() for a in result.split(",")]
    valid_actions = set(ANALYSIS_MODULES.keys())
    actions = [a for a in actions if a in valid_actions]

    if not actions:
        actions = list(valid_actions)

    print(f"  最终操作列表: {actions}")

    # ==============================================================
    # 阶段2：路由决策（对应 add_conditional_edges）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段2】路由决策 — 根据 actions 分发到模块")
    print(f"{'=' * 70}")
    print(f"  这就是 LangGraph 的 add_conditional_edges:")
    print(f"  actions 列表中的每个操作对应一个节点")
    for a in actions:
        module = ANALYSIS_MODULES[a]
        print(f"    {a} → 【{module['name']}】节点")

    # ==============================================================
    # 阶段3：逐个执行分析模块
    # ==============================================================
    results = {}

    for action in actions:
        module = ANALYSIS_MODULES[action]

        print(f"\n{'=' * 70}")
        print(f"【{module['name']}模块】执行分析 — LLM 调用")
        print(f"{'=' * 70}")

        prompt = module["prompt"].format(article=SAMPLE_ARTICLE)

        print(f"\n  >>> 发送给 LLM <<<")
        print(f"  Prompt 前100字: {prompt[:100]}...")

        analysis_result = call_llm([{"role": "user", "content": prompt}])
        results[action] = analysis_result

        print(f"\n  >>> LLM 响应:")
        print(f"  {analysis_result[:300]}...")

    # ==============================================================
    # 阶段4：格式化 Markdown 报告
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段4】格式化 Markdown 报告")
    print(f"{'=' * 70}")

    date = datetime.now().strftime("%Y-%m-%d")
    report = f"# 文章分析报告 ({date})\n\n"
    report += f"**用户查询**: {query}\n"
    report += f"**执行操作**: {', '.join(actions)}\n\n"

    section_map = {
        "summarization": "文章摘要",
        "fact-checking": "事实核查",
        "tone-analysis": "语气分析",
        "quote-extraction": "引语提取",
        "grammar-and-bias-review": "语法与偏见审查"
    }

    for action in actions:
        if action in results:
            report += f"## {section_map.get(action, action)}\n\n{results[action]}\n\n"

    # 保存
    report_file = os.path.join(os.path.dirname(__file__), f"文章分析报告_{date}_debug.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: {report_file}")

    print(f"\n{'=' * 70}")
    print(f"【完整报告】")
    print(f"{'=' * 70}")
    print(report)

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：LangGraph 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. StateGraph 定义了 State，包含 query、article、actions 和 5 个分析结果")
    print(f"  2. 意图分类节点: LLM 结构化输出 actions 列表")
    print(f"  3. add_conditional_edges: actions 映射到 5 个分析节点")
    print(f"     原教程用 lambda 返回 actions 列表实现多路分发")
    print(f"  4. 每个分析模块本质就是: Prompt模板 + LLM调用")
    print(f"     - 摘要: Map-Reduce（分块→摘要→合并）")
    print(f"     - 事实核查: 结构化 JSON 输出")
    print(f"     - 语气/引语/语法: 直接 LLM 分析")
    print(f"  5. 本质: if/for 循环 + 5次 LLM 调用 + 字符串拼接")
    print(f"     框架的价值: 可视化流程 + 统一状态管理 + 方便扩展")


if __name__ == "__main__":
    main()
