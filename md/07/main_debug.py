"""
第7课 - 科研论文助手内部机制演示（不使用 LangGraph）

目的：让你看清循环图的本质：
  1. 决策节点 = 一次 LLM 调用判断走哪个分支
  2. Agent ↔ Tools 循环 = while 循环，直到 LLM 不再调用工具
  3. Judge 评审 = 又一次 LLM 调用判断质量
  4. 不合格重试 = 外层 for 循环，最多重试 N 次

对比 main.py（LangGraph 框架版），理解框架帮你封装了什么：
  - add_conditional_edges → 就是 if-else
  - 图中的环 → 就是 while/for 循环
  - add_messages reducer → 就是 list.append()
"""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


# ================================================================
#  【数据层】模拟论文数据
# ================================================================

MOCK_PAPERS = {
    "机器学习": [
        {"标题": "深度学习在自然语言处理中的应用综述", "作者": "张三, 李四", "年份": 2024,
         "摘要": "本文综述了深度学习技术在NLP领域的最新进展，包括Transformer架构、预训练语言模型和大语言模型。"},
        {"标题": "基于注意力机制的图像分类方法研究", "作者": "王五, 赵六", "年份": 2023,
         "摘要": "提出了一种改进的多头注意力机制，在ImageNet上达到了96.2%的准确率。"},
    ],
    "量子计算": [
        {"标题": "量子机器学习算法的最新进展", "作者": "刘七, 陈八", "年份": 2024,
         "摘要": "系统介绍了量子支持向量机、量子神经网络和变分量子算法在分类任务中的应用。"},
    ],
    "大语言模型": [
        {"标题": "大语言模型的涌现能力研究", "作者": "孙一, 郑二", "年份": 2024,
         "摘要": "探讨了大模型在推理、代码生成、数学证明等方面展现的涌现能力。"},
        {"标题": "基于RLHF的大语言模型对齐技术", "作者": "黄三, 林四", "年份": 2023,
         "摘要": "介绍了利用人类反馈强化学习（RLHF）进行模型对齐的方法和挑战。"},
    ],
}


# ================================================================
#  【工具层】
# ================================================================

def tool_search_papers(query: str, max_papers: int = 2) -> str:
    """搜索论文"""
    results = []
    for category, papers in MOCK_PAPERS.items():
        for paper in papers:
            if (query in category or query in paper["标题"] or query in paper["摘要"]):
                results.append(paper)
    if not results:
        results = list(MOCK_PAPERS.values())[0]
    results = results[:max_papers]
    return "\n---\n".join([
        f"标题: {p['标题']}\n作者: {p['作者']}\n年份: {p['年份']}\n摘要: {p['摘要']}"
        for p in results
    ])


def tool_download_paper(title: str) -> str:
    """下载论文"""
    for papers in MOCK_PAPERS.values():
        for p in papers:
            if title in p["标题"] or p["标题"] in title:
                return f"论文全文（模拟）：\n标题：{p['标题']}\n摘要：{p['摘要']}\n结论：本文验证了所提方法的有效性。"
    return f"未找到'{title}'"


TOOLS_MAP = {
    "search_papers": tool_search_papers,
    "download_paper": tool_download_paper,
}

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索科研论文。根据关键词返回相关论文列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_papers": {"type": "integer", "description": "最大返回数量", "default": 2},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_paper",
            "description": "下载论文全文",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "论文标题"},
                },
                "required": ["title"],
            },
        },
    },
]


# ================================================================
#  【LLM 层】
# ================================================================

def call_llm(messages: list, tools: list = None) -> dict:
    """用 httpx 直接调用 API（绕过 openai SDK 被 WAF 屏蔽的问题）"""
    body = {"model": MODEL_NAME, "messages": messages, "temperature": 0}
    if tools:
        body["tools"] = tools

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ================================================================
#  【Agent 层】完整流程
# ================================================================

def run_paper_agent(query: str) -> str:
    """
    科研论文助手的完整流程。这就是 LangGraph 的循环图内部做的事。

    Java 类比：
        public String process(String query) {
            // 1. 决策
            if (!needsResearch(query)) return directAnswer(query);

            // 2. 外层循环：Judge 最多打回2次
            for (int retry = 0; retry < MAX_RETRIES; retry++) {
                String plan = planResearch(query);

                // 3. 内层循环：Agent ↔ Tools
                while (true) {
                    AgentResult result = agent.run(messages);
                    if (!result.hasToolCalls()) break;
                    executeTools(result.getToolCalls());
                }

                // 4. Judge 评审
                if (judgeAnswer(messages)) return getFinalAnswer();
            }
            return getFinalAnswer();  // 超过重试次数，强制返回
        }
    """

    messages = [{"role": "user", "content": query}]

    # ==========================================
    # 第1步：决策节点
    # ==========================================
    print()
    print('=' * 60)
    print('【1 - 决策节点】判断是否需要研究')
    print('=' * 60)

    decision_prompt = """判断用户问题是否需要搜索论文来回答。
用JSON回复：{"requires_research": true} 或 {"requires_research": false, "answer": "直接回答"}"""

    decision_response = call_llm([
        {"role": "system", "content": decision_prompt},
        {"role": "user", "content": query},
    ])
    decision_text = decision_response["choices"][0]["message"]["content"].strip()
    print(f'>>> 决策结果: {decision_text}')

    try:
        import re
        match = re.search(r'\{[^}]+\}', decision_text)
        decision = json.loads(match.group()) if match else {"requires_research": True}
    except:
        decision = {"requires_research": True}

    # 不需要研究 → 直接返回（对应图中 decision_making → END 的边）
    if not decision.get("requires_research", True):
        answer = decision.get("answer", decision_text)
        print(f'>>> 直接回答（跳过研究）: {answer}')
        return answer

    print('>>> 需要研究，进入规划')

    # ==========================================
    # 第2步：外层循环 — Judge 重试（最多2次）
    # ==========================================
    # 这就是 LangGraph 中 judge → planning 的循环边
    MAX_RETRIES = 2

    for retry in range(MAX_RETRIES + 1):
        # ==========================================
        # 第3步：规划节点
        # ==========================================
        print()
        print('=' * 60)
        print(f'【2 - 规划节点】制定研究计划（第 {retry + 1} 轮）')
        print('=' * 60)

        planning_messages = [
            {"role": "system", "content": "你是科研助手。为用户问题制定研究计划。可用工具：search_papers（搜索论文）、download_paper（下载论文）。"},
        ] + messages

        plan_response = call_llm(planning_messages)
        plan = plan_response["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": plan})
        print(f'>>> 研究计划: {plan[:300]}...')

        # ==========================================
        # 第4步：内层循环 — Agent ↔ Tools
        # ==========================================
        # 这就是 LangGraph 中 agent ↔ tools 的循环边
        MAX_TOOL_ROUNDS = 5

        for tool_round in range(MAX_TOOL_ROUNDS):
            print()
            print('=' * 60)
            print(f'【3 - Agent 节点】执行研究（工具调用第 {tool_round + 1} 轮）')
            print('=' * 60)

            agent_messages = [
                {"role": "system", "content": "你是科研助手，按计划使用工具完成研究。用中文回答。"},
            ] + messages

            agent_response = call_llm(agent_messages, tools=TOOLS_DEFINITION)
            choice = agent_response["choices"][0]
            msg = choice["message"]

            # 没有工具调用 → 退出内层循环（对应 should_continue 返回 "end"）
            if not msg.get("tool_calls"):
                final_content = msg["content"]
                messages.append({"role": "assistant", "content": final_content})
                print(f'>>> Agent 完成，不再调用工具')
                print(f'>>> 回答: {final_content[:200]}...')
                break

            # 有工具调用 → 执行工具（对应 should_continue 返回 "continue"）
            messages.append(msg)

            for tc in msg["tool_calls"]:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                print(f'>>> 调用工具: {func_name}({json.dumps(func_args, ensure_ascii=False)})')

                # 执行工具
                if func_name in TOOLS_MAP:
                    result = TOOLS_MAP[func_name](**func_args)
                else:
                    result = f"未知工具: {func_name}"
                print(f'>>> 工具返回: {result[:200]}...')

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # 继续内层循环（Agent 会看到工具结果，决定下一步）

        # ==========================================
        # 第5步：Judge 评审
        # ==========================================
        print()
        print('=' * 60)
        print(f'【4 - 评审节点】第 {retry + 1} 次评审')
        print('=' * 60)

        if retry >= MAX_RETRIES:
            print('>>> 已达最大重试次数，强制通过')
            break

        judge_prompt = """评估回答质量。好的回答应该直接回答问题、有论文引用。
用JSON回复：{"is_good_answer": true} 或 {"is_good_answer": false, "feedback": "改进建议"}"""

        judge_response = call_llm([
            {"role": "system", "content": judge_prompt},
        ] + messages)
        judge_text = judge_response["choices"][0]["message"]["content"].strip()
        print(f'>>> 评审结果: {judge_text}')

        try:
            match = re.search(r'\{[^}]+\}', judge_text)
            judge_result = json.loads(match.group()) if match else {"is_good_answer": True}
        except:
            judge_result = {"is_good_answer": True}

        if judge_result.get("is_good_answer", True):
            print('>>> 通过！')
            break
        else:
            feedback = judge_result.get("feedback", "需要改进")
            print(f'>>> 不合格！反馈: {feedback}')
            messages.append({"role": "assistant", "content": f"评审反馈：{feedback}"})
            print('>>> → 回到规划节点重做（外层循环）')
            # 继续外层 for 循环

    # 提取最终回答
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "未能生成回答"


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第7课 - 科研论文助手内部机制演示（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示循环图的本质：')
    print('  - Agent ↔ Tools 循环 = while 循环')
    print('  - Judge → Planning 重试 = for 循环（最多2次）')
    print('  - 条件路由 = if-else')
    print()

    examples = [
        "帮我搜索关于大语言模型的最新论文",
        "你好，今天天气怎么样？",
    ]

    print('--- 示例问题 ---')
    for q in examples:
        print()
        print('#' * 60)
        print(f'#  用户提问: {q}')
        print('#' * 60)

        result = run_paper_agent(q)

        print()
        print('=' * 60)
        print('【最终回答】')
        print('=' * 60)
        print(result)
        print()

    # 交互模式
    print('\n输入问题，输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break

        print()
        print('#' * 60)
        print(f'#  用户提问: {user_input}')
        print('#' * 60)

        result = run_paper_agent(user_input)

        print()
        print('=' * 60)
        print('【最终回答】')
        print('=' * 60)
        print(result)
        print()
