"""
第7课 - 科研论文助手（LangGraph 框架版）

核心概念：
  - 循环图：Agent ↔ Tools 循环调用，Judge → Planning 重试循环
  - 工具调用：LLM 自主决定调用哪个工具
  - 自我评估：Judge 节点评审答案质量，不合格打回重做

注意：由于第三方 API 与 langchain_openai 兼容性问题，
      本课直接使用 openai 库调用 LLM，同时保留 LangGraph 图结构。
"""

import os
import json
import re
from typing import TypedDict, Sequence, Annotated, List

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage

import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


class LLMResponse:
    """简单的响应包装，模拟 openai SDK 的返回格式"""
    def __init__(self, data: dict):
        self._data = data
        self.choices = [LLMChoice(c) for c in data.get("choices", [])]
        self.usage = data.get("usage", {})

class LLMChoice:
    def __init__(self, data: dict):
        self.finish_reason = data.get("finish_reason")
        self.message = LLMMessage(data.get("message", {}))

class LLMMessage:
    def __init__(self, data: dict):
        self.content = data.get("content", "")
        self.role = data.get("role", "assistant")
        raw_tc = data.get("tool_calls")
        self.tool_calls = [LLMToolCall(tc) for tc in raw_tc] if raw_tc else None

class LLMToolCall:
    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.function = LLMFunction(data.get("function", {}))

class LLMFunction:
    def __init__(self, data: dict):
        self.name = data.get("name", "")
        self.arguments = data.get("arguments", "{}")


def call_llm(messages: list, tools: list = None) -> LLMResponse:
    """用 httpx 直接调用 API（绕过 openai SDK 的请求头被 WAF 屏蔽的问题）"""
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
    return LLMResponse(resp.json())


def langchain_msgs_to_openai(messages: Sequence[BaseMessage]) -> list:
    """把 langchain 消息转成 openai 格式"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
    return result


# ========== 模拟论文数据（无需外部 API） ==========

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
        {"标题": "量子纠错码在容错量子计算中的应用", "作者": "吴九, 周十", "年份": 2023,
         "摘要": "研究了Surface Code和Steane Code在实际量子硬件上的表现。"},
    ],
    "大语言模型": [
        {"标题": "大语言模型的涌现能力研究", "作者": "孙一, 郑二", "年份": 2024,
         "摘要": "探讨了大模型在推理、代码生成、数学证明等方面展现的涌现能力，分析了其可能机制。"},
        {"标题": "基于RLHF的大语言模型对齐技术", "作者": "黄三, 林四", "年份": 2023,
         "摘要": "介绍了利用人类反馈强化学习（RLHF）进行模型对齐的方法和挑战。"},
    ],
}


# ========== 工具实现 ==========

def tool_search_papers(query: str, max_papers: int = 2) -> str:
    """搜索论文"""
    print(f'  [工具] 搜索论文: query="{query}", max={max_papers}')
    results = []
    for category, papers in MOCK_PAPERS.items():
        for paper in papers:
            if (query.lower() in category.lower()
                or query.lower() in paper["标题"].lower()
                or query.lower() in paper["摘要"].lower()):
                results.append(paper)
    if not results:
        results = list(MOCK_PAPERS.values())[0]
    results = results[:max_papers]
    output = "\n---\n".join([
        f"标题: {p['标题']}\n作者: {p['作者']}\n年份: {p['年份']}\n摘要: {p['摘要']}"
        for p in results
    ])
    print(f'  [工具] 找到 {len(results)} 篇论文')
    return output


def tool_download_paper(title: str) -> str:
    """下载论文"""
    print(f'  [工具] 下载论文: "{title}"')
    for papers in MOCK_PAPERS.values():
        for paper in papers:
            if title in paper["标题"] or paper["标题"] in title:
                return f"论文全文（模拟）：\n标题：{paper['标题']}\n作者：{paper['作者']}\n摘要：{paper['摘要']}\n结论：本文验证了所提方法的有效性。"
    return f"未找到标题为'{title}'的论文"


TOOLS_MAP = {"search_papers": tool_search_papers, "download_paper": tool_download_paper}

# OpenAI function calling 格式的工具定义
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索科研论文，根据关键词返回相关论文列表",
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
            "description": "下载并读取指定论文的全文内容",
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


# ========== State 定义 ==========

class AgentState(TypedDict):
    requires_research: bool
    num_feedback_requests: int
    is_good_answer: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ========== Prompts ==========

DECISION_PROMPT = """你是一个科研助手。根据用户的问题，判断是否需要搜索论文来回答。
- 学术/科研问题 → 需要研究
- 闲聊/常识 → 直接回答

用JSON回复（不要包含其他内容）：
- 需要研究：{"requires_research": true}
- 直接回答：{"requires_research": false, "answer": "你的回答"}"""

PLANNING_PROMPT = """你是科研助手。为用户问题制定研究计划。
可用工具：search_papers（搜索论文）、download_paper（下载论文全文）。
如果有评审反馈，请在新计划中改进。请制定具体的分步计划。"""

JUDGE_PROMPT = """评估回答质量。好的回答应直接回答问题、有论文引用。
用JSON回复：{"is_good_answer": true} 或 {"is_good_answer": false, "feedback": "改进建议"}"""


# ========== 工具函数：解析 JSON ==========

def parse_json(text: str, default: dict = None) -> dict:
    """容错 JSON 解析"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return default or {}


# ========== 节点定义 ==========

def decision_making_node(state: AgentState):
    """决策节点：判断是否需要研究"""
    print()
    print('=' * 50)
    print('【决策节点】判断是否需要研究')
    print('=' * 50)

    openai_msgs = [{"role": "system", "content": DECISION_PROMPT}]
    openai_msgs += langchain_msgs_to_openai(state["messages"])

    response = call_llm(openai_msgs)
    response_text = response.choices[0].message.content.strip()
    print(f'>>> 决策结果: {response_text[:200]}')

    result = parse_json(response_text, {"requires_research": True})

    output = {"requires_research": result.get("requires_research", True)}
    if result.get("answer"):
        output["messages"] = [AIMessage(content=result["answer"])]
        print(f'>>> 直接回答: {result["answer"][:100]}')
    else:
        print(f'>>> 需要研究，进入规划阶段')
    return output


def router(state: AgentState):
    return "planning" if state["requires_research"] else "end"


def planning_node(state: AgentState):
    """规划节点：制定研究计划"""
    print()
    print('=' * 50)
    print('【规划节点】制定研究计划')
    print('=' * 50)

    openai_msgs = [{"role": "system", "content": PLANNING_PROMPT}]
    openai_msgs += langchain_msgs_to_openai(state["messages"])

    response = call_llm(openai_msgs)
    content = response.choices[0].message.content.strip()
    print(f'>>> 研究计划: {content[:300]}...')
    return {"messages": [AIMessage(content=content)]}


def agent_node(state: AgentState):
    """Agent 节点：执行研究（可能调用工具）"""
    print()
    print('=' * 50)
    print('【Agent 节点】执行研究')
    print('=' * 50)

    openai_msgs = [{"role": "system", "content": "你是科研助手，按照计划使用工具完成研究。用中文回答。"}]
    openai_msgs += langchain_msgs_to_openai(state["messages"])

    response = call_llm(openai_msgs, tools=TOOLS_DEFINITION)
    choice = response.choices[0]

    if choice.message.tool_calls:
        # 有工具调用 → 转成 langchain AIMessage（带 tool_calls）
        tool_calls = []
        for tc in choice.message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            })
            print(f'>>> Agent 调用工具: {tc.function.name}({tc.function.arguments})')

        msg = AIMessage(content=choice.message.content or "", tool_calls=tool_calls)
    else:
        content = choice.message.content.strip()
        print(f'>>> Agent 最终回答: {content[:200]}...')
        msg = AIMessage(content=content)

    return {"messages": [msg]}


def tools_node(state: AgentState):
    """工具执行节点"""
    print()
    print('=' * 50)
    print('【工具节点】执行工具调用')
    print('=' * 50)

    outputs = []
    last_msg = state["messages"][-1]
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        print(f'>>> 执行: {name}({json.dumps(args, ensure_ascii=False)})')

        if name in TOOLS_MAP:
            result = TOOLS_MAP[name](**args)
        else:
            result = f"未知工具: {name}"
        print(f'>>> 返回: {result[:200]}')

        outputs.append(ToolMessage(content=result, name=name, tool_call_id=tc["id"]))

    return {"messages": outputs}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print('>>> → 继续调用工具（循环）')
        return "continue"
    print('>>> → 工具调用结束，进入评审')
    return "end"


def judge_node(state: AgentState):
    """评审节点"""
    print()
    print('=' * 50)
    num = state.get("num_feedback_requests", 0)
    print(f'【评审节点】第 {num + 1} 次评审')
    print('=' * 50)

    if num >= 2:
        print('>>> 已达最大重试次数，强制通过')
        return {"is_good_answer": True}

    openai_msgs = [{"role": "system", "content": JUDGE_PROMPT}]
    openai_msgs += langchain_msgs_to_openai(state["messages"])

    response = call_llm(openai_msgs)
    response_text = response.choices[0].message.content.strip()
    print(f'>>> 评审结果: {response_text[:200]}')

    result = parse_json(response_text, {"is_good_answer": True})

    output = {
        "is_good_answer": result.get("is_good_answer", True),
        "num_feedback_requests": num + 1,
    }

    if result.get("feedback"):
        output["messages"] = [AIMessage(content=f"评审反馈：{result['feedback']}")]
        print(f'>>> 不合格！反馈: {result["feedback"]}')
    else:
        print(f'>>> 通过！')

    return output


def final_answer_router(state: AgentState):
    if state["is_good_answer"]:
        return "end"
    print('>>> → 打回重做（循环回 planning）')
    return "planning"


# ========== 构建图工作流 ==========

def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("decision_making", decision_making_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("decision_making")

    workflow.add_conditional_edges("decision_making", router, {
        "planning": "planning", "end": END,
    })
    workflow.add_edge("planning", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "continue": "tools", "end": "judge",
    })
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("judge", final_answer_router, {
        "planning": "planning", "end": END,
    })

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
        try:
            png_data = app.get_graph().draw_mermaid_png()
            png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_graph.png')
            with open(png_path, 'wb') as f:
                f.write(png_data)
            print(f'\n>>> 图片已保存到: {png_path}')
        except Exception:
            print('\n>>> 把上面的 Mermaid 代码粘贴到 https://mermaid.live 查看')
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


def run_query(app, query: str):
    print()
    print('#' * 60)
    print(f'#  用户提问: {query}')
    print('#' * 60)

    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "requires_research": False,
        "num_feedback_requests": 0,
        "is_good_answer": False,
    })

    final = result["messages"][-1].content if result["messages"] else "无结果"
    print()
    print('=' * 60)
    print('【最终回答】')
    print('=' * 60)
    print(final)
    print()
    return final


if __name__ == '__main__':
    print('第7课 - 科研论文助手')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()

    app = build_workflow()
    print_graph(app)

    examples = [
        "帮我搜索关于大语言模型的最新论文",
        "你好，今天天气怎么样？",
    ]

    print('--- 示例问题 ---')
    for q in examples:
        run_query(app, q)

    print('\n输入问题，输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        run_query(app, user_input)
