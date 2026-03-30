"""
第8课 - Chiron 费曼学习导师（LangGraph 框架版）

核心概念：
  - Human-in-the-Loop：出题后暂停等用户回答
  - 检查点循环：逐个检查点 出题→验证→教学/下一个
  - 费曼教学：理解度不足时用简单语言+类比重新解释
  - 多路由决策：验证后三路分支（教学/下一个/结束）

简化版：去掉 Tavily 搜索和嵌入，聚焦核心流程。
"""

import os
import json
import re
from typing import TypedDict, List, Annotated, Sequence

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM，返回文本"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_from_text(text: str, default: dict = None) -> dict:
    """从 LLM 响应中提取 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 JSON 块
        match = re.search(r'```json\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        # 尝试提取花括号或方括号
        match = re.search(r'[\[{][\s\S]*[\]}]', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return default or {}


# ========== State 定义 ==========

class LearningState(TypedDict):
    topic: str                      # 学习主题
    goals: str                      # 学习目标
    context: str                    # 学习材料
    checkpoints: list               # 检查点列表 [{description, criteria, verification}]
    current_checkpoint: int         # 当前检查点索引
    current_question: str           # 当前出的题
    current_answer: str             # 用户的回答
    understanding_level: float      # 理解度 0~1
    feedback: str                   # 验证反馈
    teaching: dict                  # 费曼教学内容


# ========== 节点实现 ==========

def generate_checkpoints_node(state: LearningState):
    """生成学习检查点"""
    print()
    print('=' * 50)
    print('【生成检查点】')
    print('=' * 50)

    system = """你是学习导师。根据学习主题和目标，生成3个学习检查点。
每个检查点包含：description（描述）、criteria（3个验收标准）、verification（验证方式）。
用JSON数组格式回复，例如：
[{"description": "理解基本概念", "criteria": ["能解释什么是X", "能区分X和Y", "能举出X的例子"], "verification": "用自己的话解释X"}]"""

    prompt = f"主题：{state['topic']}\n目标：{state['goals']}"
    if state.get('context'):
        prompt += f"\n参考材料（节选）：{state['context'][:500]}"

    response = call_llm(prompt, system)
    checkpoints = parse_json_from_text(response, [])

    if isinstance(checkpoints, dict):
        checkpoints = checkpoints.get("checkpoints", [checkpoints])
    if not isinstance(checkpoints, list) or not checkpoints:
        checkpoints = [{"description": "理解核心概念", "criteria": ["能用自己的话解释"], "verification": "简述核心概念"}]

    print(f'>>> 生成了 {len(checkpoints)} 个检查点:')
    for i, cp in enumerate(checkpoints):
        print(f'    {i+1}. {cp.get("description", "?")}')

    return {"checkpoints": checkpoints, "current_checkpoint": 0}


def generate_question_node(state: LearningState):
    """根据当前检查点出题"""
    idx = state["current_checkpoint"]
    checkpoints = state["checkpoints"]

    if idx >= len(checkpoints):
        return {"current_question": ""}

    cp = checkpoints[idx]

    print()
    print('=' * 50)
    print(f'【出题】检查点 {idx + 1}/{len(checkpoints)}')
    print('=' * 50)
    print(f'>>> 检查点: {cp.get("description", "")}')

    system = "你是学习导师。根据检查点信息出一道验证理解的题目。只输出题目本身，不要其他内容。"
    prompt = f"""检查点描述：{cp.get('description', '')}
验收标准：{json.dumps(cp.get('criteria', []), ensure_ascii=False)}
验证方式：{cp.get('verification', '')}"""

    question = call_llm(prompt, system)
    print(f'>>> 题目: {question}')

    return {"current_question": question}


def user_answer_node(state: LearningState):
    """等待用户回答（Human-in-the-Loop）"""
    print()
    print('-' * 50)
    print(f'题目: {state["current_question"]}')
    print('-' * 50)
    answer = input('你的回答: ').strip()
    if not answer:
        answer = "不知道"
    return {"current_answer": answer}


def verify_answer_node(state: LearningState):
    """验证用户回答"""
    idx = state["current_checkpoint"]
    cp = state["checkpoints"][idx]

    print()
    print('=' * 50)
    print('【验证回答】')
    print('=' * 50)

    system = """你是学习评估专家。评估学生回答的质量。
用JSON格式回复：{"understanding_level": 0.0到1.0, "feedback": "详细反馈", "suggestions": ["建议1"]}"""

    prompt = f"""题目：{state['current_question']}
学生回答：{state['current_answer']}

检查点：{cp.get('description', '')}
验收标准：{json.dumps(cp.get('criteria', []), ensure_ascii=False)}"""

    if state.get('context'):
        prompt += f"\n参考材料（节选）：{state['context'][:300]}"

    response = call_llm(prompt, system)
    result = parse_json_from_text(response, {"understanding_level": 0.5, "feedback": response})

    level = float(result.get("understanding_level", 0.5))
    feedback = result.get("feedback", "")

    print(f'>>> 理解度: {level:.0%}')
    print(f'>>> 反馈: {feedback[:200]}')

    return {"understanding_level": level, "feedback": feedback}


def teach_concept_node(state: LearningState):
    """费曼教学：用简单语言和类比重新解释"""
    idx = state["current_checkpoint"]
    cp = state["checkpoints"][idx]

    print()
    print('=' * 50)
    print('【费曼教学】理解度不足，用简单方式重新解释')
    print('=' * 50)

    system = """你是费曼学习法导师。学生理解不够，请用以下方式重新教学：
1. 用最简单的语言解释（避免术语）
2. 给出生活中的类比
3. 列出必须记住的核心概念
用JSON格式回复：{"explanation": "简单解释", "analogies": ["类比1"], "key_concepts": ["概念1"]}"""

    prompt = f"""检查点：{cp.get('description', '')}
学生的回答：{state['current_answer']}
评估反馈：{state['feedback']}"""

    if state.get('context'):
        prompt += f"\n参考材料：{state['context'][:300]}"

    response = call_llm(prompt, system)
    teaching = parse_json_from_text(response, {"explanation": response, "analogies": [], "key_concepts": []})

    print(f'>>> 简化解释: {teaching.get("explanation", "")[:300]}')
    if teaching.get("analogies"):
        print(f'>>> 类比: {teaching["analogies"]}')
    if teaching.get("key_concepts"):
        print(f'>>> 核心概念: {teaching["key_concepts"]}')

    return {"teaching": teaching}


def next_checkpoint_node(state: LearningState):
    """前进到下一个检查点"""
    next_idx = state["current_checkpoint"] + 1
    print(f'\n>>> 前进到检查点 {next_idx + 1}')
    return {"current_checkpoint": next_idx}


# ========== 路由函数 ==========

def route_verification(state: LearningState):
    """验证后路由：不及格→教学，及格且有下一个→下一个，全部完成→结束"""
    level = state.get("understanding_level", 0)
    idx = state["current_checkpoint"]
    total = len(state["checkpoints"])

    if level < 0.7:
        print(f'>>> 理解度 {level:.0%} < 70%，进入费曼教学')
        return "teach_concept"
    if idx + 1 < total:
        print(f'>>> 理解度 {level:.0%} ≥ 70%，进入下一个检查点')
        return "next_checkpoint"
    print(f'>>> 所有检查点完成！')
    return "end"


def route_teaching(state: LearningState):
    """教学后路由：还有检查点→下一个，否则结束"""
    idx = state["current_checkpoint"]
    total = len(state["checkpoints"])
    if idx + 1 < total:
        return "next_checkpoint"
    return "end"


# ========== 构建图工作流 ==========

def build_workflow():
    workflow = StateGraph(LearningState)

    workflow.add_node("generate_checkpoints", generate_checkpoints_node)
    workflow.add_node("generate_question", generate_question_node)
    workflow.add_node("user_answer", user_answer_node)
    workflow.add_node("verify_answer", verify_answer_node)
    workflow.add_node("teach_concept", teach_concept_node)
    workflow.add_node("next_checkpoint", next_checkpoint_node)

    # 流程
    workflow.add_edge(START, "generate_checkpoints")
    workflow.add_edge("generate_checkpoints", "generate_question")
    workflow.add_edge("generate_question", "user_answer")
    workflow.add_edge("user_answer", "verify_answer")

    # 验证后三路分支
    workflow.add_conditional_edges("verify_answer", route_verification, {
        "teach_concept": "teach_concept",
        "next_checkpoint": "next_checkpoint",
        "end": END,
    })

    # 教学后路由
    workflow.add_conditional_edges("teach_concept", route_teaching, {
        "next_checkpoint": "next_checkpoint",
        "end": END,
    })

    # 下一个检查点 → 重新出题（循环）
    workflow.add_edge("next_checkpoint", "generate_question")

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
        print('\n>>> 粘贴到 https://mermaid.live 查看可视化')
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第8课 - Chiron 费曼学习导师')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 选择学习主题
    print('请输入学习主题和目标（或直接回车用默认示例）')
    topic_input = input('学习主题（如"Python装饰器"）: ').strip()
    goals_input = input('学习目标（如"理解装饰器原理并能手写"）: ').strip()

    if not topic_input:
        topic_input = "Python装饰器"
        goals_input = "理解装饰器的本质，能手写一个带参数的装饰器"
        print(f'\n使用默认主题: {topic_input}')
        print(f'默认目标: {goals_input}')

    context_input = input('学习材料（可选，直接回车跳过）: ').strip()

    initial_state = {
        "topic": topic_input,
        "goals": goals_input,
        "context": context_input,
        "checkpoints": [],
        "current_checkpoint": 0,
        "current_question": "",
        "current_answer": "",
        "understanding_level": 0.0,
        "feedback": "",
        "teaching": {},
    }

    print()
    print('#' * 60)
    print(f'#  开始学习: {topic_input}')
    print('#' * 60)

    result = app.invoke(initial_state)

    print()
    print('=' * 60)
    print('【学习完成】')
    print('=' * 60)
    print(f'主题: {result["topic"]}')
    print(f'完成检查点: {result["current_checkpoint"] + 1}/{len(result["checkpoints"])}')
    print(f'最终理解度: {result.get("understanding_level", 0):.0%}')
    print()
