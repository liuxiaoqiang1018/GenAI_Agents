"""
第13课 - 项目管理助手（LangGraph 框架版）

核心概念：
  - 自反思循环：排期→分配→风险评估→(风险高?)→洞察→重新排期
  - 结构化输出：LLM返回JSON，解析为dict
  - 迭代累积：每轮结果存入历史，下一轮参考上一轮
  - 6个节点：任务生成、依赖分析、排期、分配、风险评估、洞察生成
"""

import os
import json
import re
from typing import TypedDict, List

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extract_json(text: str) -> dict:
    """从LLM回复中提取JSON"""
    # 尝试找 ```json ... ``` 块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找第一个 { 或 [ 开始的部分
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        print(f'>>> JSON解析失败，原始回复: {text[:200]}')
        return {}


# ========== State ==========

class State(TypedDict):
    project_description: str        # 项目描述
    team: list                      # 团队成员列表 [{"name": "...", "profile": "..."}]
    tasks: list                     # 任务列表
    dependencies: list              # 依赖关系
    schedule: list                  # 当前排期
    allocations: list               # 当前人员分配
    risks: list                     # 当前风险评估
    insights: str                   # 改进洞察
    iteration: int                  # 当前迭代次数
    max_iteration: int              # 最大迭代次数
    risk_scores: List[int]          # 每轮风险总分
    schedule_history: List[list]    # 排期历史
    allocation_history: List[list]  # 分配历史


# ========== 节点：任务生成 ==========

def task_generation_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 任务生成】')
    print('=' * 60)

    prompt = (f"你是资深项目经理。分析以下项目描述，提取可执行的任务列表。\n\n"
              f"项目描述：{state['project_description']}\n\n"
              f"要求：\n"
              f"1. 列出所有可执行任务，每个任务估算天数\n"
              f"2. 超过5天的任务拆分为子任务\n"
              f"3. 任务要具体、可执行\n\n"
              f"返回JSON格式：\n"
              f'{{"tasks": [{{"name": "任务名", "description": "描述", "days": 3}}]}}')

    result = extract_json(call_llm(prompt))
    tasks = result.get("tasks", [])

    print(f'>>> 生成 {len(tasks)} 个任务:')
    for t in tasks:
        print(f'    - {t["name"]} ({t.get("days", "?")}天)')

    return {"tasks": tasks}


# ========== 节点：依赖分析 ==========

def task_dependency_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 依赖分析】')
    print('=' * 60)

    prompt = (f"分析以下任务之间的依赖关系。\n\n"
              f"任务列表：{json.dumps(state['tasks'], ensure_ascii=False)}\n\n"
              f"返回JSON格式：\n"
              f'{{"dependencies": [{{"task": "任务A", "depends_on": ["任务B", "任务C"]}}]}}')

    result = extract_json(call_llm(prompt))
    deps = result.get("dependencies", [])

    print(f'>>> 依赖关系:')
    for d in deps:
        depends = d.get("depends_on", [])
        if depends:
            print(f'    {d["task"]} → 依赖: {", ".join(depends)}')
        else:
            print(f'    {d["task"]} → 无依赖（可立即开始）')

    return {"dependencies": deps}


# ========== 节点：任务排期 ==========

def task_scheduler_node(state: State):
    print()
    print('=' * 60)
    iteration = state["iteration"] + 1
    print(f'【3 - 任务排期】（第{iteration}轮）')
    print('=' * 60)

    insights_part = ""
    if state["insights"]:
        insights_part = f"\n上一轮改进洞察：{state['insights']}\n上一轮排期历史：{json.dumps(state['schedule_history'], ensure_ascii=False)}\n"

    prompt = (f"你是项目排期专家。根据任务和依赖关系安排排期。\n\n"
              f"任务：{json.dumps(state['tasks'], ensure_ascii=False)}\n"
              f"依赖：{json.dumps(state['dependencies'], ensure_ascii=False)}\n"
              f"{insights_part}\n"
              f"要求：\n"
              f"1. 尊重依赖关系\n"
              f"2. 尽量并行化以缩短总工期\n"
              f"3. 如果有上轮洞察，据此优化\n\n"
              f"返回JSON格式：\n"
              f'{{"schedule": [{{"task": "任务名", "start_day": 1, "end_day": 3}}]}}')

    result = extract_json(call_llm(prompt))
    schedule = result.get("schedule", [])

    print(f'>>> 排期:')
    for s in schedule:
        print(f'    {s["task"]}: 第{s["start_day"]}天 → 第{s["end_day"]}天')

    history = state["schedule_history"] + [schedule]
    return {"schedule": schedule, "schedule_history": history}


# ========== 节点：人员分配 ==========

def task_allocation_node(state: State):
    print()
    print('=' * 60)
    print(f'【4 - 人员分配】')
    print('=' * 60)

    insights_part = ""
    if state["insights"]:
        insights_part = f"\n上一轮洞察：{state['insights']}\n上一轮分配历史：{json.dumps(state['allocation_history'], ensure_ascii=False)}\n"

    prompt = (f"你是项目经理。根据排期和团队技能分配任务。\n\n"
              f"排期：{json.dumps(state['schedule'], ensure_ascii=False)}\n"
              f"团队：{json.dumps(state['team'], ensure_ascii=False)}\n"
              f"任务详情：{json.dumps(state['tasks'], ensure_ascii=False)}\n"
              f"{insights_part}\n"
              f"要求：\n"
              f"1. 根据成员技能匹配任务\n"
              f"2. 避免同一时间段分配重叠任务\n"
              f"3. 均衡工作量\n\n"
              f"返回JSON格式：\n"
              f'{{"allocations": [{{"task": "任务名", "member": "成员名", "reason": "分配原因"}}]}}')

    result = extract_json(call_llm(prompt))
    allocations = result.get("allocations", [])

    print(f'>>> 人员分配:')
    for a in allocations:
        print(f'    {a["task"]} → {a["member"]}')

    history = state["allocation_history"] + [allocations]
    return {"allocations": allocations, "allocation_history": history}


# ========== 节点：风险评估 ==========

def risk_assessment_node(state: State):
    print()
    print('=' * 60)
    print(f'【5 - 风险评估】')
    print('=' * 60)

    prompt = (f"你是项目风险分析师。评估当前项目计划的风险。\n\n"
              f"排期：{json.dumps(state['schedule'], ensure_ascii=False)}\n"
              f"人员分配：{json.dumps(state['allocations'], ensure_ascii=False)}\n"
              f"团队：{json.dumps(state['team'], ensure_ascii=False)}\n\n"
              f"要求：\n"
              f"1. 每个任务给0-10的风险评分\n"
              f"2. 考虑任务复杂度、人员匹配度、时间紧迫性\n"
              f"3. 计算总风险分\n\n"
              f"返回JSON格式：\n"
              f'{{"risks": [{{"task": "任务名", "score": 5, "reason": "原因"}}], "total_score": 30}}')

    result = extract_json(call_llm(prompt))
    risks = result.get("risks", [])
    total = result.get("total_score", sum(r.get("score", 0) for r in risks))

    print(f'>>> 风险评估:')
    for r in risks:
        print(f'    {r["task"]}: {r.get("score", "?")}分 - {r.get("reason", "")}')
    print(f'>>> 总风险分: {total}')

    new_scores = state["risk_scores"] + [total]
    new_iteration = state["iteration"] + 1

    return {"risks": risks, "risk_scores": new_scores, "iteration": new_iteration}


# ========== 节点：洞察生成 ==========

def insight_generation_node(state: State):
    print()
    print('=' * 60)
    print(f'【6 - 洞察生成】（为下一轮改进提供建议）')
    print('=' * 60)

    prompt = (f"你是项目管理专家。根据当前风险评估生成改进建议。\n\n"
              f"排期：{json.dumps(state['schedule'], ensure_ascii=False)}\n"
              f"分配：{json.dumps(state['allocations'], ensure_ascii=False)}\n"
              f"风险：{json.dumps(state['risks'], ensure_ascii=False)}\n"
              f"风险分数历史：{state['risk_scores']}\n\n"
              f"请给出具体的改进建议，用于下一轮排期和人员分配的优化。")

    insights = call_llm(prompt)

    print(f'>>> 洞察:')
    print(insights[:300])
    if len(insights) > 300:
        print('...')

    return {"insights": insights}


# ========== 路由函数 ==========

def router(state: State) -> str:
    """自反思路由：风险降低则结束，否则继续优化"""
    iteration = state["iteration"]
    max_iter = state["max_iteration"]
    scores = state["risk_scores"]

    print()
    print('=' * 60)
    print('【路由决策】')
    print('=' * 60)
    print(f'>>> 当前迭代: {iteration}/{max_iter}')
    print(f'>>> 风险分数历史: {scores}')

    if iteration >= max_iter:
        print(f'>>> → 达到最大迭代次数，结束')
        return END
    elif len(scores) > 1 and scores[-1] < scores[0]:
        print(f'>>> → 风险已降低（{scores[0]} → {scores[-1]}），结束')
        return END
    else:
        print(f'>>> → 风险未降低，生成洞察并重新优化')
        return "insight_generator"


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("task_generation", task_generation_node)
    workflow.add_node("task_dependencies", task_dependency_node)
    workflow.add_node("task_scheduler", task_scheduler_node)
    workflow.add_node("task_allocator", task_allocation_node)
    workflow.add_node("risk_assessor", risk_assessment_node)
    workflow.add_node("insight_generator", insight_generation_node)

    # 线性流水线
    workflow.set_entry_point("task_generation")
    workflow.add_edge("task_generation", "task_dependencies")
    workflow.add_edge("task_dependencies", "task_scheduler")
    workflow.add_edge("task_scheduler", "task_allocator")
    workflow.add_edge("task_allocator", "risk_assessor")

    # 自反思循环：风险评估 → 条件路由
    workflow.add_conditional_edges("risk_assessor", router, ["insight_generator", END])

    # 洞察 → 回到排期（重新优化）
    workflow.add_edge("insight_generator", "task_scheduler")

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


# ========== 示例数据 ==========

EXAMPLE_PROJECT = "我们公司要开发一个智能客服聊天机器人，为客户提供7×24小时的产品咨询和售后支持服务。"

EXAMPLE_TEAM = [
    {"name": "张伟", "profile": "前端开发工程师，擅长React、Vue、HTML/CSS、JavaScript"},
    {"name": "李娜", "profile": "后端开发工程师，精通Python、Django、SQL、RESTful API"},
    {"name": "王强", "profile": "项目经理，有敏捷开发经验，擅长团队管理和风险控制"},
    {"name": "刘洋", "profile": "全栈工程师，前端React+后端Node.js+MongoDB都能做"},
    {"name": "陈静", "profile": "DevOps工程师，擅长CI/CD、Docker、Kubernetes、云服务"},
    {"name": "赵磊", "profile": "初级前端开发，会HTML/CSS/JavaScript和基础React"},
    {"name": "孙芳", "profile": "资深数据科学家，擅长机器学习、NLP、Python、大数据技术"},
]


# ========== 运行 ==========

if __name__ == '__main__':
    print('第13课 - 项目管理助手')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 获取项目描述
    print('请输入项目描述（回车使用默认示例）:')
    user_desc = input('项目描述: ').strip()
    if not user_desc:
        user_desc = EXAMPLE_PROJECT
        print(f'>>> 使用默认: {user_desc}')

    print()
    print('#' * 60)
    print('#  开始项目规划')
    print('#' * 60)

    result = app.invoke({
        "project_description": user_desc,
        "team": EXAMPLE_TEAM,
        "tasks": [],
        "dependencies": [],
        "schedule": [],
        "allocations": [],
        "risks": [],
        "insights": "",
        "iteration": 0,
        "max_iteration": 3,
        "risk_scores": [],
        "schedule_history": [],
        "allocation_history": [],
    })

    print()
    print('#' * 60)
    print('#  项目规划完成')
    print('#' * 60)
    print(f'  总迭代次数: {result["iteration"]}')
    print(f'  风险分数变化: {result["risk_scores"]}')
    print()
    print('【最终任务列表】')
    for t in result["tasks"]:
        print(f'  - {t["name"]} ({t.get("days", "?")}天)')
    print()
    print('【最终排期】')
    for s in result["schedule"]:
        print(f'  {s["task"]}: 第{s["start_day"]}天 → 第{s["end_day"]}天')
    print()
    print('【最终人员分配】')
    for a in result["allocations"]:
        print(f'  {a["task"]} → {a["member"]}')
    print()
