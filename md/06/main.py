"""
第6课 - ATLAS 多智能体协调系统（LangGraph 框架版）

核心概念：
  - Coordinator 协调器：分析请求，决定激活哪些 Agent
  - 条件路由：根据协调器结果动态选择分支
  - 并行执行：多个 Agent 同时工作
  - ReAct 模式：思考 → 行动 → 观察 → 响应

简化版：保留核心架构，用同步方式运行，方便学习。
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Union
from operator import add

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ========== 模拟学生数据（中文） ==========

STUDENT_PROFILE = {
    "姓名": "张明",
    "年级": "大三",
    "专业": "计算机科学",
    "学习风格": "视觉型",
    "擅长科目": ["数据结构", "操作系统"],
    "薄弱科目": ["高等数学", "概率论"],
    "学习偏好": {
        "最佳时段": "上午9-12点",
        "单次专注时长": "45分钟",
        "喜欢的方式": "思维导图 + 做题",
    },
}

STUDENT_CALENDAR = {
    "事件": [
        {"名称": "高等数学课", "时间": "周一 8:00-10:00", "类型": "课程"},
        {"名称": "数据结构课", "时间": "周二 10:00-12:00", "类型": "课程"},
        {"名称": "篮球训练", "时间": "周三 16:00-18:00", "类型": "运动"},
        {"名称": "概率论课", "时间": "周四 14:00-16:00", "类型": "课程"},
        {"名称": "编程社团", "时间": "周五 19:00-21:00", "类型": "社团"},
    ],
}

STUDENT_TASKS = {
    "任务": [
        {"名称": "高数作业第5章", "截止": "周三", "优先级": "高", "预计耗时": "3小时"},
        {"名称": "数据结构实验报告", "截止": "周五", "优先级": "中", "预计耗时": "4小时"},
        {"名称": "概率论复习", "截止": "下周一考试", "优先级": "高", "预计耗时": "8小时"},
    ],
}


# ========== 定义 State ==========

def dict_reducer(d1: Dict, d2: Dict) -> Dict:
    """深度合并两个字典（多个 Agent 的结果需要合并到同一个 state）"""
    merged = d1.copy()
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged


class AcademicState(TypedDict):
    """学术辅助系统的全局状态"""
    messages: Annotated[List, add]                     # 消息历史（追加合并）
    profile: Annotated[Dict, dict_reducer]             # 学生档案
    calendar: Annotated[Dict, dict_reducer]            # 日程表
    tasks: Annotated[Dict, dict_reducer]               # 任务列表
    results: Annotated[Dict[str, Any], dict_reducer]   # 各 Agent 的输出结果


# ========== 初始化 LLM ==========

llm = ChatOpenAI(
    model=os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    temperature=0.5,
    base_url=os.getenv('API_BASE_URL'),
    api_key=os.getenv('API_KEY'),
    use_responses_api=False,
    default_headers={"User-Agent": "Mozilla/5.0"},  # 第三方API可能屏蔽SDK默认UA
)


# ========== 节点1：协调器（Coordinator） ==========

def coordinator_node(state: AcademicState) -> Dict:
    """
    协调器：分析用户请求，决定需要激活哪些 Agent。
    这是多智能体系统的"大脑"。
    """
    query = state["messages"][-1].content
    profile = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})

    prompt = f"""你是一个学术辅助系统的协调器。分析用户的请求，判断是否与学习/学术相关，并决定需要哪些专业Agent来处理。

重要规则：
- 如果用户的请求与学习、学术、日程、考试等**无关**（比如闲聊、问天气、讲笑话等），返回 {{"required_agents": [], "reasoning": "与学术无关", "direct_answer": "你的直接回答"}}
- 如果与学术相关，从以下Agent中选择需要的：

可用的Agent：
- PLANNER（计划者）：处理日程安排、时间管理、学习计划
- NOTEWRITER（笔记员）：生成学习笔记、知识总结、复习材料
- ADVISOR（顾问）：提供学习建议、改进方案、心理疏导

学生信息：
{json.dumps(profile, ensure_ascii=False, indent=2)}

当前日程：
{json.dumps(calendar, ensure_ascii=False, indent=2)}

待办任务：
{json.dumps(tasks, ensure_ascii=False, indent=2)}

用户请求：{query}

请用JSON格式回复（不要包含其他内容）：
- 学术相关：{{"required_agents": ["PLANNER"], "reasoning": "原因"}}
- 非学术：{{"required_agents": [], "reasoning": "与学术无关", "direct_answer": "直接回答内容"}}"""

    print()
    print('=' * 50)
    print('【协调器节点】分析用户请求')
    print('=' * 50)
    print(f'>>> 用户请求: {query}')

    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    print(f'>>> 协调器决策: {response_text}')

    # 解析 JSON 响应（LLM 返回可能包含额外文本，需要容错）
    import re
    analysis = None
    try:
        analysis = json.loads(response_text)
    except json.JSONDecodeError:
        # 尝试提取第一个完整 JSON 对象
        # 用计数器匹配大括号，找到第一个平衡的 {}
        start = response_text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    depth += 1
                elif response_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            analysis = json.loads(response_text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
    if analysis is None:
        # 最终兜底：从文本中识别 Agent 名称
        agents = []
        for name in ["PLANNER", "NOTEWRITER", "ADVISOR"]:
            if name.lower() in response_text.lower() or name in response_text:
                agents.append(name)
        analysis = {"required_agents": agents or ["PLANNER"], "reasoning": "从文本中提取"}

    print(f'>>> 需要的Agent: {analysis.get("required_agents", [])}')
    print(f'>>> 原因: {analysis.get("reasoning", "")}')

    return {"results": {"coordinator_analysis": analysis}}


# ========== 节点2：档案分析 ==========

def profile_analyzer_node(state: AcademicState) -> Dict:
    """分析学生档案，提取学习偏好"""
    profile = state.get("profile", {})

    prompt = f"""分析以下学生档案，提取关键学习特征。用一段简洁的中文总结。

学生档案：
{json.dumps(profile, ensure_ascii=False, indent=2)}

请总结：学习风格、最佳学习时段、擅长和薄弱领域。"""

    print()
    print('=' * 50)
    print('【档案分析节点】')
    print('=' * 50)

    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content.strip()

    print(f'>>> 档案分析结果: {analysis[:200]}...')

    return {"results": {"profile_analysis": analysis}}


# ========== 节点3a：计划者 Agent ==========

def planner_node(state: AcademicState) -> Dict:
    """计划者Agent：分析日程和任务，生成学习计划"""
    query = state["messages"][-1].content
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})
    profile_analysis = state.get("results", {}).get("profile_analysis", "")

    prompt = f"""你是学习计划Agent。根据学生的日程和任务，生成具体的时间安排。

学生特征：{profile_analysis}

当前日程：
{json.dumps(calendar, ensure_ascii=False, indent=2)}

待办任务：
{json.dumps(tasks, ensure_ascii=False, indent=2)}

用户请求：{query}

请生成一个具体的、按天排列的学习计划。考虑：
1. 避开已有日程
2. 利用学生的最佳学习时段
3. 高优先级任务优先安排
4. 每次学习不超过学生的专注时长"""

    print()
    print('=' * 50)
    print('【计划者Agent】生成学习计划')
    print('=' * 50)

    response = llm.invoke([HumanMessage(content=prompt)])
    plan = response.content.strip()

    print(f'>>> 计划者输出: {plan[:300]}...')

    return {"results": {"planner_output": plan}}


# ========== 节点3b：笔记员 Agent ==========

def notewriter_node(state: AcademicState) -> Dict:
    """笔记员Agent：根据学习风格生成学习材料"""
    query = state["messages"][-1].content
    profile_analysis = state.get("results", {}).get("profile_analysis", "")

    prompt = f"""你是学习笔记Agent。根据学生的学习风格和请求，生成结构化的学习材料。

学生特征：{profile_analysis}

用户请求：{query}

请生成适合该学生学习风格的笔记/复习材料。如果是视觉型学习者，多用列表、分类、关键词高亮。"""

    print()
    print('=' * 50)
    print('【笔记员Agent】生成学习材料')
    print('=' * 50)

    response = llm.invoke([HumanMessage(content=prompt)])
    notes = response.content.strip()

    print(f'>>> 笔记员输出: {notes[:300]}...')

    return {"results": {"notewriter_output": notes}}


# ========== 节点3c：顾问 Agent ==========

def advisor_node(state: AcademicState) -> Dict:
    """顾问Agent：提供个性化学习建议"""
    query = state["messages"][-1].content
    profile = state.get("profile", {})
    profile_analysis = state.get("results", {}).get("profile_analysis", "")

    prompt = f"""你是学习顾问Agent。根据学生的情况，提供个性化的学习建议和改进方案。

学生特征：{profile_analysis}
学生详细信息：{json.dumps(profile, ensure_ascii=False)}

用户请求：{query}

请提供：
1. 针对当前困难的具体建议
2. 长期改进方向
3. 学习方法推荐"""

    print()
    print('=' * 50)
    print('【顾问Agent】提供学习建议')
    print('=' * 50)

    response = llm.invoke([HumanMessage(content=prompt)])
    advice = response.content.strip()

    print(f'>>> 顾问输出: {advice[:300]}...')

    return {"results": {"advisor_output": advice}}


# ========== 节点4：执行汇总 ==========

def executor_node(state: AcademicState) -> Dict:
    """汇总所有Agent的输出，生成最终回复"""
    results = state.get("results", {})

    # 非学术问题：直接返回协调器的回答
    if "direct_answer" in results:
        final = results["direct_answer"]
        print()
        print('=' * 50)
        print('【执行汇总节点】直接回答（非学术问题）')
        print('=' * 50)
        return {"results": {"final_output": final}}

    outputs = []
    if "planner_output" in results:
        outputs.append(f"【学习计划】\n{results['planner_output']}")
    if "notewriter_output" in results:
        outputs.append(f"【学习材料】\n{results['notewriter_output']}")
    if "advisor_output" in results:
        outputs.append(f"【学习建议】\n{results['advisor_output']}")

    final = "\n\n" + "\n\n---\n\n".join(outputs) if outputs else "暂无结果"

    print()
    print('=' * 50)
    print('【执行汇总节点】')
    print('=' * 50)
    print(f'>>> 汇总了 {len(outputs)} 个Agent的输出')

    return {"results": {"final_output": final}}


# ========== 构建图工作流 ==========

def build_workflow() -> StateGraph:
    """构建多智能体协调工作流"""
    workflow = StateGraph(AcademicState)

    # 添加节点
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("profile_analyzer", profile_analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("notewriter", notewriter_node)
    workflow.add_node("advisor", advisor_node)
    workflow.add_node("executor", executor_node)

    # 协调器后的路由：学术相关 → 继续分析，非学术 → 直接汇总
    def route_after_coordinator(state: AcademicState) -> str:
        """协调器判断后：学术问题走 profile_analyzer，非学术直接走 executor"""
        analysis = state.get("results", {}).get("coordinator_analysis", {})
        required = analysis.get("required_agents", [])
        if not required:
            # 非学术问题，把 direct_answer 存入结果，直接跳到 executor
            direct = analysis.get("direct_answer", "")
            if direct:
                state["results"]["direct_answer"] = direct
            print(f'>>> 非学术问题，直接回答')
            return "executor"
        print(f'>>> 学术问题，进入档案分析')
        return "profile_analyzer"

    # 条件路由函数
    def route_to_agents(state: AcademicState) -> List[str]:
        """根据协调器的分析结果，决定激活哪些Agent"""
        analysis = state.get("results", {}).get("coordinator_analysis", {})
        required = analysis.get("required_agents", ["PLANNER"])

        next_nodes = []
        if "PLANNER" in required:
            next_nodes.append("planner")
        if "NOTEWRITER" in required:
            next_nodes.append("notewriter")
        if "ADVISOR" in required:
            next_nodes.append("advisor")

        if not next_nodes:
            next_nodes = ["planner"]

        print(f'>>> 条件路由 → 激活: {next_nodes}')
        return next_nodes

    # 连接边
    workflow.add_edge(START, "coordinator")

    # 协调器后条件路由：学术 → profile_analyzer，非学术 → executor
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {"profile_analyzer": "profile_analyzer", "executor": "executor"},
    )

    # 条件路由：档案分析后，根据协调器决策路由到不同Agent
    workflow.add_conditional_edges(
        "profile_analyzer",
        route_to_agents,
        ["planner", "notewriter", "advisor"],
    )

    # 所有Agent完成后汇总
    workflow.add_edge("planner", "executor")
    workflow.add_edge("notewriter", "executor")
    workflow.add_edge("advisor", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()


# ========== 运行 ==========

def run_query(app, query: str):
    """处理一次用户请求"""
    print()
    print('#' * 60)
    print(f'#  用户请求: {query}')
    print('#' * 60)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "profile": STUDENT_PROFILE,
        "calendar": STUDENT_CALENDAR,
        "tasks": STUDENT_TASKS,
        "results": {},
    }

    result = app.invoke(initial_state)

    final_output = result.get("results", {}).get("final_output", "无结果")

    print()
    print('=' * 60)
    print('【最终输出】')
    print('=' * 60)
    print(final_output)
    print()

    return final_output


def print_graph(app):
    """打印工作流的图结构（ASCII 文本版）"""
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        # 方式1：导出 Mermaid 格式的图（文本）
        mermaid = app.get_graph().draw_mermaid()
        print(mermaid)
        print()

        # 方式2：保存为 PNG 图片
        try:
            png_data = app.get_graph().draw_mermaid_png()
            png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_graph.png')
            with open(png_path, 'wb') as f:
                f.write(png_data)
            print(f'>>> 图片已保存到: {png_path}')
        except Exception as e:
            print(f'>>> PNG 导出失败（需要网络访问 Mermaid API）: {e}')
            print('>>> 可以把上面的 Mermaid 代码粘贴到 https://mermaid.live 在线查看')
    except Exception as e:
        print(f'>>> 图可视化失败: {e}')
    print()


if __name__ == '__main__':
    print('第6课 - ATLAS 多智能体协调系统')
    print(f'模型: {os.getenv("MODEL_NAME")}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()

    # 构建工作流
    app = build_workflow()

    # 打印图结构
    print_graph(app)

    # 示例请求
    examples = [
        "帮我规划下周的学习计划，我下周一有概率论考试",
        "我高数学不好，帮我制定复习计划并给一些学习建议",
    ]

    print('--- 示例请求 ---')
    for q in examples:
        run_query(app, q)

    # 交互模式
    print('\n输入请求，输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        run_query(app, user_input)
