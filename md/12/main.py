"""
第12课 - AI职业助手（LangGraph 框架版）

核心概念：
  - 两级条件路由：第一级4选1，第二级2选1
  - 多轮对话：处理节点内维护对话历史循环
  - Few-Shot分类：用示例教LLM分类用户意图
  - 6个功能模块：教程生成、问答对话、简历制作、面试题库、模拟面试、求职建议
"""

import os
from typing import TypedDict

import httpx
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_HISTORY = 10  # 对话历史最大轮数


def call_llm(prompt: str, system: str = "", messages: list = None) -> str:
    """调用 LLM（支持单轮prompt或多轮messages）"""
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def trim_messages(messages: list) -> list:
    """裁剪对话历史，保留system消息和最近的N轮对话"""
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY * 2:
        non_system = non_system[-(MAX_HISTORY * 2):]
    return system_msgs + non_system


# ========== State ==========

class State(TypedDict):
    query: str       # 用户原始输入
    category: str    # 分类结果
    response: str    # 最终响应


# ========== 节点：一级分类 ==========

def categorize_node(state: State):
    print()
    print('=' * 60)
    print('【一级分类】')
    print('=' * 60)

    system = ("将用户请求分类到以下类别之一，只回复数字：\n"
              "1: 学习AI技术\n"
              "2: 简历制作\n"
              "3: 面试准备\n"
              "4: 求职搜索\n\n"
              "示例：\n"
              "- \"什么是RAG？怎么实现？\" → 1\n"
              "- \"帮我写一份AI工程师简历\" → 2\n"
              "- \"面试一般会问什么？\" → 3\n"
              "- \"有什么AI相关的岗位？\" → 4\n\n"
              "只回复数字。")

    category = call_llm(state["query"], system)
    print(f'>>> 用户输入: {state["query"]}')
    print(f'>>> 分类结果: {category}')
    return {"category": category}


# ========== 节点：二级分类 - 学习 ==========

def handle_learning_node(state: State):
    print()
    print('=' * 60)
    print('【二级分类 - 学习】')
    print('=' * 60)

    system = ("将用户请求分类为以下之一，只回复类别名：\n"
              "- 教程：用户想要教程、文档、博客等学习材料\n"
              "- 问答：用户有具体问题想问\n"
              "默认为「问答」\n\n"
              "示例：\n"
              "- \"帮我写一个LangChain教程\" → 教程\n"
              "- \"RAG和微调有什么区别？\" → 问答\n\n"
              "只回复：教程 或 问答")

    sub_category = call_llm(state["query"], system)
    print(f'>>> 子分类: {sub_category}')
    return {"category": sub_category}


# ========== 节点：二级分类 - 面试 ==========

def handle_interview_node(state: State):
    print()
    print('=' * 60)
    print('【二级分类 - 面试】')
    print('=' * 60)

    system = ("将用户请求分类为以下之一，只回复类别名：\n"
              "- 题库：用户想了解面试题目、准备材料\n"
              "- 模拟：用户想进行模拟面试练习\n"
              "默认为「题库」\n\n"
              "示例：\n"
              "- \"AI面试一般问什么？\" → 题库\n"
              "- \"帮我模拟一场面试\" → 模拟\n\n"
              "只回复：题库 或 模拟")

    sub_category = call_llm(state["query"], system)
    print(f'>>> 子分类: {sub_category}')
    return {"category": sub_category}


# ========== 叶子节点：教程生成 ==========

def tutorial_node(state: State):
    print()
    print('=' * 60)
    print('【教程生成】')
    print('=' * 60)

    system = ("你是资深AI开发者和技术博主。根据用户需求生成高质量的中文教程，"
              "包含清晰的解释、Python代码示例和参考资源链接。")

    print(f'>>> 正在生成教程...')
    response = call_llm(state["query"], system)
    print()
    print(response)
    return {"response": response}


# ========== 叶子节点：问答对话（多轮） ==========

def qa_bot_node(state: State):
    print()
    print('=' * 60)
    print('【问答对话】（输入 exit 退出）')
    print('=' * 60)

    messages = [
        {"role": "system", "content": ("你是资深AI工程师，擅长解答各种AI技术问题。"
                                        "用中文回答，给出有深度的专业解答。")},
        {"role": "user", "content": state["query"]},
    ]

    all_responses = []
    while True:
        messages = trim_messages(messages)
        response = call_llm("", messages=messages)
        messages.append({"role": "assistant", "content": response})
        all_responses.append(response)

        print(f'\nAI: {response}\n')

        user_input = input('你: ').strip()
        if user_input.lower() == 'exit':
            print('>>> 问答结束')
            break
        messages.append({"role": "user", "content": user_input})

    return {"response": all_responses[-1] if all_responses else ""}


# ========== 叶子节点：简历制作（多轮） ==========

def resume_node(state: State):
    print()
    print('=' * 60)
    print('【简历制作】（输入 exit 退出）')
    print('=' * 60)

    messages = [
        {"role": "system", "content": ("你是专业的简历顾问，擅长为AI/技术岗位制作简历。"
                                        "分步骤向用户收集信息（技能、经验、项目），"
                                        "4-5轮对话后生成完整的简历。用中文交流。")},
        {"role": "user", "content": state["query"]},
    ]

    last_response = ""
    while True:
        messages = trim_messages(messages)
        response = call_llm("", messages=messages)
        messages.append({"role": "assistant", "content": response})
        last_response = response

        print(f'\nAI: {response}\n')

        user_input = input('你: ').strip()
        if user_input.lower() == 'exit':
            print('>>> 简历制作结束')
            break
        messages.append({"role": "user", "content": user_input})

    return {"response": last_response}


# ========== 叶子节点：面试题库（多轮） ==========

def interview_questions_node(state: State):
    print()
    print('=' * 60)
    print('【面试题库】（输入 exit 退出）')
    print('=' * 60)

    messages = [
        {"role": "system", "content": ("你是AI面试专家。根据用户需求提供面试题目列表，"
                                        "包含题目、参考答案要点。可以追问用户想深入哪个方向。用中文回答。")},
        {"role": "user", "content": state["query"]},
    ]

    last_response = ""
    while True:
        messages = trim_messages(messages)
        response = call_llm("", messages=messages)
        messages.append({"role": "assistant", "content": response})
        last_response = response

        print(f'\nAI: {response}\n')

        user_input = input('你: ').strip()
        if user_input.lower() == 'exit':
            print('>>> 面试题库结束')
            break
        messages.append({"role": "user", "content": user_input})

    return {"response": last_response}


# ========== 叶子节点：模拟面试（多轮） ==========

def mock_interview_node(state: State):
    print()
    print('=' * 60)
    print('【模拟面试】（输入 exit 退出）')
    print('=' * 60)

    messages = [
        {"role": "system", "content": ("你是AI面试官，正在对候选人进行AI岗位的模拟面试。"
                                        "每次只问一个问题，等候选人回答后再追问或出下一题。"
                                        "面试结束时给出整体评价和改进建议。用中文交流。")},
        {"role": "user", "content": "我准备好了，请开始面试。"},
    ]

    last_response = ""
    while True:
        messages = trim_messages(messages)
        response = call_llm("", messages=messages)
        messages.append({"role": "assistant", "content": response})
        last_response = response

        print(f'\n面试官: {response}\n')

        user_input = input('候选人: ').strip()
        if user_input.lower() == 'exit':
            # 请求面试评价
            messages.append({"role": "user", "content": "面试结束，请给出整体评价。"})
            evaluation = call_llm("", messages=messages)
            print(f'\n面试官（评价）: {evaluation}\n')
            last_response = evaluation
            break
        messages.append({"role": "user", "content": user_input})

    return {"response": last_response}


# ========== 叶子节点：求职建议 ==========

def job_search_node(state: State):
    print()
    print('=' * 60)
    print('【求职建议】')
    print('=' * 60)

    system = ("你是AI行业求职顾问。根据用户需求提供求职建议，"
              "包括热门岗位、技能要求、薪资范围、求职策略等。用中文回答。")

    print(f'>>> 正在生成求职建议...')
    response = call_llm(state["query"], system)
    print()
    print(response)
    return {"response": response}


# ========== 路由函数 ==========

def route_query(state: State):
    """一级路由：4选1"""
    cat = state["category"]
    if '1' in cat:
        print(f'>>> 路由到: 学习资源')
        return "handle_learning"
    elif '2' in cat:
        print(f'>>> 路由到: 简历制作')
        return "handle_resume"
    elif '3' in cat:
        print(f'>>> 路由到: 面试准备')
        return "handle_interview"
    elif '4' in cat:
        print(f'>>> 路由到: 求职搜索')
        return "handle_job_search"
    else:
        print(f'>>> 无法识别，默认路由到: 问答')
        return "handle_learning"


def route_learning(state: State):
    """二级路由：学习 → 教程/问答"""
    cat = state["category"].lower()
    if '教程' in cat or 'tutorial' in cat:
        print(f'>>> 路由到: 教程生成')
        return "tutorial"
    else:
        print(f'>>> 路由到: 问答对话')
        return "qa_bot"


def route_interview(state: State):
    """二级路由：面试 → 题库/模拟"""
    cat = state["category"].lower()
    if '模拟' in cat or 'mock' in cat:
        print(f'>>> 路由到: 模拟面试')
        return "mock_interview"
    else:
        print(f'>>> 路由到: 面试题库')
        return "interview_questions"


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    # 添加所有节点
    workflow.add_node("categorize", categorize_node)
    workflow.add_node("handle_learning", handle_learning_node)
    workflow.add_node("handle_resume", resume_node)
    workflow.add_node("handle_interview", handle_interview_node)
    workflow.add_node("handle_job_search", job_search_node)
    workflow.add_node("tutorial", tutorial_node)
    workflow.add_node("qa_bot", qa_bot_node)
    workflow.add_node("interview_questions", interview_questions_node)
    workflow.add_node("mock_interview", mock_interview_node)

    # 入口
    workflow.set_entry_point("categorize")

    # 一级路由：分类 → 4选1
    workflow.add_conditional_edges(
        "categorize",
        route_query,
        {
            "handle_learning": "handle_learning",
            "handle_resume": "handle_resume",
            "handle_interview": "handle_interview",
            "handle_job_search": "handle_job_search",
        }
    )

    # 二级路由：学习 → 教程/问答
    workflow.add_conditional_edges(
        "handle_learning",
        route_learning,
        {
            "tutorial": "tutorial",
            "qa_bot": "qa_bot",
        }
    )

    # 二级路由：面试 → 题库/模拟
    workflow.add_conditional_edges(
        "handle_interview",
        route_interview,
        {
            "interview_questions": "interview_questions",
            "mock_interview": "mock_interview",
        }
    )

    # 所有叶子节点 → END
    workflow.add_edge("handle_resume", END)
    workflow.add_edge("handle_job_search", END)
    workflow.add_edge("tutorial", END)
    workflow.add_edge("qa_bot", END)
    workflow.add_edge("interview_questions", END)
    workflow.add_edge("mock_interview", END)

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
    print('第12课 - AI职业助手')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('功能模块：')
    print('  1. 学习 → 教程生成 / 问答对话')
    print('  2. 简历制作')
    print('  3. 面试 → 面试题库 / 模拟面试')
    print('  4. 求职建议')
    print()

    app = build_workflow()
    print_graph(app)

    while True:
        print()
        print('#' * 60)
        query = input('请输入你的需求（输入 /quit 退出）: ').strip()
        if query == '/quit':
            print('再见，祝职业发展顺利！')
            break

        result = app.invoke({
            "query": query,
            "category": "",
            "response": "",
        })

        print()
        print('=' * 60)
        print('【处理完成】')
        print('=' * 60)
