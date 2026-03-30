"""
第10课 - 作文评分系统（LangGraph 框架版）

核心概念：
  - 条件提前终止：每一步设门槛，不达标就跳过后续评分
  - 4维评分：相关性(30%) + 语法(20%) + 结构(20%) + 深度(30%)
  - 加权平均：未评分的维度=0，自然拉低总分
"""

import os
import re
from typing import TypedDict

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
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extract_score(content: str) -> float:
    """从 LLM 回复中提取分数（期望格式：分数: 0.85）"""
    match = re.search(r'分数[:：]\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    # 兜底：尝试英文格式
    match = re.search(r'Score[:：]\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"无法从回复中提取分数: {content[:100]}")


# ========== State ==========

class State(TypedDict):
    essay: str               # 作文内容
    relevance_score: float   # 相关性得分 (0~1)
    grammar_score: float     # 语法得分 (0~1)
    structure_score: float   # 结构得分 (0~1)
    depth_score: float       # 深度得分 (0~1)
    final_score: float       # 最终加权得分


# ========== 节点：相关性检查 ==========

def check_relevance_node(state: State):
    print()
    print('=' * 50)
    print('【1 - 内容相关性检查】')
    print('=' * 50)

    system = ("你是作文评分专家。评估以下作文的内容相关性：是否切题、是否围绕主题展开。\n"
              "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。")
    result = call_llm(state["essay"], system)

    try:
        score = extract_score(result)
    except ValueError as e:
        print(f'>>> 提取分数失败: {e}')
        score = 0.0

    print(f'>>> 相关性得分: {score}')
    print(f'>>> 门槛: > 0.5 继续, ≤ 0.5 提前终止')
    print(f'>>> LLM评语: {result[:150]}...')
    return {"relevance_score": score}


# ========== 节点：语法检查 ==========

def check_grammar_node(state: State):
    print()
    print('=' * 50)
    print('【2 - 语法检查】')
    print('=' * 50)

    system = ("你是语言专家。评估以下作文的语法和语言表达质量。\n"
              "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。")
    result = call_llm(state["essay"], system)

    try:
        score = extract_score(result)
    except ValueError as e:
        print(f'>>> 提取分数失败: {e}')
        score = 0.0

    print(f'>>> 语法得分: {score}')
    print(f'>>> 门槛: > 0.6 继续, ≤ 0.6 提前终止')
    print(f'>>> LLM评语: {result[:150]}...')
    return {"grammar_score": score}


# ========== 节点：结构分析 ==========

def analyze_structure_node(state: State):
    print()
    print('=' * 50)
    print('【3 - 结构分析】')
    print('=' * 50)

    system = ("你是写作结构专家。评估以下作文的组织架构、段落逻辑和行文流畅度。\n"
              "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。")
    result = call_llm(state["essay"], system)

    try:
        score = extract_score(result)
    except ValueError as e:
        print(f'>>> 提取分数失败: {e}')
        score = 0.0

    print(f'>>> 结构得分: {score}')
    print(f'>>> 门槛: > 0.7 继续, ≤ 0.7 提前终止')
    print(f'>>> LLM评语: {result[:150]}...')
    return {"structure_score": score}


# ========== 节点：深度分析 ==========

def evaluate_depth_node(state: State):
    print()
    print('=' * 50)
    print('【4 - 深度分析】')
    print('=' * 50)

    system = ("你是学术评审专家。评估以下作文的分析深度、批判性思维和独到见解。\n"
              "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。")
    result = call_llm(state["essay"], system)

    try:
        score = extract_score(result)
    except ValueError as e:
        print(f'>>> 提取分数失败: {e}')
        score = 0.0

    print(f'>>> 深度得分: {score}')
    print(f'>>> LLM评语: {result[:150]}...')
    return {"depth_score": score}


# ========== 节点：计算最终分数 ==========

def calculate_final_score_node(state: State):
    print()
    print('=' * 50)
    print('【5 - 计算最终分数】')
    print('=' * 50)

    final = (
        state["relevance_score"] * 0.3 +
        state["grammar_score"]   * 0.2 +
        state["structure_score"] * 0.2 +
        state["depth_score"]     * 0.3
    )

    print(f'>>> 相关性 {state["relevance_score"]:.2f} × 0.3 = {state["relevance_score"] * 0.3:.2f}')
    print(f'>>> 语法   {state["grammar_score"]:.2f} × 0.2 = {state["grammar_score"] * 0.2:.2f}')
    print(f'>>> 结构   {state["structure_score"]:.2f} × 0.2 = {state["structure_score"] * 0.2:.2f}')
    print(f'>>> 深度   {state["depth_score"]:.2f} × 0.3 = {state["depth_score"] * 0.3:.2f}')
    print(f'>>> ──────────────────────')
    print(f'>>> 最终得分: {final:.2f}')

    return {"final_score": final}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("check_grammar", check_grammar_node)
    workflow.add_node("analyze_structure", analyze_structure_node)
    workflow.add_node("evaluate_depth", evaluate_depth_node)
    workflow.add_node("calculate_final_score", calculate_final_score_node)

    workflow.set_entry_point("check_relevance")

    # 条件边：每步设门槛，不达标就提前终止
    workflow.add_conditional_edges(
        "check_relevance",
        lambda x: "check_grammar" if x["relevance_score"] > 0.5 else "calculate_final_score"
    )
    workflow.add_conditional_edges(
        "check_grammar",
        lambda x: "analyze_structure" if x["grammar_score"] > 0.6 else "calculate_final_score"
    )
    workflow.add_conditional_edges(
        "analyze_structure",
        lambda x: "evaluate_depth" if x["structure_score"] > 0.7 else "calculate_final_score"
    )
    workflow.add_conditional_edges(
        "evaluate_depth",
        lambda x: "calculate_final_score"
    )

    workflow.add_edge("calculate_final_score", END)

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

def grade_essay(app, essay: str):
    print()
    print('#' * 60)
    print('#  开始评分')
    print('#' * 60)
    print(f'>>> 作文前100字: {essay.strip()[:100]}...')

    result = app.invoke({
        "essay": essay,
        "relevance_score": 0.0,
        "grammar_score": 0.0,
        "structure_score": 0.0,
        "depth_score": 0.0,
        "final_score": 0.0,
    })

    print()
    print('=' * 60)
    print('【评分结果】')
    print('=' * 60)
    print(f'  相关性: {result["relevance_score"]:.2f}')
    print(f'  语法:   {result["grammar_score"]:.2f}')
    print(f'  结构:   {result["structure_score"]:.2f}')
    print(f'  深度:   {result["depth_score"]:.2f}')
    print(f'  ──────────────────')
    print(f'  最终得分: {result["final_score"]:.2f}')

    # 判断是否提前终止
    if result["grammar_score"] == 0.0 and result["relevance_score"] <= 0.5:
        print(f'  (提前终止于: 相关性检查)')
    elif result["structure_score"] == 0.0 and result["grammar_score"] <= 0.6:
        print(f'  (提前终止于: 语法检查)')
    elif result["depth_score"] == 0.0 and result["structure_score"] <= 0.7:
        print(f'  (提前终止于: 结构分析)')
    else:
        print(f'  (完成全部评分)')
    print()

    return result


if __name__ == '__main__':
    print('第10课 - 作文评分系统')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 示例作文
    examples = [
        {
            "name": "优秀作文",
            "text": """
            人工智能对现代社会的影响

            人工智能已经成为我们日常生活的重要组成部分，在医疗、金融和交通等多个领域
            引发了深刻变革。本文将探讨人工智能对现代社会的深远影响，讨论其带来的机遇
            与挑战。

            在医疗领域，人工智能的应用最为突出。AI驱动的诊断工具能够以极高的准确率
            分析医学影像，往往超越人类的能力。这使得疾病能够被更早发现，治疗方案也
            更加有效。此外，AI算法能处理海量的医学数据，发现人类难以察觉的模式和
            洞察，有望推动药物研发和个性化医疗的突破。

            在金融领域，AI改变了交易处理和监控方式。机器学习算法能实时检测欺诈行为，
            为消费者和机构提供更强的安全保障。智能投顾利用AI提供个性化的投资建议，
            让更多人享受到专业的理财服务。

            交通运输也是AI大显身手的领域。自动驾驶汽车有望减少人为因素导致的交通
            事故，为无法驾驶的人群提供出行方案。在物流方面，AI优化路线规划和库存
            管理，提高了供应链效率，降低了环境影响。

            然而，AI的快速发展也带来了挑战。人们担心AI取代人类岗位，引发了对劳动力
            再培训和技能转型的讨论。隐私和伦理问题同样不容忽视——训练AI需要大量
            数据，这引发了关于数据隐私和知情同意的讨论。

            总之，人工智能在带来巨大机遇的同时，也需要我们审慎对待其社会影响。在将
            AI融入生活各个方面的过程中，必须在技术进步与伦理考量之间取得平衡。
            """
        },
        {
            "name": "跑题作文（预期提前终止）",
            "text": """
            我最喜欢的食物

            我最喜欢吃火锅。火锅好吃又热闹，冬天吃特别暖和。我喜欢毛肚和鸭肠，
            涮七上八下刚刚好。配上麻酱和蒜泥，简直是人间美味。

            除了火锅我还喜欢烧烤。夏天的晚上，和朋友一起撸串喝啤酒，特别开心。
            烤羊肉串要放孜然和辣椒面，外焦里嫩。

            总之，美食让生活更美好。
            """
        },
    ]

    for ex in examples:
        print()
        print(f'>>> 测试: {ex["name"]}')
        grade_essay(app, ex["text"])

    # 交互模式
    print('\n输入作文内容（输入 END 结束作文，输入 /quit 退出）\n')
    while True:
        print('请输入作文（多行输入，单独一行输入 END 结束）:')
        lines = []
        while True:
            line = input()
            if line.strip() == '/quit':
                print('再见！')
                exit()
            if line.strip() == 'END':
                break
            lines.append(line)
        essay = '\n'.join(lines)
        if essay.strip():
            grade_essay(app, essay)
