"""
第14课 - 合同分析助手（LangGraph 框架版）

核心概念：
  - Map-Reduce并行：Send()动态扇出，每个条款/角色并行处理
  - operator.add：并行节点的结果自动合并到列表
  - 两次扇出：条款检查 + 多角色审查
  - 结构化输出：JSON解析合同分类、修改建议
"""

import os
import json
import re
import time
import operator
from typing import TypedDict, List, Optional, Annotated

import httpx
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3  # 并行请求可能触发API限流，需要重试


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（带重试，防止并行请求触发API 500错误）"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3  # 递增等待：3s, 6s
                print(f'    ⚠ API错误，{wait}秒后重试({attempt+1}/{MAX_RETRIES}): {e}')
                time.sleep(wait)
            else:
                raise


def extract_json(text: str) -> dict:
    """从LLM回复中提取JSON"""
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        return {}


# ========== 标准条款库（替代 Pinecone 向量检索） ==========

STANDARD_CLAUSES = {
    "劳动合同": [
        "工作内容和工作地点条款：应明确岗位职责、工作地点，变更需协商一致",
        "劳动报酬条款：应明确工资标准、支付方式、支付时间，不得低于最低工资",
        "工作时间和休假条款：应明确工时制度、加班规定、年假天数",
        "社会保险条款：应明确五险一金的缴纳基数和比例",
        "合同期限条款：应明确固定期限、无固定期限或完成一定工作任务的期限",
        "竞业限制条款：应明确限制范围、期限（不超过2年）、补偿标准",
        "保密条款：应明确保密范围、保密期限、违约责任",
        "解除和终止条款：应明确双方解除合同的条件和程序",
    ],
    "技术服务合同": [
        "服务内容条款：应详细描述技术服务的具体内容和交付标准",
        "服务期限条款：应明确服务起止时间、里程碑节点",
        "知识产权归属条款：应明确开发成果的知识产权归属",
        "保密条款：应明确技术秘密的保护范围和期限",
        "验收标准条款：应明确验收流程、标准和争议处理",
        "违约责任条款：应明确违约情形和赔偿计算方式",
    ],
    "通用条款": [
        "争议解决条款：应明确仲裁或诉讼的管辖地和方式",
        "不可抗力条款：应明确不可抗力的定义、通知义务和后果",
        "通知条款：应明确双方的联系地址和通知送达方式",
    ],
}


def get_clauses_for_type(contract_type: str) -> list:
    """根据合同类型获取标准条款"""
    clauses = STANDARD_CLAUSES.get("通用条款", [])[:]
    for key, val in STANDARD_CLAUSES.items():
        if key != "通用条款" and key in contract_type:
            clauses.extend(val)
            break
    if len(clauses) == len(STANDARD_CLAUSES["通用条款"]):
        # 未匹配到具体类型，用劳动合同兜底
        clauses.extend(STANDARD_CLAUSES.get("劳动合同", []))
    return clauses


# ========== State ==========

class State(TypedDict):
    contract_text: str                                           # 合同原文
    objective: str                                               # 审查目标
    contract_type: str                                           # 合同类型
    industry: str                                                # 行业
    clauses: list                                                # 标准条款列表
    review_roles: list                                           # 审查角色列表
    clause_analysis: Annotated[List[str], operator.add]          # 条款审查结果（并行汇聚）
    clause_modifications: Annotated[List[dict], operator.add]    # 条款修改建议（并行汇聚）
    role_analysis: Annotated[List[str], operator.add]            # 角色审查结果（并行汇聚）
    role_modifications: Annotated[List[dict], operator.add]      # 角色修改建议（并行汇聚）
    final_report: str                                            # 最终报告


# ========== 节点：合同分类 ==========

def classify_contract_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 合同分类】')
    print('=' * 60)

    system = ("分析合同文本，判断合同类型和所属行业。\n"
              "返回JSON：{\"contract_type\": \"劳动合同\", \"industry\": \"互联网\"}")
    result = extract_json(call_llm(state["contract_text"][:2000], system))

    contract_type = result.get("contract_type", "未知")
    industry = result.get("industry", "未知")
    print(f'>>> 合同类型: {contract_type}')
    print(f'>>> 行业: {industry}')

    return {"contract_type": contract_type, "industry": industry}


# ========== 节点：条款检索 ==========

def retrieve_clauses_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 条款检索】')
    print('=' * 60)

    clauses = get_clauses_for_type(state["contract_type"])
    print(f'>>> 获取 {len(clauses)} 个标准条款:')
    for i, c in enumerate(clauses, 1):
        print(f'    {i}. {c[:40]}...')

    return {"clauses": clauses}


# ========== 节点：单条款检查（被Send并行调用） ==========

def check_clause_node(state: State):
    """检查单个标准条款在合同中是否清晰体现"""
    clause = state["clauses"]  # Send传入的是单个条款字符串

    system = (f"你是法律条款审查专家。检查以下合同中是否清晰包含此标准条款：\n\n"
              f"标准条款：{clause}\n\n"
              f"要求：\n"
              f"1. 检查合同中是否包含此条款的关键要素\n"
              f"2. 如果缺失或不清晰，给出修改建议\n"
              f"返回JSON：{{\"analysis\": \"分析结果\", \"modifications\": [{{\"original\": \"原文\", \"suggested\": \"建议\", \"reason\": \"原因\"}}]}}\n"
              f"如果条款已完善，modifications 返回空列表。")

    result = extract_json(call_llm(state["contract_text"][:2000], system))
    analysis = result.get("analysis", "分析失败")
    modifications = result.get("modifications", [])

    print(f'    ✓ 条款审查: {clause[:30]}... → {len(modifications)}条建议')

    return {
        "clause_analysis": [f"【{clause[:20]}】{analysis}"],
        "clause_modifications": modifications,
    }


# ========== 节点：生成审查计划 ==========

def create_review_plan_node(state: State):
    print()
    print('=' * 60)
    print('【4 - 生成审查计划（多角色）】')
    print('=' * 60)

    system = (f"你是法律审查规划师。为以下合同创建多角色审查计划。\n"
              f"合同类型：{state['contract_type']}\n"
              f"行业：{state['industry']}\n"
              f"审查目标：{state['objective']}\n\n"
              f"列出3-5个需要审查的专业角色，每个角色负责不同方面。\n"
              f"返回JSON：{{\"roles\": [\"劳动法专家\", \"知识产权顾问\", \"合规审查官\"]}}")

    result = extract_json(call_llm("生成角色审查计划", system))
    roles = result.get("roles", ["法律风险专家", "合规审查官", "条款完整性审查员"])

    print(f'>>> 审查角色:')
    for r in roles:
        print(f'    - {r}')

    return {"review_roles": roles}


# ========== 节点：单角色审查（被Send并行调用） ==========

def role_review_node(state: State):
    """单个法律角色对合同进行审查"""
    role = state["review_roles"]  # Send传入的是单个角色字符串

    system = (f"你是{role}。从你的专业角度审查以下合同。\n\n"
              f"要求：\n"
              f"1. 识别你专业领域内的问题\n"
              f"2. 给出具体修改建议\n"
              f"返回JSON：{{\"analysis\": \"审查分析\", \"modifications\": [{{\"original\": \"原文\", \"suggested\": \"建议\", \"reason\": \"原因\"}}]}}\n"
              f"如果没有问题，modifications 返回空列表。")

    result = extract_json(call_llm(state["contract_text"][:2000], system))
    analysis = result.get("analysis", "分析失败")
    modifications = result.get("modifications", [])

    print(f'    ✓ {role}: {len(modifications)}条修改建议')

    return {
        "role_analysis": [f"【{role}】{analysis}"],
        "role_modifications": modifications,
    }


# ========== 节点：生成最终报告 ==========

def generate_report_node(state: State):
    print()
    print('=' * 60)
    print('【6 - 生成最终报告】')
    print('=' * 60)

    all_mods = state["clause_modifications"] + state["role_modifications"]

    # 让LLM总结修改建议
    if all_mods:
        system = ("你是法律报告撰写专家。总结以下合同修改建议，"
                  "从法律专业角度解释每条建议的重要性。用中文。")
        mod_summary = call_llm(json.dumps(all_mods, ensure_ascii=False)[:3000], system)
    else:
        mod_summary = "未发现需要修改的内容。"

    report = "\n".join([
        "=" * 50,
        "         合同审查报告",
        "=" * 50,
        "",
        f"合同类型: {state['contract_type']}",
        f"行业: {state['industry']}",
        f"审查目标: {state['objective']}",
        "",
        f"条款审查数: {len(state['clause_analysis'])}",
        f"角色审查数: {len(state['role_analysis'])}",
        f"修改建议数: {len(all_mods)}",
        "",
        "【条款审查结果】",
        *[f"  {a}" for a in state["clause_analysis"]],
        "",
        "【角色审查结果】",
        *[f"  {a}" for a in state["role_analysis"]],
        "",
        "【修改建议摘要】",
        mod_summary,
        "",
        "=" * 50,
        "         报告结束",
        "=" * 50,
    ])

    print(report)
    return {"final_report": report}


# ========== Send 扇出函数 ==========

def fan_out_clauses(state: State):
    """Map: 为每个条款创建一个并行检查任务"""
    print()
    print('=' * 60)
    print(f'【3 - 条款检查】（Map: 扇出 {len(state["clauses"])} 个并行任务）')
    print('=' * 60)

    return [
        Send("check_clause", {
            "contract_text": state["contract_text"],
            "clauses": clause,  # 注意：Send传单个条款，不是列表
        })
        for clause in state["clauses"]
    ]


def fan_out_roles(state: State):
    """Map: 为每个审查角色创建一个并行审查任务"""
    print()
    print('=' * 60)
    print(f'【5 - 角色审查】（Map: 扇出 {len(state["review_roles"])} 个并行任务）')
    print('=' * 60)

    return [
        Send("role_review", {
            "contract_text": state["contract_text"],
            "review_roles": role,  # 注意：Send传单个角色，不是列表
        })
        for role in state["review_roles"]
    ]


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("classify_contract", classify_contract_node)
    workflow.add_node("retrieve_clauses", retrieve_clauses_node)
    workflow.add_node("check_clause", check_clause_node)
    workflow.add_node("create_review_plan", create_review_plan_node)
    workflow.add_node("role_review", role_review_node)
    workflow.add_node("generate_report", generate_report_node)

    # 线性部分
    workflow.set_entry_point("classify_contract")
    workflow.add_edge("classify_contract", "retrieve_clauses")

    # 第一次 Map-Reduce：条款检查
    workflow.add_conditional_edges("retrieve_clauses", fan_out_clauses, ["check_clause"])
    workflow.add_edge("check_clause", "create_review_plan")  # Reduce：汇聚到审查计划

    # 第二次 Map-Reduce：角色审查
    workflow.add_conditional_edges("create_review_plan", fan_out_roles, ["role_review"])
    workflow.add_edge("role_review", "generate_report")  # Reduce：汇聚到报告生成

    workflow.add_edge("generate_report", END)

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


# ========== 示例合同 ==========

EXAMPLE_CONTRACT = """
劳动合同

甲方（用人单位）：北京智能科技有限公司
乙方（劳动者）：张三

一、工作内容
乙方同意根据甲方的工作需要，担任高级软件工程师职务，主要负责AI产品的研发工作。

二、合同期限
本合同为固定期限劳动合同，自2024年1月1日起至2026年12月31日止。

三、工作时间
实行标准工时制度，每日工作8小时，每周工作5天。

四、劳动报酬
月工资为人民币35000元，每月15日发放。

五、保密义务
乙方在职期间及离职后须对甲方的商业秘密和技术秘密保密。

六、合同解除
任何一方提前30日书面通知对方，可以解除本合同。

签订日期：2024年1月1日
"""


# ========== 运行 ==========

if __name__ == '__main__':
    print('第14课 - 合同分析助手')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 获取合同文本
    print('请输入合同文本（回车使用默认示例，多行输入以 END 结束）:')
    first_line = input().strip()
    if not first_line:
        contract = EXAMPLE_CONTRACT
        print(f'>>> 使用默认示例合同')
    else:
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        contract = '\n'.join(lines)

    objective = input('\n审查目标（回车默认"全面审查"）: ').strip() or "全面审查合同条款完整性和法律风险"

    print()
    print('#' * 60)
    print('#  开始合同审查')
    print('#' * 60)

    result = app.invoke({
        "contract_text": contract,
        "objective": objective,
        "contract_type": "",
        "industry": "",
        "clauses": [],
        "review_roles": [],
        "clause_analysis": [],
        "clause_modifications": [],
        "role_analysis": [],
        "role_modifications": [],
        "final_report": "",
    })

    print()
    print(f'>>> 审查完成，共 {len(result["clause_modifications"]) + len(result["role_modifications"])} 条修改建议')
