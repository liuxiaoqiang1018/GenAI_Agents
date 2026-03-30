"""
第14课 - 合同分析助手内部机制（不使用 LangGraph）

目的：让你看清 Map-Reduce 并行的本质：
  1. Send() = for 循环，依次（或并发）调用同一个函数
  2. operator.add = list.extend()，把各子任务的结果合并
  3. 两次 Map-Reduce = 两个 for 循环
  4. 整个系统 = 2次分类/检索 + N次条款检查 + M次角色审查 + 1次报告

对比 main.py（LangGraph 框架版），理解：
  - Send("check_clause", {...}) → 就是 for clause in clauses: check(clause)
  - Annotated[List, operator.add] → 就是 results.extend(single_result)
  - 扇出+汇聚 → 就是 for 循环 + 结果列表
"""

import os
import json
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
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


# ========== 标准条款库 ==========

STANDARD_CLAUSES = {
    "劳动合同": [
        "工作内容和工作地点条款",
        "劳动报酬条款",
        "工作时间和休假条款",
        "社会保险条款",
        "合同期限条款",
        "竞业限制条款",
        "保密条款",
        "解除和终止条款",
    ],
    "通用条款": [
        "争议解决条款",
        "不可抗力条款",
        "通知条款",
    ],
}


def get_clauses(contract_type: str) -> list:
    clauses = STANDARD_CLAUSES.get("通用条款", [])[:]
    for key, val in STANDARD_CLAUSES.items():
        if key != "通用条款" and key in contract_type:
            clauses.extend(val)
            break
    if len(clauses) == len(STANDARD_CLAUSES["通用条款"]):
        clauses.extend(STANDARD_CLAUSES.get("劳动合同", []))
    return clauses


# ========== 示例合同 ==========

EXAMPLE_CONTRACT = """
劳动合同

甲方（用人单位）：北京智能科技有限公司
乙方（劳动者）：张三

一、工作内容
乙方担任高级软件工程师职务，负责AI产品研发。

二、合同期限
固定期限，2024年1月1日至2026年12月31日。

三、工作时间
标准工时制，每日8小时，每周5天。

四、劳动报酬
月工资35000元，每月15日发放。

五、保密义务
乙方在职期间及离职后须保密。

六、合同解除
任何一方提前30日书面通知可解除。
"""


# ================================================================
#  完整流程
# ================================================================

def analyze_contract(contract: str, objective: str) -> dict:
    """
    合同分析的完整流程。

    Java 类比：
        @Service
        public class ContractAnalyzer {
            public Report analyze(String contract, String objective) {
                // 1. 分类
                ContractInfo info = classifier.classify(contract);
                // 2. 获取标准条款
                List<String> clauses = clauseDB.get(info.type);

                // 3. Map: 并行检查每个条款（就是 parallelStream）
                List<ClauseResult> clauseResults = clauses.parallelStream()
                    .map(clause -> clauseChecker.check(contract, clause))
                    .collect(toList());

                // 4. 生成审查角色
                List<String> roles = planner.createRoles(info);

                // 5. Map: 并行角色审查（又一个 parallelStream）
                List<RoleResult> roleResults = roles.parallelStream()
                    .map(role -> roleReviewer.review(contract, role))
                    .collect(toList());

                // 6. Reduce: 汇总报告
                return reportGenerator.generate(clauseResults, roleResults);
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：合同分类
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：合同分类】（LLM调用 #1）')
    print('=' * 60)

    system = ("分析合同，判断类型和行业。\n"
              "返回JSON：{\"contract_type\": \"劳动合同\", \"industry\": \"互联网\"}")
    result = extract_json(call_llm(contract[:2000], system))
    contract_type = result.get("contract_type", "未知")
    industry = result.get("industry", "未知")
    total_llm_calls += 1

    print(f'>>> 合同类型: {contract_type}')
    print(f'>>> 行业: {industry}')

    # ==========================================
    # 第2步：条款检索
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：条款检索】（不调LLM，查本地条款库）')
    print('=' * 60)

    clauses = get_clauses(contract_type)
    print(f'>>> 获取 {len(clauses)} 个标准条款')

    # ==========================================
    # 第3步：Map — 逐条检查（Send的本质）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第3步：条款检查 — Map】')
    print(f'    Send() 的本质：for clause in clauses: check(clause)')
    print(f'    共 {len(clauses)} 个条款，逐个调用LLM')
    print('=' * 60)

    clause_analysis = []       # operator.add 的本质：一个列表
    clause_modifications = []  # 不断 extend

    for i, clause in enumerate(clauses, 1):
        print(f'\n    [{i}/{len(clauses)}] 检查: {clause}')

        system = (f"你是法律条款审查专家。检查合同中是否清晰包含此条款：{clause}\n"
                  f"返回JSON：{{\"analysis\": \"分析\", \"modifications\": [{{\"original\": \"原文\", \"suggested\": \"建议\", \"reason\": \"原因\"}}]}}")

        result = extract_json(call_llm(contract[:2000], system))
        total_llm_calls += 1

        analysis = result.get("analysis", "分析失败")
        mods = result.get("modifications", [])

        clause_analysis.append(f"【{clause}】{analysis}")
        clause_modifications.extend(mods)  # ← 这就是 operator.add

        print(f'    → {len(mods)}条修改建议')

    print(f'\n    >>> Map完成: {len(clause_analysis)}条分析, {len(clause_modifications)}条修改建议')

    # ==========================================
    # 第4步：生成审查角色
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第4步：生成审查角色】（LLM调用 #{total_llm_calls + 1}）')
    print('=' * 60)

    system = (f"为{contract_type}（{industry}行业）生成3-5个审查角色。\n"
              f"返回JSON：{{\"roles\": [\"劳动法专家\", \"合规审查官\"]}}")
    result = extract_json(call_llm("生成审查角色", system))
    roles = result.get("roles", ["法律风险专家", "合规审查官", "条款审查员"])
    total_llm_calls += 1

    print(f'>>> 审查角色: {roles}')

    # ==========================================
    # 第5步：Map — 多角色审查（又一个 Send）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第5步：角色审查 — Map】')
    print(f'    又一个 for 循环，共 {len(roles)} 个角色')
    print('=' * 60)

    role_analysis = []
    role_modifications = []

    for i, role in enumerate(roles, 1):
        print(f'\n    [{i}/{len(roles)}] {role} 审查中...')

        system = (f"你是{role}。从专业角度审查合同。\n"
                  f"返回JSON：{{\"analysis\": \"审查分析\", \"modifications\": [{{\"original\": \"原文\", \"suggested\": \"建议\", \"reason\": \"原因\"}}]}}")

        result = extract_json(call_llm(contract[:2000], system))
        total_llm_calls += 1

        analysis = result.get("analysis", "分析失败")
        mods = result.get("modifications", [])

        role_analysis.append(f"【{role}】{analysis}")
        role_modifications.extend(mods)

        print(f'    → {len(mods)}条修改建议')

    print(f'\n    >>> Map完成: {len(role_analysis)}条分析, {len(role_modifications)}条修改建议')

    # ==========================================
    # 第6步：Reduce — 生成报告
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第6步：生成报告 — Reduce】（LLM调用 #{total_llm_calls + 1}）')
    print('=' * 60)

    all_mods = clause_modifications + role_modifications

    if all_mods:
        system = ("总结合同修改建议，从法律角度解释每条建议的重要性。用中文。")
        mod_summary = call_llm(json.dumps(all_mods, ensure_ascii=False)[:3000], system)
        total_llm_calls += 1
    else:
        mod_summary = "未发现需要修改的内容。"

    print()
    print('=' * 50)
    print('         合同审查报告')
    print('=' * 50)
    print(f'合同类型: {contract_type}')
    print(f'行业: {industry}')
    print(f'条款检查: {len(clause_analysis)}个')
    print(f'角色审查: {len(role_analysis)}个')
    print(f'修改建议: {len(all_mods)}条')
    print(f'LLM调用总次数: {total_llm_calls}')
    print()
    print('【修改建议摘要】')
    print(mod_summary)

    return {
        "contract_type": contract_type,
        "industry": industry,
        "clause_analysis": clause_analysis,
        "role_analysis": role_analysis,
        "all_modifications": all_mods,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第14课 - 合同分析助手（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - Send() = for 循环，逐个调用同一函数')
    print('  - operator.add = list.extend()，把结果合并到列表')
    print('  - 两次 Map-Reduce = 两个 for 循环 + 两个结果列表')
    print('  - 整个系统 = 2次分类 + N次条款检查 + M次角色审查 + 1次报告')
    print()

    print('请输入合同文本（回车使用默认示例）:')
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

    objective = input('\n审查目标（回车默认"全面审查"）: ').strip() or "全面审查"

    result = analyze_contract(contract, objective)

    print()
    print('#' * 60)
    print(f'#  审查完成')
    print(f'#  LLM调用: {result["total_llm_calls"]}次')
    print(f'#  修改建议: {len(result["all_modifications"])}条')
    print('#' * 60)
