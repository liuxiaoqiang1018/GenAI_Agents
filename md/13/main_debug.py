"""
第13课 - 项目管理助手内部机制（不使用 LangGraph）

目的：让你看清自反思循环的本质：
  1. 任务生成 = 一次LLM调用，返回JSON任务列表
  2. 依赖分析 = 一次LLM调用，返回JSON依赖关系
  3. 排期 = 一次LLM调用，返回JSON排期表
  4. 分配 = 一次LLM调用，返回JSON分配方案
  5. 风险评估 = 一次LLM调用，返回风险分数
  6. 洞察 = 一次LLM调用，返回文本建议
  7. 自反思 = while循环：如果风险没降低就重做3-6步

对比 main.py（LangGraph 框架版），理解：
  - 6个节点 → 就是6个函数
  - add_conditional_edges(router) → 就是 while 循环里的 if-break
  - insight→task_scheduler 的回边 → 就是循环体回到第3步
  - 迭代累积 → 就是往 list 里 append
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
        print(f'>>> JSON解析失败: {text[:200]}')
        return {}


# ========== 示例数据 ==========

EXAMPLE_PROJECT = "我们公司要开发一个智能客服聊天机器人，为客户提供7×24小时的产品咨询和售后支持服务。"

EXAMPLE_TEAM = [
    {"name": "张伟", "profile": "前端开发工程师，擅长React、Vue、HTML/CSS"},
    {"name": "李娜", "profile": "后端开发工程师，精通Python、Django、SQL"},
    {"name": "王强", "profile": "项目经理，敏捷开发经验，团队管理"},
    {"name": "刘洋", "profile": "全栈工程师，React+Node.js+MongoDB"},
    {"name": "陈静", "profile": "DevOps工程师，CI/CD、Docker、K8s"},
    {"name": "赵磊", "profile": "初级前端开发，HTML/CSS/JavaScript"},
    {"name": "孙芳", "profile": "资深数据科学家，机器学习、NLP、Python"},
]


# ================================================================
#  完整流程
# ================================================================

def plan_project(project_desc: str, team: list, max_iteration: int = 3) -> dict:
    """
    项目管理助手的完整流程。

    Java 类比：
        @Service
        public class ProjectPlannerService {
            @Retryable(maxAttempts = 3)
            public ProjectPlan plan(String description, Team team) {
                // 线性流水线（只做一次）
                List<Task> tasks = taskGenerator.generate(description);
                List<Dependency> deps = depAnalyzer.analyze(tasks);

                // 自反思循环（可能做多次）
                String insights = "";
                List<Integer> riskScores = new ArrayList<>();
                for (int i = 0; i < maxIteration; i++) {
                    Schedule schedule = scheduler.schedule(tasks, deps, insights);
                    Allocation alloc = allocator.assign(tasks, schedule, team, insights);
                    int risk = riskAssessor.assess(schedule, alloc);
                    riskScores.add(risk);
                    if (riskScores.size() > 1 && risk < riskScores.get(0)) break;
                    insights = insightGen.generate(schedule, alloc, risk);
                }
                return new ProjectPlan(tasks, schedule, allocation);
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：任务生成（只做一次）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：任务生成】（LLM调用 #1）')
    print('=' * 60)

    prompt = (f"你是资深项目经理。分析项目描述，提取任务列表。\n\n"
              f"项目描述：{project_desc}\n\n"
              f"超过5天的任务拆分为子任务。\n"
              f"返回JSON：{{\"tasks\": [{{\"name\": \"任务名\", \"description\": \"描述\", \"days\": 3}}]}}")

    result = extract_json(call_llm(prompt))
    tasks = result.get("tasks", [])
    total_llm_calls += 1

    print(f'>>> 生成 {len(tasks)} 个任务')
    for t in tasks:
        print(f'    - {t["name"]} ({t.get("days", "?")}天)')

    # ==========================================
    # 第2步：依赖分析（只做一次）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：依赖分析】（LLM调用 #2）')
    print('=' * 60)

    prompt = (f"分析任务依赖关系。\n"
              f"任务：{json.dumps(tasks, ensure_ascii=False)}\n"
              f"返回JSON：{{\"dependencies\": [{{\"task\": \"任务A\", \"depends_on\": [\"任务B\"]}}]}}")

    result = extract_json(call_llm(prompt))
    dependencies = result.get("dependencies", [])
    total_llm_calls += 1

    print(f'>>> 依赖关系:')
    for d in dependencies:
        deps = d.get("depends_on", [])
        print(f'    {d["task"]} → {"依赖: " + ", ".join(deps) if deps else "无依赖"}')

    # ==========================================
    # 第3-6步：自反思循环（可能做多次）
    # ==========================================
    print()
    print('*' * 60)
    print('*  进入自反思循环（这就是 while + if-break）')
    print('*' * 60)

    insights = ""
    risk_scores = []
    schedule = []
    allocations = []
    risks = []
    schedule_history = []
    allocation_history = []

    for iteration in range(1, max_iteration + 1):
        print()
        print(f'{"=" * 60}')
        print(f'  >>> 第 {iteration} 轮迭代')
        print(f'{"=" * 60}')

        # 第3步：排期
        print()
        print(f'    【第3步：任务排期】（LLM调用 #{total_llm_calls + 1}）')
        insights_part = f"\n上一轮洞察：{insights}" if insights else ""
        prompt = (f"安排任务排期。\n"
                  f"任务：{json.dumps(tasks, ensure_ascii=False)}\n"
                  f"依赖：{json.dumps(dependencies, ensure_ascii=False)}\n"
                  f"{insights_part}\n"
                  f"要求：尊重依赖、尽量并行化。\n"
                  f"返回JSON：{{\"schedule\": [{{\"task\": \"任务名\", \"start_day\": 1, \"end_day\": 3}}]}}")

        result = extract_json(call_llm(prompt))
        schedule = result.get("schedule", [])
        schedule_history.append(schedule)
        total_llm_calls += 1

        for s in schedule:
            print(f'      {s["task"]}: 第{s["start_day"]}天→第{s["end_day"]}天')

        # 第4步：人员分配
        print()
        print(f'    【第4步：人员分配】（LLM调用 #{total_llm_calls + 1}）')
        prompt = (f"分配任务给团队成员。\n"
                  f"排期：{json.dumps(schedule, ensure_ascii=False)}\n"
                  f"团队：{json.dumps(team, ensure_ascii=False)}\n"
                  f"任务：{json.dumps(tasks, ensure_ascii=False)}\n"
                  f"{insights_part}\n"
                  f"根据技能匹配，避免重叠。\n"
                  f"返回JSON：{{\"allocations\": [{{\"task\": \"任务名\", \"member\": \"成员名\"}}]}}")

        result = extract_json(call_llm(prompt))
        allocations = result.get("allocations", [])
        allocation_history.append(allocations)
        total_llm_calls += 1

        for a in allocations:
            print(f'      {a["task"]} → {a["member"]}')

        # 第5步：风险评估
        print()
        print(f'    【第5步：风险评估】（LLM调用 #{total_llm_calls + 1}）')
        prompt = (f"评估项目风险。\n"
                  f"排期：{json.dumps(schedule, ensure_ascii=False)}\n"
                  f"分配：{json.dumps(allocations, ensure_ascii=False)}\n"
                  f"团队：{json.dumps(team, ensure_ascii=False)}\n"
                  f"每个任务给0-10风险分。\n"
                  f"返回JSON：{{\"risks\": [{{\"task\": \"任务名\", \"score\": 5, \"reason\": \"原因\"}}], \"total_score\": 30}}")

        result = extract_json(call_llm(prompt))
        risks = result.get("risks", [])
        total_score = result.get("total_score", sum(r.get("score", 0) for r in risks))
        risk_scores.append(total_score)
        total_llm_calls += 1

        for r in risks:
            print(f'      {r["task"]}: {r.get("score", "?")}分')
        print(f'      总风险分: {total_score}')

        # 路由判断 —— 这就是 router 的本质
        print()
        print(f'    【路由判断】（就是 if-elif-else）')
        print(f'    风险分数历史: {risk_scores}')

        if len(risk_scores) > 1 and risk_scores[-1] < risk_scores[0]:
            print(f'    → 风险已降低（{risk_scores[0]}→{risk_scores[-1]}），break 退出循环')
            break
        elif iteration >= max_iteration:
            print(f'    → 达到最大迭代{max_iteration}次，break 退出循环')
            break
        else:
            print(f'    → 风险未降低，继续优化（生成洞察）')

        # 第6步：洞察生成
        print()
        print(f'    【第6步：洞察生成】（LLM调用 #{total_llm_calls + 1}）')
        prompt = (f"生成改进建议。\n"
                  f"排期：{json.dumps(schedule, ensure_ascii=False)}\n"
                  f"分配：{json.dumps(allocations, ensure_ascii=False)}\n"
                  f"风险：{json.dumps(risks, ensure_ascii=False)}\n"
                  f"给出具体改进建议。")

        insights = call_llm(prompt)
        total_llm_calls += 1
        print(f'      洞察: {insights[:200]}...')

    print()
    print(f'>>> 自反思循环结束，共 {len(risk_scores)} 轮，LLM调用 {total_llm_calls} 次')

    return {
        "tasks": tasks,
        "dependencies": dependencies,
        "schedule": schedule,
        "allocations": allocations,
        "risks": risks,
        "risk_scores": risk_scores,
        "iterations": len(risk_scores),
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第13课 - 项目管理助手（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 自反思循环 = while循环 + if-break')
    print('  - 6个节点 = 6个函数调用（前2个只做一次，后4个可能重复）')
    print('  - 结构化输出 = LLM返回JSON + json.loads()解析')
    print('  - 洞察反馈 = 把上一轮的建议塞进下一轮的prompt')
    print()

    print('请输入项目描述（回车使用默认示例）:')
    user_desc = input('项目描述: ').strip()
    if not user_desc:
        user_desc = EXAMPLE_PROJECT
        print(f'>>> 使用默认: {user_desc}')

    print()
    print('#' * 60)
    print('#  开始项目规划')
    print('#' * 60)

    result = plan_project(user_desc, EXAMPLE_TEAM, max_iteration=3)

    print()
    print('#' * 60)
    print('#  项目规划完成')
    print('#' * 60)
    print(f'  迭代次数: {result["iterations"]}')
    print(f'  LLM调用: {result["total_llm_calls"]}次')
    print(f'  风险变化: {result["risk_scores"]}')
    print()
    print('【最终任务列表】')
    for t in result["tasks"]:
        print(f'  - {t["name"]} ({t.get("days", "?")}天)')
    print()
    print('【最终排期】')
    for s in result["schedule"]:
        print(f'  {s["task"]}: 第{s["start_day"]}天→第{s["end_day"]}天')
    print()
    print('【最终人员分配】')
    for a in result["allocations"]:
        print(f'  {a["task"]} → {a["member"]}')
    print()
