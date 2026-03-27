"""
第6课 - 多智能体协调系统内部机制演示（不使用 LangGraph）

目的：让你看清多智能体系统内部做了什么：
  1. 协调器（Coordinator）分析请求，决定激活哪些 Agent
  2. 条件路由：根据协调器结果选择分支
  3. 并行执行：多个 Agent 同时处理（这里用顺序模拟）
  4. 结果汇总：合并所有 Agent 的输出

对比 main.py（LangGraph 框架版），理解框架帮你封装了什么：
  - StateGraph → 就是一个 {节点名: 函数} 的字典 + 路由规则
  - add_conditional_edges → 就是一个 if-else 分支
  - Annotated[Dict, dict_reducer] → 就是 dict.update() 的深度合并版
  - 并行执行 → 在同步版里就是依次调用
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化 API 客户端
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE_URL'),
    default_headers={"User-Agent": "Mozilla/5.0"},  # 第三方API可能屏蔽SDK默认UA
)
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


# ================================================================
#  【数据层】模拟学生数据
# ================================================================

STUDENT_PROFILE = {
    "姓名": "张明", "年级": "大三", "专业": "计算机科学",
    "学习风格": "视觉型",
    "擅长科目": ["数据结构", "操作系统"],
    "薄弱科目": ["高等数学", "概率论"],
    "学习偏好": {"最佳时段": "上午9-12点", "单次专注时长": "45分钟", "喜欢的方式": "思维导图 + 做题"},
}

STUDENT_CALENDAR = {
    "事件": [
        {"名称": "高等数学课", "时间": "周一 8:00-10:00"},
        {"名称": "数据结构课", "时间": "周二 10:00-12:00"},
        {"名称": "篮球训练", "时间": "周三 16:00-18:00"},
        {"名称": "概率论课", "时间": "周四 14:00-16:00"},
        {"名称": "编程社团", "时间": "周五 19:00-21:00"},
    ],
}

STUDENT_TASKS = {
    "任务": [
        {"名称": "高数作业第5章", "截止": "周三", "优先级": "高", "预计耗时": "3小时"},
        {"名称": "数据结构实验报告", "截止": "周五", "优先级": "中", "预计耗时": "4小时"},
        {"名称": "概率论复习", "截止": "下周一考试", "优先级": "高", "预计耗时": "8小时"},
    ],
}


# ================================================================
#  【LLM 层】
# ================================================================

def call_llm(prompt: str, temperature: float = 0.5) -> dict:
    """调用 LLM"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return {
        "content": response.choices[0].message.content.strip(),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


# ================================================================
#  【State 层】模拟 LangGraph 的状态管理
# ================================================================

def dict_reducer(d1: dict, d2: dict) -> dict:
    """
    深度合并字典。
    这就是 LangGraph 的 Annotated[Dict, dict_reducer] 内部做的事。

    Java 类比：
        public static Map<String, Object> deepMerge(Map a, Map b) {
            Map merged = new HashMap(a);
            for (var entry : b.entrySet()) {
                if (merged.containsKey(entry.getKey())
                    && merged.get(entry.getKey()) instanceof Map
                    && entry.getValue() instanceof Map) {
                    merged.put(entry.getKey(), deepMerge(
                        (Map)merged.get(entry.getKey()),
                        (Map)entry.getValue()));
                } else {
                    merged.put(entry.getKey(), entry.getValue());
                }
            }
            return merged;
        }
    """
    merged = d1.copy()
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged


# ================================================================
#  【Agent 层】各个 Agent 的实现
# ================================================================

def coordinator_agent(state: dict) -> dict:
    """
    协调器：分析请求，决定需要哪些 Agent。

    Java 类比：
        public class Coordinator {
            public RoutingDecision analyze(Request request) {
                // 分析请求内容，返回需要的Agent列表
                return new RoutingDecision(List.of("PLANNER", "ADVISOR"));
            }
        }
    这就像微服务架构中的 API Gateway / Router。
    """
    query = state["messages"][-1]

    prompt = f"""你是一个学术辅助系统的协调器。分析用户请求，决定需要哪些Agent。

可用Agent：
- PLANNER（计划者）：日程安排、时间管理
- NOTEWRITER（笔记员）：学习笔记、复习材料
- ADVISOR（顾问）：学习建议、改进方案

学生信息：{json.dumps(state['profile'], ensure_ascii=False)}
待办任务：{json.dumps(state['tasks'], ensure_ascii=False)}

用户请求：{query}

请用JSON格式回复：{{"required_agents": ["PLANNER"], "reasoning": "原因"}}
只列出确实需要的Agent。"""

    print()
    print('=' * 60)
    print('【1 - 协调器（Coordinator）】')
    print('=' * 60)
    print()
    print('>>> 发送给 LLM 的 Prompt（核心部分）:')
    print(f'    用户请求: {query}')
    print(f'    可用Agent: PLANNER, NOTEWRITER, ADVISOR')
    print()

    result = call_llm(prompt)
    print(f'>>> LLM 返回: {result["content"]}')
    print(f'>>> Token: {result["usage"]}')
    print()

    # 解析（LLM 返回可能包含额外文本，需要容错）
    response_text = result["content"]
    analysis = None
    try:
        analysis = json.loads(response_text)
    except json.JSONDecodeError:
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
        agents = []
        for name in ["PLANNER", "NOTEWRITER", "ADVISOR"]:
            if name.lower() in response_text.lower() or name in response_text:
                agents.append(name)
        analysis = {"required_agents": agents or ["PLANNER"], "reasoning": "从文本中提取"}

    print(f'>>> 协调器决策: 需要 {analysis["required_agents"]}')
    print(f'>>> 原因: {analysis.get("reasoning", "")}')

    return {"required_agents": analysis["required_agents"], "reasoning": analysis.get("reasoning", "")}


def profile_analyzer_agent(state: dict) -> dict:
    """档案分析：提取学习特征"""
    prompt = f"""分析学生档案，用一句话总结学习特征。
档案：{json.dumps(state['profile'], ensure_ascii=False)}"""

    print()
    print('=' * 60)
    print('【2 - 档案分析】')
    print('=' * 60)

    result = call_llm(prompt)
    print(f'>>> 分析结果: {result["content"][:200]}')

    return {"profile_analysis": result["content"]}


def planner_agent(state: dict) -> dict:
    """计划者Agent"""
    prompt = f"""你是学习计划Agent。生成具体的时间安排。
学生特征：{state.get('profile_analysis', '')}
日程：{json.dumps(state['calendar'], ensure_ascii=False)}
任务：{json.dumps(state['tasks'], ensure_ascii=False)}
请求：{state['messages'][-1]}"""

    print()
    print('=' * 60)
    print('【3a - 计划者Agent（PLANNER）】')
    print('=' * 60)
    print()
    print('>>> 发送给 LLM 的 Prompt（核心部分）:')
    print(f'    学生日程: {len(state["calendar"].get("事件", []))} 项')
    print(f'    待办任务: {len(state["tasks"].get("任务", []))} 项')
    print()

    result = call_llm(prompt)
    print(f'>>> 计划者输出: {result["content"][:300]}...')
    print(f'>>> Token: {result["usage"]}')

    return {"planner_output": result["content"]}


def notewriter_agent(state: dict) -> dict:
    """笔记员Agent"""
    prompt = f"""你是学习笔记Agent。根据学生学习风格生成结构化学习材料。
学生特征：{state.get('profile_analysis', '')}
请求：{state['messages'][-1]}
如果是视觉型学习者，多用列表、分类、关键词。"""

    print()
    print('=' * 60)
    print('【3b - 笔记员Agent（NOTEWRITER）】')
    print('=' * 60)

    result = call_llm(prompt)
    print(f'>>> 笔记员输出: {result["content"][:300]}...')
    print(f'>>> Token: {result["usage"]}')

    return {"notewriter_output": result["content"]}


def advisor_agent(state: dict) -> dict:
    """顾问Agent"""
    prompt = f"""你是学习顾问Agent。提供个性化学习建议。
学生特征：{state.get('profile_analysis', '')}
学生信息：{json.dumps(state['profile'], ensure_ascii=False)}
请求：{state['messages'][-1]}
请提供具体建议和改进方向。"""

    print()
    print('=' * 60)
    print('【3c - 顾问Agent（ADVISOR）】')
    print('=' * 60)

    result = call_llm(prompt)
    print(f'>>> 顾问输出: {result["content"][:300]}...')
    print(f'>>> Token: {result["usage"]}')

    return {"advisor_output": result["content"]}


# ================================================================
#  【编排层】手写多智能体协调流程
# ================================================================

def run_multi_agent(query: str) -> str:
    """
    多智能体协调的完整流程。
    这就是 LangGraph 的 StateGraph + conditional_edges + compile + invoke 内部做的事。

    Java 类比：
        public class MultiAgentOrchestrator {
            private Coordinator coordinator;
            private Map<String, Agent> agents;

            public String process(String query) {
                // 1. 协调器分析
                RoutingDecision decision = coordinator.analyze(query);
                // 2. 档案分析
                ProfileAnalysis profile = profileAnalyzer.analyze(studentData);
                // 3. 条件路由 + 并行执行
                List<Future<String>> futures = new ArrayList<>();
                for (String agentName : decision.getRequiredAgents()) {
                    futures.add(executorService.submit(() -> agents.get(agentName).run(context)));
                }
                // 4. 汇总结果
                return mergeResults(futures);
            }
        }
    """

    # 初始化状态
    state = {
        "messages": [query],
        "profile": STUDENT_PROFILE,
        "calendar": STUDENT_CALENDAR,
        "tasks": STUDENT_TASKS,
        "results": {},
    }

    print()
    print('#' * 60)
    print(f'#  用户请求: {query}')
    print('#' * 60)

    # ==========================================
    # 第1步：协调器
    # ==========================================
    coordinator_result = coordinator_agent(state)
    required_agents = coordinator_result["required_agents"]

    # ==========================================
    # 第2步：档案分析
    # ==========================================
    profile_result = profile_analyzer_agent(state)
    state["profile_analysis"] = profile_result["profile_analysis"]

    # ==========================================
    # 第3步：条件路由 + Agent 执行
    # ==========================================
    print()
    print('=' * 60)
    print('【条件路由 - 这就是 add_conditional_edges 的本质】')
    print('=' * 60)
    print(f'>>> 协调器要求: {required_agents}')
    print(f'>>> 路由逻辑: if "PLANNER" in required → 走计划者分支')
    print(f'>>>           if "NOTEWRITER" in required → 走笔记员分支')
    print(f'>>>           if "ADVISOR" in required → 走顾问分支')
    print()

    agent_outputs = {}

    # LangGraph 会并行执行这些 Agent，这里用顺序执行模拟
    if "PLANNER" in required_agents:
        planner_result = planner_agent(state)
        agent_outputs["planner_output"] = planner_result["planner_output"]

    if "NOTEWRITER" in required_agents:
        notewriter_result = notewriter_agent(state)
        agent_outputs["notewriter_output"] = notewriter_result["notewriter_output"]

    if "ADVISOR" in required_agents:
        advisor_result = advisor_agent(state)
        agent_outputs["advisor_output"] = advisor_result["advisor_output"]

    # ==========================================
    # 第4步：结果汇总
    # ==========================================
    print()
    print('=' * 60)
    print('【4 - 结果汇总（State Reducer 合并）】')
    print('=' * 60)
    print()
    print('>>> 这就是 Annotated[Dict, dict_reducer] 做的事：')
    print(f'    合并前的 results: {list(state["results"].keys())}')

    state["results"] = dict_reducer(state["results"], agent_outputs)

    print(f'    合并后的 results: {list(state["results"].keys())}')
    print()

    # 组装最终输出
    outputs = []
    if "planner_output" in state["results"]:
        outputs.append(f"【学习计划】\n{state['results']['planner_output']}")
    if "notewriter_output" in state["results"]:
        outputs.append(f"【学习材料】\n{state['results']['notewriter_output']}")
    if "advisor_output" in state["results"]:
        outputs.append(f"【学习建议】\n{state['results']['advisor_output']}")

    final = "\n\n---\n\n".join(outputs) if outputs else "暂无结果"

    print('=' * 60)
    print('【最终输出】')
    print('=' * 60)
    print(final)
    print()

    return final


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第6课 - 多智能体协调系统内部机制演示（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示多智能体系统内部做了什么：')
    print('  - 协调器 = 分析请求 + 路由决策')
    print('  - 条件路由 = if-else 分支')
    print('  - 并行执行 = 依次调用多个 Agent（简化版）')
    print('  - State Reducer = dict 深度合并')
    print()

    # 示例请求
    examples = [
        "帮我规划下周的学习计划，我下周一有概率论考试",
        "我高数学不好，帮我制定复习计划并给一些学习建议",
    ]

    print('--- 示例请求 ---')
    for q in examples:
        run_multi_agent(q)

    # 交互模式
    print('\n输入请求，输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        run_multi_agent(user_input)
