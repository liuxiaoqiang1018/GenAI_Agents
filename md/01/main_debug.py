"""
第1课 - Agent 内部运行机制演示（不使用任何框架）

目的：让你看清 Agent 内部的每一步：
  1. Agent 拼装 Prompt（包含工具描述）发给 LLM
  2. LLM 决策：要不要调用工具
  3. Agent 执行工具，拿到结果
  4. Agent 再拼装 Prompt（包含工具结果）发给 LLM
  5. LLM 生成最终回复
"""

import json
import re


# ================================================================
#  【Tool 层】—— 相当于 Java 的 @Service 方法
# ================================================================

def calculate_course(distance_meters: int) -> str:
    """根据起航线距离计算航线方案"""
    return f'航线长度：{distance_meters}米，建议航向：185°，预计用时：{distance_meters / 100:.1f}分钟'


# 工具注册表（相当于 Spring 容器中注册的 Bean）
TOOLS = {
    'calculate_course': {
        'function': calculate_course,
        'description': '根据起航线距离计算航线方案。当用户提到起航线或航线距离时调用此工具。',
        'parameters': {
            'distance_meters': {'type': 'int', 'description': '起航线距离，单位为米'}
        }
    }
}


# ================================================================
#  【LLM 层】—— 模拟大模型（fake_llm）
# ================================================================

def fake_llm(prompt: str) -> str:
    """模拟 LLM 的行为。

    真实场景中，这里是调用 OpenAI/DeepSeek 等 API。
    现在用规则模拟，让你看清 LLM 的输入和输出。
    """
    print(prompt)
    print()

    # 模拟决策阶段：如果 prompt 中包含工具列表，LLM 需要判断是否调用工具
    if '你可以使用以下工具' in prompt:
        # 模拟 LLM 理解语义后的决策
        match = re.search(r'(\d+)\s*米', prompt)
        if match:
            distance = int(match.group(1))
            result = json.dumps(
                {'tool': 'calculate_course', 'args': {'distance_meters': distance}},
                ensure_ascii=False
            )
        else:
            result = json.dumps({'tool': None, 'args': {}}, ensure_ascii=False)

        print(f'[LLM输出-决策阶段]')
        print(result)
        print()
        return result

    # 模拟最终回答阶段：根据工具结果生成自然语言
    if '工具返回结果' in prompt:
        # 提取工具结果，生成自然语言回复
        tool_result_match = re.search(r'工具返回结果：\n(.+)', prompt)
        tool_result = tool_result_match.group(1) if tool_result_match else '无结果'
        final_answer = f'根据计算，{tool_result}。请注意航行安全！'

        print(f'[LLM输出-最终回答]')
        print(final_answer)
        print()
        return final_answer

    # 普通对话（不涉及工具）
    answer = '你好！我是AI助手，有什么可以帮你的？'
    print(f'[LLM输出]')
    print(answer)
    print()
    return answer


# ================================================================
#  【Agent 层】—— 调度员，串联 LLM 和 Tool
# ================================================================

def build_tool_description() -> str:
    """把注册的工具信息拼成 LLM 能理解的文本"""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f'- {name}: {info["description"]}')
        lines.append(f'  参数：')
        for param_name, param_info in info['parameters'].items():
            lines.append(f'    - {param_name}: {param_info["type"]}（{param_info["description"]}）')
    return '\n'.join(lines)


def agent_run(user_input: str) -> str:
    """Agent 的核心运行逻辑"""

    # ==========================================
    # 第1步：拼装决策 Prompt，发给 LLM
    # ==========================================
    tool_desc = build_tool_description()

    prompt1 = f"""===== PROMPT 1（决策阶段）=====
你是一个AI助手，你可以使用以下工具：

工具列表：
{tool_desc}

请根据用户输入判断：
1. 是否需要调用工具
2. 如果需要，返回 JSON 格式：
{{
  "tool": "工具名",
  "args": {{ 参数 }}
}}
3. 如果不需要工具，返回：
{{
  "tool": null,
  "args": {{}}
}}

用户输入：
{user_input}
=============================="""

    print()
    print('=' * 50)
    print('【1️⃣  决策阶段 - Agent 拼装 Prompt 发给 LLM】')
    print('=' * 50)
    print()

    llm_decision = fake_llm(prompt1)

    # ==========================================
    # 第2步：解析 LLM 的决策
    # ==========================================
    decision = json.loads(llm_decision)
    tool_name = decision.get('tool')

    if not tool_name:
        # LLM 认为不需要工具，直接回复
        print('=' * 50)
        print('【LLM 决定：不需要调用工具，直接回复】')
        print('=' * 50)
        return llm_decision

    # ==========================================
    # 第3步：执行 Tool
    # ==========================================
    print('=' * 50)
    print('【3️⃣  工具执行阶段】')
    print('=' * 50)
    print()

    tool_info = TOOLS[tool_name]
    tool_func = tool_info['function']
    tool_args = decision['args']

    print(f'[Agent] 调用工具: {tool_name}')
    print(f'[Agent] 参数: {json.dumps(tool_args, ensure_ascii=False)}')
    print()

    # 实际执行工具函数
    tool_result = tool_func(**tool_args)

    print(f'[Tool] 执行: {tool_name}')
    print(f'[Tool] 输入参数: {tool_args}')
    print(f'[Tool] 返回结果: {tool_result}')
    print()

    # ==========================================
    # 第4步：把工具结果喂回 LLM，生成最终回答
    # ==========================================
    prompt2 = f"""===== PROMPT 2（生成最终回答）=====
用户输入：
{user_input}

工具返回结果：
{tool_result}

请生成自然语言回答，让用户易于理解。
=============================="""

    print('=' * 50)
    print('【4️⃣  最终回答阶段 - Agent 把工具结果拼进 Prompt 再发给 LLM】')
    print('=' * 50)
    print()

    final_answer = fake_llm(prompt2)

    return final_answer


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('Agent 内部运行机制演示')
    print('输入 /quit 退出\n')

    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        print()
        print('#' * 60)
        print(f'#  用户输入: {user_input}')
        print('#' * 60)

        result = agent_run(user_input)

        print('=' * 50)
        print('【5️⃣  最终返回给用户的回复】')
        print('=' * 50)
        print(f'\nAI: {result}\n')
