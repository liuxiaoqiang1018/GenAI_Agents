"""
第28课 - 销售通话分析器内部机制

目的：让你看清 CrewAI Agent/Task/Crew 模式的本质：
  1. Agent = 一段 system prompt（role + goal + backstory 拼起来）
  2. Task = 一段 user prompt（description + expected_output 拼起来）
  3. Crew.kickoff() = call_llm(agent.system_prompt, task.description)
  4. 整个系统 = 一次精心组织的 LLM 调用

CrewAI 的价值不在技术复杂度，而在"清晰的职责划分":
  - Agent 定义"谁"（角色、能力）
  - Task 定义"做什么"（任务、期望输出）
  - Crew 定义"怎么组织"（谁做哪个任务）
"""

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(messages: list) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.5},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not choices[0].get("message"):
                raise ValueError("API返回空响应")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ========== 示例通话 ==========

EXAMPLE_CALL = """
销售：您好，我是智云科技的小张，请问是李总吗？
客户：嗯，我是。你有什么事？
销售：李总好，我们是做企业级AI解决方案的。
客户：是的，我们公司人工成本越来越高，想看看自动化方案。
销售：我们的AI客服系统可以自动处理80%常见咨询，每月节省3-5人工成本。
客户：听起来不错，但之前试过的智能客服太死板了。
销售：我们用大语言模型技术，理解能力完全不同。有一个月免费试用。
客户：那倒可以考虑。我需要和技术部确认对接问题。
销售：我安排技术顾问和您团队开个线上会。下周二下午可以吗？
客户：好的，下周二下午可以。
销售：好的李总，下周二见！
"""


# ================================================================
#  完整流程：CrewAI 模式的本质
# ================================================================

def analyze_call(call_text: str) -> str:
    """
    销售通话分析的完整流程。

    CrewAI 本质拆解：
        # Agent = system prompt
        system = "你是AI销售通话分析师。目标：提供洞察。背景：资深专家。"

        # Task = user prompt
        user = "分析这段通话：情感、关键词、痛点、效果、建议。返回JSON。"

        # Crew.kickoff() = call_llm(system, user)
        result = call_llm([system, user])

        # 就这么简单。CrewAI 的价值在于"组织清晰"，不在于"技术复杂"。

    Java 类比：
        // Agent = @Service 注解的类
        @Service("销售通话分析师")
        public class CallAnalysisAgent {
            // Task = @Override 的方法
            @Override
            public AnalysisReport analyze(String transcription) {
                return llm.invoke(buildPrompt(transcription));
            }
        }
        // Crew = Spring ApplicationContext 组织和执行
        context.getBean(CallAnalysisAgent.class).analyze(text);
    """

    # ==========================================
    # Agent 定义（就是拼 system prompt）
    # ==========================================
    print()
    print('=' * 60)
    print('【Agent 定义】（本质 = system prompt）')
    print('=' * 60)

    role = "AI销售通话分析师"
    goal = "从通话中提供可执行的洞察，帮助提升销售表现"
    backstory = "你是资深销售培训专家和数据分析师，擅长分析通话发现机会和问题"

    system_prompt = f"你是{role}。\n目标：{goal}\n背景：{backstory}"
    print(f'  role: {role}')
    print(f'  goal: {goal}')
    print(f'  backstory: {backstory}')
    print(f'  → system_prompt = role + goal + backstory 拼起来')

    # ==========================================
    # Task 定义（就是拼 user prompt）
    # ==========================================
    print()
    print('=' * 60)
    print('【Task 定义】（本质 = user prompt）')
    print('=' * 60)

    task_desc = ("对以下销售通话进行全面分析：\n"
                 "1. 情感分析：客户情绪变化\n"
                 "2. 关键短语：核心需求和关注点\n"
                 "3. 痛点识别：问题和不满\n"
                 "4. 销售效果：表现评估（1-10分）\n"
                 "5. 改进建议：3-5条具体建议\n"
                 "6. 成交概率：高/中/低\n\n"
                 f"通话记录：\n{call_text}\n\n"
                 "请返回JSON格式的分析报告。")

    print(f'  description: 情感+关键词+痛点+效果+建议+概率')
    print(f'  expected_output: JSON格式')
    print(f'  → user_prompt = description + 通话内容 + 输出格式')

    # ==========================================
    # Crew.kickoff()（就是一次 LLM 调用）
    # ==========================================
    print()
    print('=' * 60)
    print('【Crew.kickoff()】（本质 = call_llm(system, user)）')
    print('=' * 60)
    print('  → 把 Agent 的 system_prompt 和 Task 的 description 组装成 messages')
    print('  → 调用一次 LLM')
    print('  → 返回结果')

    result = call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
    ])

    print()
    print('=' * 60)
    print('【分析报告】')
    print('=' * 60)
    print(result)

    print()
    print('>>> CrewAI 模式总结:')
    print('    Agent(role+goal+backstory) = 一段精心设计的 system prompt')
    print('    Task(description+output)   = 一段详细的 user prompt')
    print('    Crew.kickoff()             = call_llm(system, user)')
    print('    LLM调用次数: 1次')

    return result


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第28课 - 销售通话分析器（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - Agent = system prompt（role+goal+backstory拼起来）')
    print('  - Task = user prompt（description+期望输出拼起来）')
    print('  - Crew = call_llm(agent, task)（就一次调用）')
    print('  - CrewAI的价值在组织清晰，不在技术复杂')
    print()

    print('请输入通话记录（回车使用示例）:')
    first_line = input().strip()
    if not first_line:
        call_text = EXAMPLE_CALL
        print('>>> 使用示例')
    else:
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        call_text = '\n'.join(lines)

    analyze_call(call_text)
