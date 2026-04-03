"""
第28课 - 销售通话分析器（框架版 - 实现CrewAI的Agent/Task/Crew模式）

核心概念：
  - Agent: 角色定义（role + goal + backstory → system prompt）
  - Task: 任务定义（description + expected_output）
  - Crew: 执行组织（agent执行task）
  - 结构化输出：JSON格式的多维度分析报告

本课不使用CrewAI框架，手工实现Agent/Task/Crew模式。
"""

import os
import json
import re
import time
from typing import List, Optional

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


# ========== CrewAI 模式实现 ==========

class Agent:
    """CrewAI Agent：定义"谁来做" """
    def __init__(self, role: str, goal: str, backstory: str):
        self.role = role
        self.goal = goal
        self.backstory = backstory

    @property
    def system_prompt(self) -> str:
        return (f"你是{self.role}。\n"
                f"目标：{self.goal}\n"
                f"背景：{self.backstory}")

    def execute(self, task_description: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_description},
        ]
        return call_llm(messages)


class Task:
    """CrewAI Task：定义"做什么" """
    def __init__(self, description: str, agent: Agent, expected_output: str = ""):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output

    def run(self, context: str = "") -> str:
        full_desc = self.description
        if context:
            full_desc = f"{self.description}\n\n待分析内容：\n{context}"
        if self.expected_output:
            full_desc += f"\n\n期望输出格式：{self.expected_output}"
        return self.agent.execute(full_desc)


class Crew:
    """CrewAI Crew：组织执行"""
    def __init__(self, agents: List[Agent], tasks: List[Task]):
        self.agents = agents
        self.tasks = tasks

    def kickoff(self, context: str = "") -> str:
        print()
        print('#' * 60)
        print(f'#  Crew 开始执行（{len(self.tasks)} 个任务）')
        print('#' * 60)

        results = []
        for i, task in enumerate(self.tasks, 1):
            print()
            print(f'--- 任务 {i}/{len(self.tasks)}: {task.agent.role} ---')
            result = task.run(context)
            results.append(result)
            print(f'>>> 完成（{len(result)}字）')

        return results[-1] if results else ""


# ========== 分析器 ==========

def build_analyzer():
    """构建销售通话分析的 Agent/Task/Crew"""

    # Agent：销售通话分析师
    analyst = Agent(
        role="AI销售通话分析师",
        goal="从销售通话中提供可执行的洞察和分析，帮助提升销售团队的表现",
        backstory=("你是一位资深的销售培训专家和数据分析师。"
                    "你擅长通过分析通话录音发现销售机会、识别客户痛点、"
                    "评估销售人员表现，并给出具体的改进建议。")
    )

    # Task：通话分析任务
    task = Task(
        description=("对以下销售通话记录进行全面分析，生成详细报告，包括：\n"
                     "1. 情感分析：评估客户在通话中的情绪变化（积极/中性/消极）\n"
                     "2. 关键短语：提取客户的核心需求和关注点（5-8个关键词）\n"
                     "3. 痛点识别：客户提到的问题、不满或担忧\n"
                     "4. 销售效果：评估销售人员的表现（开场、探需、方案、异议处理、促成）\n"
                     "5. 改进建议：给销售人员3-5条具体的改进建议\n"
                     "6. 成交概率：估算本次通话后的成交可能性（高/中/低）"),
        agent=analyst,
        expected_output=("返回JSON格式：\n"
                         '{"sentiment": {"overall": "中性", "trend": "从消极到积极"}, '
                         '"key_phrases": ["关键词1", "关键词2"], '
                         '"pain_points": ["痛点1", "痛点2"], '
                         '"effectiveness": {"score": 7, "strengths": ["..."], "weaknesses": ["..."]}, '
                         '"recommendations": ["建议1", "建议2"], '
                         '"closing_probability": "中"}')
    )

    crew = Crew(agents=[analyst], tasks=[task])
    return crew


# ========== 示例通话记录 ==========

EXAMPLE_CALL = """
销售：您好，我是智云科技的小张，请问是李总吗？
客户：嗯，我是。你有什么事？
销售：李总好，我们是做企业级AI解决方案的，之前您在我们网站上留过咨询信息。
客户：哦，是的，我确实在了解AI方面的东西。我们公司最近人工成本越来越高，想看看有没有什么自动化的方案。
销售：完全理解。很多企业都在面临这个问题。我们的AI客服系统可以帮您自动处理80%的常见咨询，每月能节省3-5个人工的成本。
客户：听起来不错，但是我有个担心，之前我们试过一个智能客服，客户反馈很差，觉得机器人回答太死板了。
销售：这个我非常理解。传统的规则引擎确实会这样。我们用的是大语言模型技术，理解能力和传统方案完全不同。而且我们有一个月的免费试用期，您可以先试试效果。
客户：免费试用？那倒可以考虑一下。不过我需要跟我们技术部的人确认一下技术对接的问题。
销售：没问题，我可以安排我们的技术顾问和您的技术团队开个线上会。您看这周有空吗？
客户：这周比较忙，下周二下午可以吗？
销售：完全没问题！下周二下午两点，我安排技术顾问准备一个针对您行业的演示方案。我稍后把会议链接发到您微信上。
客户：好的，那就先这样。
销售：好的李总，感谢您的时间！下周二见。
"""


# ========== 运行 ==========

if __name__ == '__main__':
    print('第28课 - 销售通话分析器')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('CrewAI 模式: Agent(谁做) + Task(做什么) + Crew(怎么执行)')
    print()

    # 获取通话记录
    print('请输入通话记录（回车使用示例，多行输入以 END 结束）:')
    first_line = input().strip()
    if not first_line:
        call_text = EXAMPLE_CALL
        print('>>> 使用示例通话记录')
    else:
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        call_text = '\n'.join(lines)

    # 构建并执行
    crew = build_analyzer()
    result = crew.kickoff(context=call_text)

    print()
    print('=' * 60)
    print('【分析报告】')
    print('=' * 60)
    print(result)
