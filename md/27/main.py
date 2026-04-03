"""
第27课 - AutoGen研究团队群聊（框架版 - 手工实现群聊模式）

核心概念：
  - 群聊：所有Agent共享一个对话线程
  - 发言权转移图：有向图定义谁之后可以是谁
  - Manager选择：LLM根据上下文从候选人中选下一个发言者
  - 5个角色：管理员、规划师、开发者、执行者、质检员

本课不使用AutoGen框架，手工实现群聊模式。
"""

import os
import json
import re
import time
from typing import Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3
MAX_ROUNDS = 15


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


# ========== Agent 定义 ==========

AGENTS = {
    "规划师": {
        "system": ("你是研究项目的规划师。负责：\n"
                   "1. 分析任务，制定研究计划\n"
                   "2. 将任务分解为具体步骤\n"
                   "3. 协调团队分工\n"
                   "回答时以 [规划师] 开头。用中文。简洁回答，100字以内。"),
    },
    "开发者": {
        "system": ("你是研究团队的开发者。负责：\n"
                   "1. 根据计划编写代码或查找资料\n"
                   "2. 实现具体的技术任务\n"
                   "3. 解决技术问题\n"
                   "回答时以 [开发者] 开头。用中文。简洁回答，100字以内。"),
    },
    "执行者": {
        "system": ("你是研究团队的执行者。负责：\n"
                   "1. 执行开发者提供的方案\n"
                   "2. 报告执行结果\n"
                   "3. 反馈遇到的问题\n"
                   "回答时以 [执行者] 开头。用中文。简洁回答，100字以内。"),
    },
    "质检员": {
        "system": ("你是研究团队的质检员。负责：\n"
                   "1. 审查研究结果的质量和完整性\n"
                   "2. 发现问题并指出需要改进的地方\n"
                   "3. 当结果满意时，宣布任务完成（说「任务完成」）\n"
                   "回答时以 [质检员] 开头。用中文。简洁回答，100字以内。"),
    },
}


# ========== 发言权转移图 ==========

TRANSITIONS = {
    "管理员": ["规划师", "质检员"],
    "规划师": ["管理员", "开发者", "质检员"],
    "开发者": ["执行者", "质检员", "管理员"],
    "执行者": ["开发者"],
    "质检员": ["规划师", "开发者", "执行者", "管理员"],
}


# ========== Manager：选择下一个发言者 ==========

def select_next_speaker(current_speaker: str, thread: List[str]) -> str:
    """Manager根据对话内容和转移规则选择下一个发言者"""
    candidates = TRANSITIONS.get(current_speaker, list(AGENTS.keys()))

    # 让LLM选择
    recent = "\n".join(thread[-6:])  # 最近6条消息
    system = (f"你是群聊主持人。根据对话内容，从候选人中选择最合适的下一个发言者。\n"
              f"候选人：{candidates}\n"
              f"只回复一个名字（规划师/开发者/执行者/质检员/管理员）。")

    choice = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"最近对话：\n{recent}\n\n谁应该下一个发言？"},
    ])

    # 匹配候选人
    for c in candidates:
        if c in choice:
            return c

    return candidates[0]  # 兜底


# ========== Agent 发言 ==========

def agent_speak(agent_name: str, thread: List[str]) -> str:
    """让指定Agent基于对话线程发言"""
    agent = AGENTS[agent_name]

    messages = [{"role": "system", "content": agent["system"]}]

    # 注入对话线程（所有人共享）
    for msg in thread[-10:]:
        messages.append({"role": "user", "content": msg})

    messages.append({"role": "user", "content": "请基于以上对话，做出你的回应。"})

    return call_llm(messages)


# ========== 群聊主循环 ==========

def group_chat(task: str):
    """群聊主循环"""

    print()
    print('#' * 60)
    print(f'#  研究团队群聊')
    print(f'#  任务: {task[:50]}...')
    print('#' * 60)

    # 对话线程（所有人共享）
    thread = [f"[管理员] {task}"]
    print(f'\n[管理员] {task}')

    current_speaker = "管理员"

    for round_num in range(1, MAX_ROUNDS + 1):
        # Manager选择下一个发言者
        next_speaker = select_next_speaker(current_speaker, thread)

        # 如果选到管理员，让用户输入
        if next_speaker == "管理员":
            print(f'\n--- 第{round_num}轮: 管理员（你）---')
            user_input = input('[管理员] ').strip()
            if not user_input:
                user_input = "继续"
            if user_input.lower() in ('/quit', '结束', '退出'):
                print('\n>>> 管理员结束了群聊')
                break
            thread.append(f"[管理员] {user_input}")
            current_speaker = "管理员"
        else:
            # AI Agent 发言
            print(f'\n--- 第{round_num}轮: {next_speaker} ---')
            response = agent_speak(next_speaker, thread)
            thread.append(f"[{next_speaker}] {response}")
            print(response)
            current_speaker = next_speaker

            # 检查是否完成
            if "任务完成" in response and next_speaker == "质检员":
                print('\n>>> 质检员宣布任务完成！')
                break

    # 输出对话记录
    print()
    print('=' * 60)
    print('【群聊记录】')
    print('=' * 60)
    for msg in thread:
        print(f'  {msg[:100]}')
    print(f'\n  总轮数: {round_num}')


# ========== 运行 ==========

if __name__ == '__main__':
    print('第27课 - AutoGen研究团队群聊')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('团队成员:')
    for name in AGENTS:
        print(f'  - {name}')
    print(f'  - 管理员（你）')
    print()
    print('发言权转移规则:')
    for src, dsts in TRANSITIONS.items():
        print(f'  {src} → {", ".join(dsts)}')
    print()
    print('命令：输入 /quit 结束群聊')
    print()

    examples = [
        "请研究目前最流行的5个AI Agent开发框架，比较它们的优缺点",
        "调查大语言模型在医疗领域的应用案例",
    ]

    print('示例任务:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    task = input('研究任务（回车用示例1）: ').strip()
    if not task:
        task = examples[0]
        print(f'>>> 使用: {task}')

    group_chat(task)
