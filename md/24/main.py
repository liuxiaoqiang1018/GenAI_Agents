"""
第24课 - 自我改进代理（框架版 - SelfImprovingAgent类）

核心概念：
  - 反思-学习循环：对话→反思→学习→改进后的对话
  - 洞察注入：学习成果作为system prompt的一部分
  - 三个能力：respond（对话）、reflect（反思）、learn（学习）

本课不使用LangGraph，聚焦于自我改进机制。
"""

import os
import time
from typing import List, Dict

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3
MAX_HISTORY = 20


def call_llm(messages: list) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
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


# ========== 自我改进Agent ==========

class SelfImprovingAgent:
    """能够通过反思和学习自我改进的Agent"""

    def __init__(self):
        self.history: List[Dict] = []    # 对话历史（短期记忆）
        self.insights: str = ""           # 改进洞察（学习成果）
        self.learn_count: int = 0         # 学习次数

    def respond(self, user_input: str) -> str:
        """对话：回答用户问题（注入改进洞察）"""

        # 组装 prompt：system + 洞察 + 历史 + 当前输入
        messages = [
            {"role": "system", "content": (
                "你是一个自我改进的AI助手。用中文回答。\n"
                "根据之前的学习洞察不断改进你的回答质量。"
            )},
        ]

        # 注入改进洞察
        if self.insights:
            messages.append({"role": "system", "content": f"改进洞察（请据此优化回答）：{self.insights}"})

        # 加入对话历史
        history_slice = self.history[-MAX_HISTORY:] if len(self.history) > MAX_HISTORY else self.history
        messages.extend(history_slice)

        # 当前输入
        messages.append({"role": "user", "content": user_input})

        # 调用LLM
        response = call_llm(messages)

        # 更新对话历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    def reflect(self) -> str:
        """反思：回顾对话历史，生成改进洞察"""
        print()
        print('--- 反思中 ---')

        messages = [
            {"role": "system", "content": (
                "你是一个AI教练。回顾以下对话历史，分析AI助手的回答质量。\n"
                "指出哪些回答可以改进，给出具体的改进建议。\n"
                "例如：回答是否太长/太短、是否有遗漏、语气是否合适等。\n"
                "用中文回答。"
            )},
        ]

        # 把对话历史作为待分析内容
        for msg in self.history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": "请对以上对话进行反思，生成改进洞察。"})

        self.insights = call_llm(messages)
        print(f'>>> 反思洞察: {self.insights[:150]}...')
        return self.insights

    def learn(self) -> str:
        """学习：从反思洞察中提取关键要点"""
        # 先反思
        self.reflect()

        print()
        print('--- 学习中 ---')

        messages = [
            {"role": "system", "content": (
                "从以下改进洞察中提取3-5条具体的行为要点。\n"
                "这些要点将用于改进后续对话。\n"
                "格式：每条一行，简短精炼。用中文。"
            )},
            {"role": "user", "content": self.insights},
        ]

        learned = call_llm(messages)
        self.insights = learned  # 用精炼后的要点替换原始洞察
        self.learn_count += 1

        # 记录学习事件到历史
        self.history.append({
            "role": "assistant",
            "content": f"[系统] 第{self.learn_count}次学习完成，改进要点：{learned}"
        })

        print(f'>>> 学习要点: {learned}')
        print(f'>>> 累计学习: {self.learn_count}次')
        return learned

    def show_status(self):
        """显示Agent状态"""
        print()
        print('=' * 40)
        print(f'  Agent 状态')
        print('=' * 40)
        print(f'  对话轮数: {len(self.history) // 2}')
        print(f'  学习次数: {self.learn_count}')
        print(f'  当前洞察: {self.insights[:100] if self.insights else "暂无"}')
        print('=' * 40)


# ========== 运行 ==========

if __name__ == '__main__':
    print('第24课 - 自我改进代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('命令：')
    print('  /learn   - 触发反思+学习')
    print('  /status  - 查看Agent状态')
    print('  /quit    - 退出')
    print()
    print('提示：聊几轮后输入 /learn，Agent会反思并改进后续回答')
    print()

    agent = SelfImprovingAgent()

    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        if user_input == '/status':
            agent.show_status()
            continue

        if user_input == '/learn':
            if len(agent.history) < 4:
                print('>>> 对话太少，先多聊几轮再学习')
                continue
            agent.learn()
            print('>>> 学习完成，后续回答将会改进')
            continue

        response = agent.respond(user_input)
        print(f'\nAI: {response}\n')
