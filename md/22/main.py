"""
第22课 - 记忆增强对话代理（框架版 - 使用LangChain记忆管理思路）

核心概念：
  - 短期记忆：messages列表，自动裁剪
  - 长期记忆：从对话中提取关键事实，注入system prompt
  - 会话隔离：不同session_id独立记忆
  - 双层记忆协同：短期保上下文连贯，长期保跨轮次记忆

本课不使用LangGraph，聚焦于记忆机制本身。
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
MAX_SHORT_TERM = 20   # 短期记忆最大消息数
MAX_LONG_TERM = 10    # 长期记忆最大事实数


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


# ========== 记忆存储 ==========

class MemoryStore:
    """双层记忆管理器"""

    def __init__(self):
        self.chat_store: Dict[str, List[dict]] = {}       # 短期记忆
        self.long_term_store: Dict[str, List[str]] = {}   # 长期记忆

    def get_short_term(self, session_id: str) -> List[dict]:
        """获取短期记忆（对话历史）"""
        if session_id not in self.chat_store:
            self.chat_store[session_id] = []
        return self.chat_store[session_id]

    def add_to_short_term(self, session_id: str, role: str, content: str):
        """追加到短期记忆，超长自动裁剪"""
        messages = self.get_short_term(session_id)
        messages.append({"role": role, "content": content})
        # 裁剪：保留最近 MAX_SHORT_TERM 条
        if len(messages) > MAX_SHORT_TERM:
            self.chat_store[session_id] = messages[-MAX_SHORT_TERM:]

    def get_long_term(self, session_id: str) -> str:
        """获取长期记忆（拼成字符串注入prompt）"""
        facts = self.long_term_store.get(session_id, [])
        return "。".join(facts) if facts else "暂无"

    def update_long_term(self, session_id: str, user_input: str, ai_response: str):
        """从对话中提取关键事实到长期记忆"""
        if session_id not in self.long_term_store:
            self.long_term_store[session_id] = []

        # 提取策略：用LLM从对话中提取事实
        extract_messages = [
            {"role": "system", "content": (
                "从以下对话中提取值得记住的关键事实（用户的名字、偏好、身份、需求等）。\n"
                "如果没有值得记住的信息，回复「无」。\n"
                "如果有，每条事实一行，简短精炼。")},
            {"role": "user", "content": f"用户说：{user_input}\nAI回复：{ai_response}"},
        ]
        extracted = call_llm(extract_messages)

        if extracted and '无' not in extracted:
            for line in extracted.split('\n'):
                line = line.strip().lstrip('-·•').strip()
                if line and len(line) > 3:
                    # 去重
                    if not any(line in existing for existing in self.long_term_store[session_id]):
                        self.long_term_store[session_id].append(line)

        # 裁剪：保留最近 MAX_LONG_TERM 条
        if len(self.long_term_store[session_id]) > MAX_LONG_TERM:
            self.long_term_store[session_id] = self.long_term_store[session_id][-MAX_LONG_TERM:]

    def show_memory(self, session_id: str):
        """展示当前记忆状态"""
        print()
        print('--- 记忆状态 ---')
        short = self.get_short_term(session_id)
        print(f'  短期记忆: {len(short)} 条消息')
        long_facts = self.long_term_store.get(session_id, [])
        print(f'  长期记忆: {len(long_facts)} 条事实')
        for f in long_facts:
            print(f'    - {f}')
        print('---')


# ========== 对话函数 ==========

def chat(memory: MemoryStore, session_id: str, user_input: str) -> str:
    """带双层记忆的对话"""

    # 1. 组装 prompt（注入长期记忆 + 短期历史）
    long_term_mem = memory.get_long_term(session_id)
    short_term = memory.get_short_term(session_id)

    messages = [
        {"role": "system", "content": (
            "你是一个友好的AI助手。如果长期记忆中有相关信息，请自然地使用。用中文回答。\n"
            f"长期记忆：{long_term_mem}")},
    ]
    # 加入短期历史
    messages.extend(short_term)
    # 加入当前输入
    messages.append({"role": "user", "content": user_input})

    # 2. 调用LLM
    response = call_llm(messages)

    # 3. 更新短期记忆
    memory.add_to_short_term(session_id, "user", user_input)
    memory.add_to_short_term(session_id, "assistant", response)

    # 4. 更新长期记忆（提取事实）
    memory.update_long_term(session_id, user_input, response)

    return response


# ========== 运行 ==========

if __name__ == '__main__':
    print('第22课 - 记忆增强对话代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('功能：')
    print('  - 短期记忆：记住当前对话上下文')
    print('  - 长期记忆：提取并记住关键事实（名字、偏好等）')
    print('  - 输入 /memory 查看记忆状态')
    print('  - 输入 /new 开启新会话')
    print('  - 输入 /quit 退出')
    print()

    memory = MemoryStore()
    session_id = "user_001"
    session_count = 1

    print(f'=== 会话 {session_count}（session: {session_id}）===')
    print()

    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        if user_input == '/memory':
            memory.show_memory(session_id)
            continue

        if user_input == '/new':
            session_count += 1
            session_id = f"user_{session_count:03d}"
            print(f'\n=== 新会话 {session_count}（session: {session_id}）===')
            print('>>> 短期记忆已清空，长期记忆保留')
            print()
            continue

        response = chat(memory, session_id, user_input)
        print(f'\nAI: {response}\n')
