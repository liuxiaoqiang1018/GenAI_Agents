"""
第23课 - 多Agent协作系统（框架版 - Agent类+协作流程）

核心概念：
  - Agent类：名字+角色+技能 → 不同的system prompt
  - 共享上下文：所有Agent读写同一个context列表
  - 固定轮换：研究Agent→分析Agent→研究Agent→分析Agent→研究Agent
  - 5步协作：背景→数据需求→数据提供→数据分析→综合总结

本课不使用LangGraph，聚焦于多Agent协作设计模式。
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


# ========== Agent 类 ==========

class Agent:
    """Agent基类：角色化的LLM调用"""

    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills

    def process(self, task: str, context: List[Dict] = None) -> str:
        """处理任务，可选传入共享上下文"""
        messages = [
            {"role": "system", "content": (
                f"你是{self.name}，一位{self.role}。\n"
                f"你的专业技能包括：{'、'.join(self.skills)}。\n"
                f"请基于你的角色和技能回答任务。用中文回答。"
            )}
        ]

        # 注入共享上下文
        if context:
            for msg in context:
                if msg["role"] == "human":
                    messages.append({"role": "user", "content": msg["content"]})
                else:
                    messages.append({"role": "assistant", "content": msg["content"]})

        messages.append({"role": "user", "content": task})
        return call_llm(messages)

    def __repr__(self):
        return f"Agent({self.name}, {self.role})"


# ========== 专业Agent ==========

class ResearchAgent(Agent):
    """研究Agent：领域知识专家"""
    def __init__(self):
        super().__init__(
            "研究员小李",
            "领域研究专家",
            ["深厚的领域知识", "历史背景分析", "趋势识别", "数据搜集"]
        )


class AnalysisAgent(Agent):
    """分析Agent：数据分析专家"""
    def __init__(self):
        super().__init__(
            "分析师小王",
            "数据分析专家",
            ["数据解读", "统计分析", "趋势预测", "可视化描述"]
        )


# ========== 协作步骤 ==========

def step1_research_context(agent: Agent, task: str, context: list) -> list:
    """第1步：研究Agent提供背景"""
    print()
    print(f'【第1步】{agent.name}：研究背景...')
    result = agent.process(f"为以下问题提供相关背景知识和上下文：{task}")
    context.append({"role": "ai", "content": f"{agent.name}：{result}"})
    print(f'  >>> {result[:100]}...')
    return context


def step2_identify_needs(agent: Agent, task: str, context: list) -> list:
    """第2步：分析Agent识别数据需求"""
    print()
    print(f'【第2步】{agent.name}：识别数据需求...')
    history = context[-1]["content"]
    result = agent.process(
        f"基于已有的背景知识，回答这个问题还需要哪些数据或统计信息？\n背景：{history}",
        context
    )
    context.append({"role": "ai", "content": f"{agent.name}：{result}"})
    print(f'  >>> {result[:100]}...')
    return context


def step3_provide_data(agent: Agent, task: str, context: list) -> list:
    """第3步：研究Agent提供数据"""
    print()
    print(f'【第3步】{agent.name}：提供数据...')
    needs = context[-1]["content"]
    result = agent.process(
        f"根据数据需求，提供相关的数据和统计信息。\n数据需求：{needs}",
        context
    )
    context.append({"role": "ai", "content": f"{agent.name}：{result}"})
    print(f'  >>> {result[:100]}...')
    return context


def step4_analyze(agent: Agent, task: str, context: list) -> list:
    """第4步：分析Agent分析数据"""
    print()
    print(f'【第4步】{agent.name}：分析数据...')
    data = context[-1]["content"]
    result = agent.process(
        f"分析提供的数据，描述趋势和洞察。\n数据：{data}",
        context
    )
    context.append({"role": "ai", "content": f"{agent.name}：{result}"})
    print(f'  >>> {result[:100]}...')
    return context


def step5_synthesize(agent: Agent, task: str, context: list) -> str:
    """第5步：研究Agent综合总结"""
    print()
    print(f'【第5步】{agent.name}：综合总结...')
    result = agent.process(
        "基于所有背景知识、数据和分析，给出对原始问题的综合性回答。",
        context
    )
    print(f'  >>> {result[:100]}...')
    return result


# ========== 协作系统 ==========

class CollaborationSystem:
    """多Agent协作系统"""

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()

    def solve(self, task: str) -> str:
        print()
        print('#' * 60)
        print(f'#  开始协作解决: {task[:50]}...')
        print('#' * 60)

        context = []

        # 固定轮换协议：A→B→A→B→A
        steps = [
            (step1_research_context, self.research_agent),   # A: 研究背景
            (step2_identify_needs,   self.analysis_agent),   # B: 识别需求
            (step3_provide_data,     self.research_agent),   # A: 提供数据
            (step4_analyze,          self.analysis_agent),   # B: 分析数据
        ]

        for step_func, agent in steps:
            context = step_func(agent, task, context)

        # 最后一步返回字符串（不再追加到context）
        final = step5_synthesize(self.research_agent, task, context)

        print()
        print('#' * 60)
        print('#  协作完成')
        print('#' * 60)
        print(f'  总协作步骤: 5')
        print(f'  上下文条目: {len(context)}')

        return final


# ========== 运行 ==========

if __name__ == '__main__':
    print('第23课 - 多Agent协作系统')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('Agent团队:')
    system = CollaborationSystem()
    print(f'  - {system.research_agent}')
    print(f'  - {system.analysis_agent}')
    print()

    examples = [
        "中国互联网行业在2010-2020年间的发展趋势是什么？哪些因素推动了这些变化？",
        "人工智能对全球就业市场产生了什么影响？未来趋势如何？",
    ]

    print('示例问题:')
    for i, q in enumerate(examples, 1):
        print(f'  {i}. {q}')
    print()

    question = input('请输入复杂问题（回车用示例1）: ').strip()
    if not question:
        question = examples[0]
        print(f'>>> 使用: {question}')

    result = system.solve(question)

    print()
    print('=' * 60)
    print('【最终答案】')
    print('=' * 60)
    print(result)
