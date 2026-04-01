"""
第23课 - 多Agent协作系统内部机制

目的：让你看清多Agent协作的本质：
  1. Agent = 一个函数，不同的system prompt就是不同的"专家"
  2. 协作 = 固定顺序调用函数，共享一个context列表
  3. 共享上下文 = list.append()，后面的Agent能读到前面所有Agent的输出
  4. 轮换协议 = for (func, agent) in steps: context = func(agent, context)

Java 类比：
  - Agent = 实现同一接口的不同Service
  - 协作 = Chain of Responsibility 或 Pipeline 模式
  - 共享上下文 = 共享的 DTO 在 Service 间传递
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


# ================================================================
#  Agent的本质：一个带角色的函数
# ================================================================

def agent_call(name: str, role: str, skills: str, task: str, context: list = None) -> str:
    """
    Agent 的本质就是一个函数调用，不同的参数产生不同的"专家"。

    Java 类比：
        // 策略模式
        interface Agent {
            String process(String task, List<Message> context);
        }
        class ResearchAgent implements Agent {
            String process(String task, List<Message> context) {
                return llm.generate(
                    new SystemMessage("你是研究员，技能：" + skills),
                    context,
                    new UserMessage(task)
                );
            }
        }
    """
    messages = [
        {"role": "system", "content": f"你是{name}，{role}。技能：{skills}。用中文回答。"},
    ]
    if context:
        # 把共享上下文注入（后面的Agent能看到前面所有Agent的输出）
        for entry in context:
            messages.append({"role": "assistant", "content": entry})
    messages.append({"role": "user", "content": task})
    return call_llm(messages)


# ================================================================
#  完整流程
# ================================================================

def collaborate(question: str) -> str:
    """
    多Agent协作的完整流程。

    Java 类比：
        public class CollaborationService {
            private Agent researchAgent = new ResearchAgent();
            private Agent analysisAgent = new AnalysisAgent();

            public String solve(String question) {
                // 共享上下文（就是一个 List）
                List<String> context = new ArrayList<>();

                // 固定轮换（就是按顺序调用不同的 Service）
                context.add(researchAgent.process("提供背景", context));  // A
                context.add(analysisAgent.process("识别需求", context));  // B
                context.add(researchAgent.process("提供数据", context));  // A
                context.add(analysisAgent.process("分析数据", context));  // B
                return researchAgent.process("综合总结", context);       // A
            }
        }
    """

    # 两个"Agent"其实就是两组参数
    RESEARCH = ("研究员小李", "领域研究专家", "领域知识、背景分析、数据搜集")
    ANALYSIS = ("分析师小王", "数据分析专家", "数据解读、统计分析、趋势预测")

    # 共享上下文（就是一个列表）
    context = []
    total_llm_calls = 0

    # ==========================================
    # 第1步：研究Agent提供背景（A）
    # ==========================================
    print()
    print(f'【第1步】{RESEARCH[0]}：研究背景...')
    result = agent_call(*RESEARCH,
                         f"为以下问题提供相关背景知识：{question}")
    context.append(f"{RESEARCH[0]}：{result}")  # ← 追加到共享上下文
    total_llm_calls += 1
    print(f'  >>> {result[:120]}...')

    # ==========================================
    # 第2步：分析Agent识别数据需求（B）
    # ==========================================
    print()
    print(f'【第2步】{ANALYSIS[0]}：识别数据需求...')
    print(f'    （能看到第1步的输出 → 共享上下文）')
    result = agent_call(*ANALYSIS,
                         f"基于已有背景，还需要哪些数据？\n背景：{context[-1]}",
                         context)  # ← 传入共享上下文
    context.append(f"{ANALYSIS[0]}：{result}")
    total_llm_calls += 1
    print(f'  >>> {result[:120]}...')

    # ==========================================
    # 第3步：研究Agent提供数据（A）
    # ==========================================
    print()
    print(f'【第3步】{RESEARCH[0]}：提供数据...')
    print(f'    （能看到第1+2步的输出 → context越来越丰富）')
    result = agent_call(*RESEARCH,
                         f"根据数据需求提供数据。\n需求：{context[-1]}",
                         context)
    context.append(f"{RESEARCH[0]}：{result}")
    total_llm_calls += 1
    print(f'  >>> {result[:120]}...')

    # ==========================================
    # 第4步：分析Agent分析数据（B）
    # ==========================================
    print()
    print(f'【第4步】{ANALYSIS[0]}：分析数据...')
    print(f'    （能看到第1+2+3步所有输出）')
    result = agent_call(*ANALYSIS,
                         f"分析数据，描述趋势和洞察。\n数据：{context[-1]}",
                         context)
    context.append(f"{ANALYSIS[0]}：{result}")
    total_llm_calls += 1
    print(f'  >>> {result[:120]}...')

    # ==========================================
    # 第5步：研究Agent综合总结（A）
    # ==========================================
    print()
    print(f'【第5步】{RESEARCH[0]}：综合总结...')
    print(f'    （能看到全部4步的输出 → 最全的上下文）')
    final = agent_call(*RESEARCH,
                        "基于所有背景知识、数据和分析，给出综合性回答。",
                        context)
    total_llm_calls += 1

    print()
    print(f'>>> 协作完成！')
    print(f'    轮换序列: A→B→A→B→A（研究→分析→研究→分析→研究）')
    print(f'    共享上下文条目: {len(context)}')
    print(f'    LLM调用: {total_llm_calls}次')

    return final


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第23课 - 多Agent协作系统（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - Agent = 带角色的函数调用（不同system prompt）')
    print('  - 协作 = 固定顺序调函数，共享一个list')
    print('  - 上下文 = list.append()，后面的能看到前面所有')
    print('  - 5步轮换 = A→B→A→B→A')
    print()

    examples = [
        "中国互联网行业2010-2020年的发展趋势是什么？",
        "人工智能对就业市场的影响和未来趋势？",
    ]

    print('示例:')
    for i, q in enumerate(examples, 1):
        print(f'  {i}. {q}')
    print()

    question = input('问题（回车用示例1）: ').strip()
    if not question:
        question = examples[0]
        print(f'>>> 使用: {question}')

    result = collaborate(question)

    print()
    print('=' * 60)
    print('【最终答案】')
    print('=' * 60)
    print(result)
