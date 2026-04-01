"""
第24课 - 自我改进代理内部机制

目的：让你看清反思-学习循环的本质：
  1. respond = call_llm(system + insights + history + input)
  2. reflect = call_llm("分析以下对话历史的回答质量")
  3. learn = call_llm("从洞察中提取行为要点")
  4. 改进 = 把学习要点塞进后续 respond 的 system prompt

整个自我改进就是：
  - 对话几轮 → 用LLM审查自己的对话 → 提取改进建议 → 注入prompt → 后续回答变好

Java 类比：
  - 像 @AfterReturning 的AOP切面，定期触发自我评估
  - 或像CI/CD的代码审查：写代码→Review→改进→下次写得更好
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
#  三个核心操作的本质
# ================================================================

# 全局状态（就是Agent类里的实例变量）
history = []      # 对话历史
insights = ""     # 学习成果
learn_count = 0   # 学习次数


def respond(user_input: str) -> str:
    """
    对话的本质：call_llm，但prompt里多了一段insights。

    改进前: system + history + input
    改进后: system + insights + history + input
                      ↑ 这就是"自我改进"的全部秘密
    """
    global history

    messages = [
        {"role": "system", "content": "你是自我改进的AI助手。用中文回答。"},
    ]

    # 注入学习成果（改进的关键）
    if insights:
        messages.append({"role": "system", "content": f"改进洞察：{insights}"})
        print(f'    [注入洞察] {insights[:60]}...')

    # 加入对话历史
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_input})

    response = call_llm(messages)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    return response


def reflect() -> str:
    """
    反思的本质：让LLM审查自己之前的对话。

    就是一次普通的LLM调用，只不过：
    - 输入 = 之前的对话历史
    - 任务 = "分析哪些回答可以改进"
    - 输出 = 改进建议
    """
    print()
    print('    === 反思（就是一次LLM调用） ===')
    print(f'    输入: 最近 {min(len(history), 10)} 条对话')
    print(f'    任务: "分析AI回答质量，给改进建议"')

    messages = [
        {"role": "system", "content": "回顾对话历史，分析AI回答质量，给出改进建议。用中文。"},
    ]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": "请反思以上对话，生成改进洞察。"})

    result = call_llm(messages)
    print(f'    输出: {result[:100]}...')
    return result


def learn() -> str:
    """
    学习的本质：从反思结果中提取行为要点，存到全局变量。

    reflect() → 原始洞察（可能很长）
    learn()   → 精炼为3-5条要点 → 存到 insights 变量
    后续 respond() 把 insights 注入 prompt → 回答改进
    """
    global insights, learn_count

    # 先反思
    raw_insights = reflect()

    print()
    print('    === 学习（又一次LLM调用） ===')
    print(f'    输入: 反思洞察')
    print(f'    任务: "提取3-5条行为要点"')

    messages = [
        {"role": "system", "content": "从以下洞察中提取3-5条行为要点，简短精炼。用中文。"},
        {"role": "user", "content": raw_insights},
    ]

    learned = call_llm(messages)
    insights = learned  # 更新全局洞察
    learn_count += 1

    print(f'    输出: {learned}')
    print(f'    >>> insights 变量已更新，后续 respond() 会注入这些洞察')
    print(f'    >>> 累计学习 {learn_count} 次')
    return learned


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第24课 - 自我改进代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - respond = call_llm(system + insights + history + input)')
    print('  - reflect = call_llm("审查对话历史")')
    print('  - learn   = call_llm("提取行为要点") → 更新 insights 变量')
    print('  - 自我改进 = 把学习要点注入后续prompt的system部分')
    print()
    print('命令：/learn 触发学习，/status 查看状态，/quit 退出')
    print()

    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        if user_input == '/status':
            print(f'\n  对话: {len(history)//2}轮, 学习: {learn_count}次')
            print(f'  洞察: {insights[:100] if insights else "暂无"}\n')
            continue
        if user_input == '/learn':
            if len(history) < 4:
                print('>>> 先多聊几轮再学习')
                continue
            learn()
            continue

        response = respond(user_input)
        print(f'\nAI: {response}\n')
