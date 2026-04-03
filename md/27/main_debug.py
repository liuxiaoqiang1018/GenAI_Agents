"""
第27课 - AutoGen研究团队群聊内部机制

目的：让你看清群聊模式的本质：
  1. 群聊 = 一个共享的 list[str]（对话线程），所有Agent都往里读写
  2. 发言权转移 = dict[current] → list[candidates]（有向图）
  3. Manager选择 = call_llm("从候选人里选谁？")
  4. Agent发言 = call_llm(system_prompt + 对话线程)
  5. 整个群聊 = while循环: 选人 → 发言 → 追加到线程

Java 类比：
  - 群聊线程 = ConcurrentLinkedQueue（共享消息队列）
  - 发言权转移 = 状态机的转移表 Map<State, Set<State>>
  - Manager = DispatcherServlet 根据规则分发
  - Agent = Controller，各有各的处理逻辑
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


# ================================================================
#  群聊的本质
# ================================================================

# Agent列表（就是不同的system prompt）
AGENTS = {
    "规划师": "你是规划师。制定计划、分解任务。以[规划师]开头。中文，100字以内。",
    "开发者": "你是开发者。编写代码、查找资料。以[开发者]开头。中文，100字以内。",
    "执行者": "你是执行者。执行方案、报告结果。以[执行者]开头。中文，100字以内。",
    "质检员": "你是质检员。审查质量，满意时说「任务完成」。以[质检员]开头。中文，100字以内。",
}

# 发言权转移图（就是一个dict，有向图的邻接表）
TRANSITIONS = {
    "管理员": ["规划师", "质检员"],
    "规划师": ["管理员", "开发者", "质检员"],
    "开发者": ["执行者", "质检员", "管理员"],
    "执行者": ["开发者"],
    "质检员": ["规划师", "开发者", "执行者", "管理员"],
}


def group_chat(task: str):
    """
    群聊的完整流程。

    Java 类比：
        public class GroupChatService {
            Map<String, Agent> agents;
            Map<String, Set<String>> transitions;

            public void chat(String task) {
                // 共享对话线程（就是一个 List）
                List<String> thread = new ArrayList<>();
                thread.add("[管理员] " + task);

                String current = "管理员";
                // 群聊循环
                while (round < MAX) {
                    // 1. Manager选下一个（LLM决策 + 转移规则约束）
                    String next = manager.select(current, transitions.get(current), thread);
                    // 2. Agent发言（LLM调用，传入整个线程）
                    String msg = agents.get(next).speak(thread);
                    // 3. 追加到线程
                    thread.add("[" + next + "] " + msg);
                    current = next;
                }
            }
        }
    """

    total_llm_calls = 0

    # 共享对话线程（就是一个list，所有人往里写）
    thread = [f"[管理员] {task}"]
    print(f'\n[管理员] {task}')

    current = "管理员"

    for round_num in range(1, 12):
        # ==========================================
        # 第1步：Manager选下一个发言者
        # ==========================================
        candidates = TRANSITIONS.get(current, list(AGENTS.keys()))

        recent = "\n".join(thread[-6:])
        choice = call_llm([
            {"role": "system", "content": f"从候选人{candidates}中选谁下一个发言。只回复名字。"},
            {"role": "user", "content": f"对话：\n{recent}"},
        ])
        total_llm_calls += 1

        next_speaker = None
        for c in candidates:
            if c in choice:
                next_speaker = c
                break
        if not next_speaker:
            next_speaker = candidates[0]

        print(f'\n    [Manager决策] {current} → 候选{candidates} → 选择: {next_speaker}')

        # ==========================================
        # 第2步：发言
        # ==========================================
        if next_speaker == "管理员":
            user_input = input('[管理员] ').strip() or "继续"
            if user_input in ('/quit', '结束'):
                break
            thread.append(f"[管理员] {user_input}")
        else:
            response = call_llm([
                {"role": "system", "content": AGENTS[next_speaker]},
                *[{"role": "user", "content": m} for m in thread[-8:]],
                {"role": "user", "content": "基于以上对话，做出回应。"},
            ])
            total_llm_calls += 1
            thread.append(f"[{next_speaker}] {response}")
            print(f'{response}')

            if "任务完成" in response and next_speaker == "质检员":
                print('\n>>> 任务完成！')
                break

        current = next_speaker

    print()
    print(f'>>> 群聊结束。总轮数: {round_num}, LLM调用: {total_llm_calls}次')
    print(f'>>> 群聊本质: while循环 + LLM选人 + LLM发言 + 共享list')


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第27课 - AutoGen研究团队群聊（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 群聊 = 共享的 list + while循环')
    print('  - Manager选人 = call_llm("从候选人里选谁")')
    print('  - Agent发言 = call_llm(角色prompt + 对话线程)')
    print('  - 转移规则 = dict[current] → [candidates]')
    print()

    task = input('研究任务（回车用默认）: ').strip()
    if not task:
        task = "请研究目前最流行的5个AI Agent开发框架，比较优缺点"
        print(f'>>> 使用: {task}')

    group_chat(task)
