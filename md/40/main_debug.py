"""
第40课：智能任务规划代理 Taskifier（透明调试版）

不使用任何框架，展示风格分析+知识检索+个性化方案的Prompt构建过程。
"""

import os
import re
import sys
import time
import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


def call_llm(messages: list, temperature: float = 0.5) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError("无choices")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    !! LLM失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# 模拟历史记录
HISTORY = [
    "数据分析项目：先收集所有数据再统一处理，用表格整理，偏好详细步骤清单。",
    "Web应用开发：敏捷开发，每天写一小部分，频繁测试，快速迭代。",
    "技术博客：先做大纲，逐节填充，最后统一润色。从整体到细节。",
    "团队活动：提前一月规划，甘特图跟踪进度，注重时间节点管理。"
]


def main():
    task = "我想开发一个基于AI的个人学习助手应用，能根据用户学习进度自动推荐学习内容，支持语音交互和知识图谱可视化。"

    print("=" * 70)
    print("  智能任务规划代理 Taskifier（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  任务: {task}")

    # ==============================================================
    # 阶段1：分析工作风格（对应 approach_analysis 节点）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段1】工作风格分析")
    print(f"{'=' * 70}")
    print(f"  原教程: 从 history/*.txt 文件读取历史记录")
    print(f"  本课: 用模拟数据")

    history_text = "\n".join(HISTORY)
    print(f"\n  历史记录:")
    for h in HISTORY:
        print(f"    - {h}")

    style_prompt = (
        "分析以下工作历史记录，总结用户的工作风格偏好。\n"
        "包括：工作节奏、组织方式、思维模式等。中文，100字以内。\n\n"
        f"工作历史:\n{history_text}"
    )

    print(f"\n  >>> 发送给 LLM <<<")
    print(f"  {'-' * 60}")
    print(f"  {style_prompt}")
    print(f"  {'-' * 60}")

    style = call_llm([{"role": "user", "content": style_prompt}])

    print(f"\n  >>> LLM 响应（工作风格）:")
    print(f"  {style}")

    # ==============================================================
    # 阶段2：检索任务知识（对应 task_manifest 节点）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段2】任务知识检索")
    print(f"{'=' * 70}")
    print(f"  原教程: Tavily 搜索 'What are the steps for [task]?'")
    print(f"  本课: LLM 模拟搜索结果")

    knowledge_prompt = (
        f"请提供完成以下任务所需的关键步骤和专业知识。\n"
        f"列出 5-8 个关键步骤。中文，每步一行。\n\n"
        f"任务: {task}"
    )

    print(f"\n  >>> 发送给 LLM <<<")
    print(f"  {'-' * 60}")
    print(f"  {knowledge_prompt}")
    print(f"  {'-' * 60}")

    details = call_llm([{"role": "user", "content": knowledge_prompt}])

    print(f"\n  >>> LLM 响应（任务知识）:")
    print(f"  {details}")

    # ==============================================================
    # 阶段3：个性化方案生成（对应 result_approach 节点）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段3】个性化方案生成")
    print(f"{'=' * 70}")
    print(f"  这是本课核心: 将工作风格注入到执行计划中")
    print(f"  同样的任务，不同风格的人会得到不同的方案")

    plan_prompt = (
        "请根据用户的工作风格偏好，为以下任务制定个性化执行计划。\n"
        "你必须特别关注用户的工作风格，调整步骤的顺序、粒度和方式。\n\n"
        f"任务: {task}\n\n"
        f"任务知识:\n{details}\n\n"
        f"用户工作风格:\n{style}\n\n"
        "输出: 编号列表，每步含做什么、为什么、如何契合风格。中文。"
    )

    print(f"\n  >>> 发送给 LLM（注意Prompt如何组合三个信息源）<<<")
    print(f"  {'-' * 60}")
    print(f"  任务: {task[:50]}...")
    print(f"  知识: {details[:80]}...")
    print(f"  风格: {style[:80]}...")
    print(f"  {'-' * 60}")

    plan = call_llm([{"role": "user", "content": plan_prompt}])

    print(f"\n  >>> LLM 响应（个性化计划）:")
    print(f"  {plan}")

    # ==============================================================
    # 最终结果
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【最终结果】")
    print(f"{'=' * 70}")
    print(f"\n任务: {task}")
    print(f"\n工作风格: {style}")
    print(f"\n个性化计划:\n{plan}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结")
    print(f"{'=' * 70}")
    print(f"  1. 三步线性流水线: 风格分析 → 知识检索 → 方案生成")
    print(f"  2. 核心创新: '个性化'——同样的任务，根据你的工作习惯定制方案")
    print(f"     - 习惯列清单的人 → 给详细步骤清单")
    print(f"     - 习惯迭代的人 → 给MVP优先的方案")
    print(f"     - 习惯整体规划的人 → 先给全局蓝图再细化")
    print(f"  3. 历史记录 = 用户画像数据源")
    print(f"     原教程从 txt 文件读取，实际场景可以从日历/项目管理工具获取")
    print(f"  4. Tavily 搜索 = 外部知识补充")
    print(f"     让计划不是凭空生成，而是基于真实的任务步骤")
    print(f"  5. 本质: 三次 LLM 调用，每次的输出作为下一次的输入")
    print(f"     最后一次调用把三个信息源（任务+知识+风格）合并生成方案")


if __name__ == "__main__":
    main()
