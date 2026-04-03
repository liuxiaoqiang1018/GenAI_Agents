"""
第36课：博客写作 Swarm 代理（透明调试版）

不使用任何框架，纯手写 5 Agent 接力流程。
让你看清 OpenAI Swarm 的 Handoff 接力机制的内部本质。
"""

import os
import re
import sys
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            return content
        except Exception as e:
            print(f"    !! LLM调用失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


def main():
    topic = "大语言模型如何改变软件开发"

    print("=" * 70)
    print("  博客写作 Swarm 代理（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  主题: {topic}")
    print(f"\n  Swarm 的本质: 5 个 LLM 调用串联，每个用不同的 system prompt")
    print(f"  Handoff 的本质: 把上一个 Agent 的输出作为下一个的输入")

    # ==============================================================
    # Agent 1: 管理员 — 确认主题
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Agent 1】管理员 — 确认主题并启动项目")
    print(f"{'=' * 70}")

    admin_prompt = (
        f"你是管理员，负责启动博客写作项目。\n"
        f"博客主题: {topic}\n"
        "确认主题后，简要说明项目目标和期望。用中文输出，50字以内。"
    )

    print(f"\n  >>> system prompt <<<")
    print(f"  {admin_prompt}")

    admin_result = call_llm([
        {"role": "system", "content": admin_prompt},
        {"role": "user", "content": f"博客主题: {topic}"}
    ])

    print(f"\n  >>> 管理员输出: {admin_result}")
    print(f"\n  → Handoff: 管理员 → 规划者")
    print(f"    (把管理员的输出传给规划者作为输入)")

    # ==============================================================
    # Agent 2: 规划者 — 拟定大纲
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Agent 2】规划者 — 拟定博客大纲")
    print(f"{'=' * 70}")

    planner_prompt = (
        f"你是规划者，负责为博客主题'{topic}'拟定大纲。\n"
        "将内容组织成 4-6 个章节，每个章节有清晰的标题和简短描述。\n"
        "输出格式：编号列表，每项包含章节标题和一句话描述。用中文输出。"
    )

    print(f"\n  >>> system prompt: {planner_prompt[:80]}...")

    planner_result = call_llm([
        {"role": "system", "content": planner_prompt},
        {"role": "user", "content": f"前序工作成果:\n{admin_result}"}
    ])

    print(f"\n  >>> 规划者输出:\n  {planner_result}")
    print(f"\n  → Handoff: 规划者 → 研究者")

    # ==============================================================
    # Agent 3: 研究者 — 调研内容
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Agent 3】研究者 — 对每个章节进行调研")
    print(f"{'=' * 70}")

    researcher_prompt = (
        f"你是研究者，负责对博客主题'{topic}'的各个章节进行详细调研。\n"
        "为每个章节提供详实的背景信息、数据、案例和关键观点。\n"
        "输出格式：按章节列出调研笔记。用中文输出。"
    )

    print(f"\n  >>> system prompt: {researcher_prompt[:80]}...")

    researcher_result = call_llm([
        {"role": "system", "content": researcher_prompt},
        {"role": "user", "content": f"前序工作成果（大纲）:\n{planner_result}"}
    ])

    print(f"\n  >>> 研究者输出:\n  {researcher_result[:500]}...")
    print(f"\n  → Handoff: 研究者 → 写手")

    # ==============================================================
    # Agent 4: 写手 — 撰写文章
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Agent 4】写手 — 撰写完整博客文章")
    print(f"{'=' * 70}")

    writer_prompt = (
        "你是写手，负责撰写完整的博客文章。\n"
        "根据规划者的大纲和研究者的调研内容，写出一篇结构清晰、内容丰富的博客。\n"
        "文章应该有吸引力，面向普通读者。用中文输出，字数 800-1200 字。"
    )

    print(f"\n  >>> system prompt: {writer_prompt[:80]}...")

    writer_result = call_llm([
        {"role": "system", "content": writer_prompt},
        {"role": "user", "content": f"大纲:\n{planner_result}\n\n调研内容:\n{researcher_result}"}
    ])

    print(f"\n  >>> 写手输出:\n  {writer_result[:500]}...")
    print(f"\n  → Handoff: 写手 → 编辑")

    # ==============================================================
    # Agent 5: 编辑 — 润色定稿
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Agent 5】编辑 — 审阅润色定稿")
    print(f"{'=' * 70}")

    editor_prompt = (
        "你是编辑，负责审阅和润色博客文章。\n"
        "检查语法、表达、逻辑和结构，进行必要的修正和改进。\n"
        "输出最终定稿版本的完整博客文章。用中文输出。"
    )

    print(f"\n  >>> system prompt: {editor_prompt[:80]}...")

    editor_result = call_llm([
        {"role": "system", "content": editor_prompt},
        {"role": "user", "content": f"请审阅并润色以下博客文章:\n\n{writer_result}"}
    ])

    print(f"\n  >>> 编辑输出（最终定稿）:\n  {editor_result[:500]}...")

    # ==============================================================
    # 保存文件
    # ==============================================================
    filename = topic.replace(" ", "-") + "_debug.md"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{editor_result}")
    print(f"\n  博客已保存: {filepath}")

    # ==============================================================
    # 最终文章
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【最终博客文章】")
    print(f"{'=' * 70}")
    print(editor_result)

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：OpenAI Swarm 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. Swarm 的 Agent 类: name + instructions(system prompt) + functions(工具/转接)")
    print(f"  2. Handoff（接力）: transfer_to_xxx() 函数返回下一个 Agent 对象")
    print(f"     LLM 自主决定何时调用 transfer 函数 → 控制权转交")
    print(f"  3. context_variables: 跨 Agent 共享的 dict（类似 LangGraph 的 State）")
    print(f"  4. run_demo_loop: 交互式循环，用户输入 → Agent 处理 → 可能 Handoff")
    print(f"  5. 本质: 5 次 LLM 调用，每次用不同的 system prompt")
    print(f"     上一个 Agent 的输出 = 下一个 Agent 的输入")
    print(f"     Swarm 和 LangGraph 的区别:")
    print(f"     - Swarm: Agent 自己决定何时转交（LLM 调 function）")
    print(f"     - LangGraph: 图结构预定义好了转交规则（边和路由）")
    print(f"     - Swarm 更适合对话式、线性接力")
    print(f"     - LangGraph 更适合复杂分支、循环、并行")


if __name__ == "__main__":
    main()
