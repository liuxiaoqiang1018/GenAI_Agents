"""
第36课：博客写作 Swarm 代理（模拟 Swarm 接力版）

架构：5个 Agent 接力写博客
  管理员 → 规划者 → 研究者 → 写手 → 编辑 → 保存文件

核心模式：Agent 接力传递（Handoff）+ 5 角色分工
使用 httpx 手写 Swarm 的接力机制，无需安装 openai-swarm
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
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 简易 Swarm Agent 模拟 =====
class SwarmAgent:
    """模拟 OpenAI Swarm 的 Agent 类"""

    def __init__(self, name: str, instructions: str, next_agent=None):
        self.name = name
        self.instructions = instructions
        self.next_agent = next_agent  # 接力的下一个 Agent

    def run(self, context: dict) -> str:
        """执行 Agent 任务，返回结果"""
        print(f"\n{'=' * 60}")
        print(f"【{self.name}】开始工作")
        print(f"{'=' * 60}")
        print(f"  指令: {self.instructions[:100]}...")

        # 构建消息
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": context.get("input", "")}
        ]

        result = call_llm(messages)
        print(f"  输出: {result[:300]}...")

        return result

    def handoff(self):
        """接力给下一个 Agent（Swarm 的核心机制）"""
        if self.next_agent:
            print(f"\n  → 接力传递（Handoff）: {self.name} → {self.next_agent.name}")
            return self.next_agent
        return None


# ===== 创建 5 个 Agent =====
def create_agents(topic: str):
    """创建接力链: 管理员 → 规划者 → 研究者 → 写手 → 编辑"""

    editor = SwarmAgent(
        name="编辑 Agent",
        instructions=(
            "你是编辑，负责审阅和润色博客文章。\n"
            "检查语法、表达、逻辑和结构，进行必要的修正和改进。\n"
            "输出最终定稿版本的完整博客文章。用中文输出。"
        )
    )

    writer = SwarmAgent(
        name="写手 Agent",
        instructions=(
            "你是写手，负责撰写完整的博客文章。\n"
            "根据规划者的大纲和研究者的调研内容，写出一篇结构清晰、内容丰富的博客。\n"
            "文章应该有吸引力，面向普通读者。用中文输出，字数 800-1200 字。"
        ),
        next_agent=editor
    )

    researcher = SwarmAgent(
        name="研究者 Agent",
        instructions=(
            f"你是研究者，负责对博客主题'{topic}'的各个章节进行详细调研。\n"
            "为每个章节提供详实的背景信息、数据、案例和关键观点。\n"
            "输出格式：按章节列出调研笔记。用中文输出。"
        ),
        next_agent=writer
    )

    planner = SwarmAgent(
        name="规划者 Agent",
        instructions=(
            f"你是规划者，负责为博客主题'{topic}'拟定大纲。\n"
            "将内容组织成 4-6 个章节，每个章节有清晰的标题和简短描述。\n"
            "输出格式：编号列表，每项包含章节标题和一句话描述。用中文输出。"
        ),
        next_agent=researcher
    )

    admin = SwarmAgent(
        name="管理员 Agent",
        instructions=(
            f"你是管理员，负责启动博客写作项目。\n"
            f"博客主题: {topic}\n"
            "确认主题后，简要说明项目目标和期望，然后交给规划者。\n"
            "用中文输出，简洁明了，50字以内。"
        ),
        next_agent=planner
    )

    return admin


# ===== 运行 Swarm 接力 =====
def run_swarm(topic: str):
    """运行 Swarm 接力流程"""
    print("=" * 60)
    print("  博客写作 Swarm 代理")
    print("=" * 60)
    print(f"  主题: {topic}")

    # 创建接力链
    current_agent = create_agents(topic)

    # 保存每步结果，作为下一步的输入
    context = {"input": f"博客主题: {topic}"}
    all_outputs = {}

    # 接力执行
    while current_agent:
        result = current_agent.run(context)
        all_outputs[current_agent.name] = result

        # 把当前输出作为下一个 Agent 的输入
        context["input"] = f"前序工作成果:\n{result}"

        # 接力
        current_agent = current_agent.handoff()

    # 保存最终文章
    final_content = all_outputs.get("编辑 Agent", "")
    filename = topic.replace(" ", "-") + ".md"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{final_content}")
    print(f"\n  博客已保存: {filepath}")

    # 打印最终文章
    print(f"\n{'=' * 60}")
    print(f"【最终博客文章】")
    print(f"{'=' * 60}")
    print(final_content)

    return all_outputs


# ===== 主函数 =====
def main():
    topic = "大语言模型如何改变软件开发"
    run_swarm(topic)


if __name__ == "__main__":
    main()
