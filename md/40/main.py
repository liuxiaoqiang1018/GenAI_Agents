"""
第40课：智能任务规划代理 Taskifier（LangGraph 框架版）

架构：风格分析 → 知识检索 → 个性化方案生成（三步线性流水线）
核心模式：用户工作风格分析 + 搜索任务知识 + 定制化执行计划
"""

import os
import re
import sys
import time
import httpx
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.5) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"无choices: {str(data)[:200]}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("LLM返回空内容")
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# ===== 状态定义 =====
class ApproachState(TypedDict):
    task: str        # 用户输入的任务
    style: str       # 分析出的工作风格
    history: str     # 历史工作记录
    details: str     # 任务相关知识（搜索结果）
    plan: str        # 最终的执行计划


# ===== 模拟历史工作记录 =====
MOCK_HISTORY = [
    "完成了一个数据分析项目：习惯先收集所有数据再统一处理，喜欢用表格整理信息，偏好详细的步骤清单。",
    "开发了一个Web应用：采用敏捷开发，每天写一小部分，频繁测试。偏好快速迭代而非一次性完成。",
    "写了一篇技术博客：先做大纲，然后逐节填充内容，最后统一润色。习惯从整体到细节。",
    "组织了一次团队活动：提前一个月开始规划，使用甘特图跟踪进度。注重时间节点和里程碑管理。"
]


# ===== 节点函数 =====

def approach_analysis(state: ApproachState) -> dict:
    """节点1：分析用户工作风格"""
    print("\n" + "=" * 60)
    print("【节点1】工作风格分析")
    print("=" * 60)

    history = "\n".join(MOCK_HISTORY)
    print(f"  历史记录（{len(MOCK_HISTORY)}条）:")
    for h in MOCK_HISTORY:
        print(f"    - {h[:50]}...")

    prompt = (
        "分析以下工作历史记录，总结用户的工作风格偏好。\n"
        "包括：工作节奏（一次性 vs 分步）、组织方式（清单 vs 自由）、思维模式（整体 vs 细节）等。\n"
        "用中文，简洁明了，100字以内。\n\n"
        f"工作历史:\n{history}"
    )

    style = call_llm([{"role": "user", "content": prompt}])
    print(f"\n  工作风格分析: {style}")

    return {**state, "history": history, "style": style}


def task_knowledge_retrieval(state: ApproachState) -> dict:
    """节点2：检索任务相关知识（模拟搜索）"""
    print("\n" + "=" * 60)
    print("【节点2】任务知识检索")
    print("=" * 60)

    # 用 LLM 模拟搜索结果（替代 Tavily）
    prompt = (
        f"请提供完成以下任务所需的关键步骤和专业知识。\n"
        f"像一个搜索引擎返回的摘要一样，列出 5-8 个关键步骤和注意事项。\n\n"
        f"任务: {state['task']}\n\n"
        f"用中文，每个步骤一行。"
    )

    details = call_llm([{"role": "user", "content": prompt}])
    print(f"  检索到的任务知识:\n{details[:300]}...")

    return {**state, "details": details}


def customized_approach_generation(state: ApproachState) -> dict:
    """节点3：结合风格+知识生成个性化方案"""
    print("\n" + "=" * 60)
    print("【节点3】个性化方案生成")
    print("=" * 60)

    prompt = (
        "请根据用户的工作风格偏好，为以下任务制定个性化执行计划。\n"
        "你必须特别关注用户的工作风格，调整计划的步骤顺序、粒度和方式。\n\n"
        f"任务: {state['task']}\n\n"
        f"任务知识:\n{state['details']}\n\n"
        f"用户工作风格:\n{state['style']}\n\n"
        "输出格式: 编号列表，每步包含：做什么、为什么这么安排、如何契合用户工作风格。\n"
        "用中文输出。"
    )

    plan = call_llm([{"role": "user", "content": prompt}])
    print(f"  个性化执行计划:\n{plan[:500]}...")

    return {**state, "plan": plan}


# ===== 构建工作流 =====
def build_workflow():
    workflow = StateGraph(ApproachState)

    workflow.add_node("approach_analysis", approach_analysis)
    workflow.add_node("task_knowledge_retrieval", task_knowledge_retrieval)
    workflow.add_node("customized_approach_generation", customized_approach_generation)

    workflow.set_entry_point("approach_analysis")
    workflow.add_edge("approach_analysis", "task_knowledge_retrieval")
    workflow.add_edge("task_knowledge_retrieval", "customized_approach_generation")
    workflow.add_edge("customized_approach_generation", END)

    return workflow.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  智能任务规划代理 Taskifier（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()

    task = "我想开发一个基于AI的个人学习助手应用，能根据用户的学习进度自动推荐学习内容，支持语音交互和知识图谱可视化。"

    print(f"  任务: {task}")

    result = app.invoke({
        "task": task,
        "style": "",
        "history": "",
        "details": "",
        "plan": ""
    })

    print(f"\n{'=' * 60}")
    print(f"【最终结果】")
    print(f"{'=' * 60}")
    print(f"\n任务:\n{result['task']}")
    print(f"\n工作风格:\n{result['style']}")
    print(f"\n个性化执行计划:\n{result['plan']}")


if __name__ == "__main__":
    main()
