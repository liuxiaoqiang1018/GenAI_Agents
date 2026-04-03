"""
第32课：记忆增强邮件代理（LangGraph 框架版）

架构：三层记忆（情景+语义+程序）驱动的邮件分类与回复系统
  - 情景记忆：历史邮件分类案例（Few-Shot）
  - 语义记忆：用户偏好和知识
  - 程序记忆：可更新的系统提示词

核心模式：三层记忆架构 + 邮件分类路由 + Prompt 自我优化
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, Literal, List, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages

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


# ===== 三层记忆存储（手动实现，替代 LangMem）=====
class MemoryStore:
    """简易记忆存储，模拟 LangGraph 的 InMemoryStore"""

    def __init__(self):
        self.store = {}  # {namespace_key: {item_key: value}}

    def put(self, namespace: tuple, key: str, value):
        ns_key = "/".join(namespace)
        if ns_key not in self.store:
            self.store[ns_key] = {}
        self.store[ns_key][key] = value

    def get(self, namespace: tuple, key: str):
        ns_key = "/".join(namespace)
        return self.store.get(ns_key, {}).get(key)

    def search(self, namespace: tuple, query: str = "") -> list:
        """简易搜索：返回该命名空间下的所有记忆"""
        ns_key = "/".join(namespace)
        items = self.store.get(ns_key, {})
        return [{"key": k, "value": v} for k, v in items.items()]

    def list_all(self) -> dict:
        return self.store


# ===== 全局记忆实例 =====
memory_store = MemoryStore()


# ===== 状态定义 =====
class State(TypedDict):
    email_input: dict       # 输入邮件
    triage_result: str      # 分类结果: ignore / notify / respond
    response: str           # 生成的回复
    user_id: str            # 用户ID


# ===== 初始化记忆 =====
def init_memory():
    """初始化三层记忆"""
    user_id = "test_user"

    # 程序记忆：系统提示词
    memory_store.put(
        ("email_assistant", user_id, "prompts"), "triage_prompt",
        "你是一个邮件分类助手。根据邮件内容将其分类为 ignore（忽略）、notify（仅通知）或 respond（需回复）。"
    )
    memory_store.put(
        ("email_assistant", user_id, "prompts"), "response_prompt",
        "你是一个专业的邮件回复助手。根据邮件内容生成合适的中文回复，语气友好专业。"
    )

    # 情景记忆：历史分类案例
    memory_store.put(
        ("email_assistant", user_id, "examples"), "spam_example",
        {
            "email": {
                "author": "垃圾广告 <spam@example.com>",
                "subject": "限时大促销!!!",
                "body": "立即购买我们的产品，享受五折优惠！"
            },
            "label": "ignore"
        }
    )
    memory_store.put(
        ("email_assistant", user_id, "examples"), "meeting_example",
        {
            "email": {
                "author": "张经理 <zhang@company.com>",
                "subject": "明天会议通知",
                "body": "提醒您明天上午10点有项目评审会议。"
            },
            "label": "notify"
        }
    )
    memory_store.put(
        ("email_assistant", user_id, "examples"), "question_example",
        {
            "email": {
                "author": "李工 <li@company.com>",
                "subject": "API接口文档问题",
                "body": "你好，我在查看API文档时发现几个接口缺失，能帮忙补充吗？"
            },
            "label": "respond"
        }
    )

    # 语义记忆：用户偏好
    memory_store.put(
        ("email_assistant", user_id, "preferences"), "style",
        "用户偏好简洁专业的回复风格，喜欢列出具体行动步骤。"
    )
    memory_store.put(
        ("email_assistant", user_id, "preferences"), "priority",
        "API文档相关的邮件视为高优先级，必须回复。"
    )


# ===== 辅助函数 =====
def format_few_shot_examples(examples: list) -> str:
    """将情景记忆格式化为 Few-Shot 示例"""
    formatted = []
    for eg in examples:
        email = eg["value"]["email"]
        label = eg["value"]["label"]
        formatted.append(
            f"发件人: {email['author']}\n"
            f"主题: {email['subject']}\n"
            f"内容: {email['body'][:200]}\n"
            f"分类: {label}"
        )
    return "\n\n".join(formatted)


# ===== 节点函数 =====

def triage_email(state: State) -> dict:
    """邮件分类节点（情景记忆 + 程序记忆辅助）"""
    print("\n" + "=" * 60)
    print("【邮件分类】情景记忆 + 程序记忆辅助决策")
    print("=" * 60)

    email = state["email_input"]
    user_id = state["user_id"]

    # 读取程序记忆（系统提示词）
    triage_prompt = memory_store.get(("email_assistant", user_id, "prompts"), "triage_prompt")
    print(f"  程序记忆（提示词）: {triage_prompt[:80]}...")

    # 读取情景记忆（历史案例）
    examples = memory_store.search(("email_assistant", user_id, "examples"))
    formatted_examples = format_few_shot_examples(examples)
    print(f"  情景记忆（{len(examples)}条历史案例）:")
    for eg in examples:
        print(f"    - {eg['key']}: {eg['value']['label']}")

    # 构建分类 Prompt
    prompt = (
        f"{triage_prompt}\n\n"
        f"以下是历史分类案例供参考:\n{formatted_examples}\n\n"
        f"请对以下邮件进行分类:\n"
        f"发件人: {email.get('author', '')}\n"
        f"收件人: {email.get('to', '')}\n"
        f"主题: {email.get('subject', '')}\n"
        f"内容: {email.get('body', '')}\n\n"
        f"只回复 ignore、notify 或 respond 中的一个。"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0)
    classification = result.strip().lower()

    # 规范化
    if "respond" in classification:
        classification = "respond"
    elif "notify" in classification:
        classification = "notify"
    else:
        classification = "ignore"

    print(f"  分类结果: {classification}")
    return {**state, "triage_result": classification}


def respond_to_email(state: State) -> dict:
    """邮件回复节点（语义记忆辅助）"""
    print("\n" + "=" * 60)
    print("【邮件回复】语义记忆辅助生成回复")
    print("=" * 60)

    email = state["email_input"]
    user_id = state["user_id"]

    # 读取程序记忆（回复提示词）
    response_prompt = memory_store.get(("email_assistant", user_id, "prompts"), "response_prompt")
    print(f"  程序记忆（回复提示词）: {response_prompt[:80]}...")

    # 读取语义记忆（用户偏好）
    preferences = memory_store.search(("email_assistant", user_id, "preferences"))
    pref_text = "\n".join([f"- {p['value']}" for p in preferences])
    print(f"  语义记忆（用户偏好）:")
    for p in preferences:
        print(f"    - {p['key']}: {p['value']}")

    # 构建回复 Prompt
    prompt = (
        f"{response_prompt}\n\n"
        f"用户偏好:\n{pref_text}\n\n"
        f"请为以下邮件生成回复:\n"
        f"发件人: {email.get('author', '')}\n"
        f"主题: {email.get('subject', '')}\n"
        f"内容: {email.get('body', '')}\n\n"
        f"请用中文生成专业、友好的回复邮件内容。"
    )

    response = call_llm([{"role": "user", "content": prompt}])
    print(f"  生成的回复: {response[:200]}...")

    return {**state, "response": response}


def handle_ignore_or_notify(state: State) -> dict:
    """处理忽略/通知类邮件"""
    print("\n" + "=" * 60)
    print(f"【处理结果】邮件分类为: {state['triage_result']}")
    print("=" * 60)

    if state["triage_result"] == "ignore":
        msg = f"邮件已忽略: {state['email_input'].get('subject', '')}"
    else:
        msg = f"邮件已标记通知: {state['email_input'].get('subject', '')}"

    print(f"  {msg}")
    return {**state, "response": msg}


# ===== 路由 =====
def route_based_on_triage(state: State) -> str:
    if state["triage_result"] == "respond":
        print("  [路由] 需要回复 → 进入回复生成")
        return "respond"
    else:
        print("  [路由] 忽略/通知 → 简单处理")
        return "handle"


# ===== Prompt 优化（程序记忆更新）=====
def optimize_prompts(feedback: str, user_id: str):
    """根据反馈优化提示词（程序记忆更新）"""
    print("\n" + "=" * 60)
    print("【程序记忆更新】根据反馈优化提示词")
    print("=" * 60)

    old_triage = memory_store.get(("email_assistant", user_id, "prompts"), "triage_prompt")
    old_response = memory_store.get(("email_assistant", user_id, "prompts"), "response_prompt")

    prompt = (
        f"你是一个 Prompt 优化专家。根据用户反馈改进以下两个提示词。\n\n"
        f"用户反馈: {feedback}\n\n"
        f"当前分类提示词:\n{old_triage}\n\n"
        f"当前回复提示词:\n{old_response}\n\n"
        f"请分别输出改进后的两个提示词，用 === 分隔。\n"
        f"格式:\n[分类提示词]\n===\n[回复提示词]"
    )

    result = call_llm([{"role": "user", "content": prompt}], temperature=0.3)

    parts = result.split("===")
    if len(parts) >= 2:
        new_triage = parts[0].strip()
        new_response = parts[1].strip()
        memory_store.put(("email_assistant", user_id, "prompts"), "triage_prompt", new_triage)
        memory_store.put(("email_assistant", user_id, "prompts"), "response_prompt", new_response)
        print(f"  分类提示词已更新: {new_triage[:80]}...")
        print(f"  回复提示词已更新: {new_response[:80]}...")
    else:
        print(f"  优化结果格式不符预期，保持原提示词")


# ===== 构建工作流 =====
def build_workflow():
    builder = StateGraph(State)

    builder.add_node("triage", triage_email)
    builder.add_node("respond", respond_to_email)
    builder.add_node("handle", handle_ignore_or_notify)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route_based_on_triage)
    builder.add_edge("respond", END)
    builder.add_edge("handle", END)

    return builder.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  记忆增强邮件代理（LangGraph 框架版）")
    print("=" * 60)

    # 初始化三层记忆
    init_memory()
    print("  三层记忆已初始化")

    graph = build_workflow()

    # 测试邮件
    test_emails = [
        {
            "author": "营销推广 <promo@shop.com>",
            "to": "张三 <zhangsan@company.com>",
            "subject": "双十一特惠，全场五折!!!",
            "body": "亲爱的用户，双十一大促来袭，全场商品五折起！立即抢购！"
        },
        {
            "author": "王主管 <wang@company.com>",
            "to": "张三 <zhangsan@company.com>",
            "subject": "明天下午团队周会",
            "body": "提醒你明天下午3点参加团队周会，会议室B301。"
        },
        {
            "author": "李工 <li@company.com>",
            "to": "张三 <zhangsan@company.com>",
            "subject": "API接口文档缺失问题",
            "body": "你好张三，我在查看API文档时发现用户认证模块的几个接口缺少参数说明，能帮忙补充一下吗？这影响了前端的对接工作。谢谢！"
        }
    ]

    for i, email in enumerate(test_emails, 1):
        print(f"\n{'*' * 60}")
        print(f"  测试邮件 {i}: {email['subject']}")
        print(f"{'*' * 60}")

        state = {
            "email_input": email,
            "triage_result": "",
            "response": "",
            "user_id": "test_user"
        }

        result = graph.invoke(state)
        print(f"\n  最终结果: {result['response'][:300]}")

    # 演示程序记忆更新（Prompt 优化）
    print(f"\n{'*' * 60}")
    print(f"  演示：程序记忆更新（Prompt 自我优化）")
    print(f"{'*' * 60}")

    feedback = "API文档相关的邮件必须被视为高优先级并立即回复。回复时应该具体提到对方提到的问题，而不是泛泛而谈。"
    optimize_prompts(feedback, "test_user")

    # 用更新后的提示词重新处理第3封邮件
    print(f"\n{'*' * 60}")
    print(f"  用更新后的提示词重新处理 API 文档邮件")
    print(f"{'*' * 60}")

    state = {
        "email_input": test_emails[2],
        "triage_result": "",
        "response": "",
        "user_id": "test_user"
    }
    result = graph.invoke(state)
    print(f"\n  优化后的回复: {result['response'][:300]}")

    # 展示记忆内容
    print(f"\n{'=' * 60}")
    print(f"  当前记忆存储内容")
    print(f"{'=' * 60}")
    for ns, items in memory_store.list_all().items():
        print(f"\n  命名空间: {ns}")
        for k, v in items.items():
            v_str = str(v)[:100]
            print(f"    {k}: {v_str}...")


if __name__ == "__main__":
    main()
