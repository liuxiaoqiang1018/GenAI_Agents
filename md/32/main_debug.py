"""
第32课：记忆增强邮件代理（透明调试版）

不使用任何框架，纯手写三层记忆 + 邮件处理流程。
让你看清情景记忆(Few-Shot)、语义记忆(偏好)、程序记忆(Prompt更新)的内部机制。
"""

import os
import re
import sys
import json
import time
import httpx
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


# ===== 三层记忆存储 =====
# 用最简单的 dict 实现，让你看清本质就是一个键值存储

# 情景记忆：历史分类案例
episodic_memory = {}

# 语义记忆：用户偏好和知识
semantic_memory = {}

# 程序记忆：系统提示词（可更新）
procedural_memory = {}


def init_memory():
    """初始化三层记忆"""
    global episodic_memory, semantic_memory, procedural_memory

    # 情景记忆
    episodic_memory = {
        "spam_example": {
            "email": {"author": "垃圾广告 <spam@example.com>", "subject": "限时大促销!!!", "body": "立即购买我们的产品，享受五折优惠！"},
            "label": "ignore"
        },
        "meeting_example": {
            "email": {"author": "张经理 <zhang@company.com>", "subject": "明天会议通知", "body": "提醒您明天上午10点有项目评审会议。"},
            "label": "notify"
        },
        "question_example": {
            "email": {"author": "李工 <li@company.com>", "subject": "API接口文档问题", "body": "你好，我在查看API文档时发现几个接口缺失，能帮忙补充吗？"},
            "label": "respond"
        }
    }

    # 语义记忆
    semantic_memory = {
        "style": "用户偏好简洁专业的回复风格，喜欢列出具体行动步骤。",
        "priority": "API文档相关的邮件视为高优先级，必须回复。"
    }

    # 程序记忆
    procedural_memory = {
        "triage_prompt": "你是一个邮件分类助手。根据邮件内容将其分类为 ignore（忽略）、notify（仅通知）或 respond（需回复）。",
        "response_prompt": "你是一个专业的邮件回复助手。根据邮件内容生成合适的中文回复，语气友好专业。"
    }


def process_email(email: dict, round_num: int):
    """处理单封邮件的完整流程"""

    print(f"\n{'*' * 70}")
    print(f"  第 {round_num} 封邮件: {email['subject']}")
    print(f"{'*' * 70}")

    # ==============================================================
    # 阶段1：邮件分类（情景记忆 + 程序记忆）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段1】邮件分类 — 情景记忆(Few-Shot) + 程序记忆(Prompt)")
    print(f"{'=' * 70}")

    # 读取程序记忆
    triage_prompt = procedural_memory["triage_prompt"]
    print(f"\n  [程序记忆] 分类提示词:")
    print(f"    {triage_prompt}")

    # 读取情景记忆，格式化为 Few-Shot 示例
    print(f"\n  [情景记忆] 历史分类案例 ({len(episodic_memory)}条):")
    few_shot_lines = []
    for key, eg in episodic_memory.items():
        e = eg["email"]
        label = eg["label"]
        print(f"    {key}: {e['subject']} → {label}")
        few_shot_lines.append(
            f"发件人: {e['author']}\n主题: {e['subject']}\n内容: {e['body'][:200]}\n分类: {label}"
        )
    few_shot_text = "\n\n".join(few_shot_lines)

    # 构建完整 Prompt
    full_prompt = (
        f"{triage_prompt}\n\n"
        f"以下是历史分类案例供参考:\n{few_shot_text}\n\n"
        f"请对以下邮件进行分类:\n"
        f"发件人: {email.get('author', '')}\n"
        f"收件人: {email.get('to', '')}\n"
        f"主题: {email.get('subject', '')}\n"
        f"内容: {email.get('body', '')}\n\n"
        f"只回复 ignore、notify 或 respond 中的一个。"
    )

    print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
    print(f"  {'-' * 60}")
    print(f"  {full_prompt}")
    print(f"  {'-' * 60}")

    result = call_llm([{"role": "user", "content": full_prompt}], temperature=0)

    # 规范化分类结果
    classification = result.strip().lower()
    if "respond" in classification:
        classification = "respond"
    elif "notify" in classification:
        classification = "notify"
    else:
        classification = "ignore"

    print(f"\n  >>> LLM 响应: {result}")
    print(f"  >>> 规范化分类: {classification}")

    # ==============================================================
    # 阶段2：路由决策
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【路由决策】根据分类结果路由")
    print(f"{'=' * 70}")
    print(f"  classification = {classification}")
    print(f"  规则: respond → 生成回复, ignore/notify → 简单处理")

    if classification != "respond":
        if classification == "ignore":
            print(f"  结果: → 邮件已忽略")
        else:
            print(f"  结果: → 邮件已标记通知")
        return

    print(f"  结果: → 进入回复生成")

    # ==============================================================
    # 阶段3：生成回复（语义记忆 + 程序记忆）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段3】生成回复 — 语义记忆(偏好) + 程序记忆(Prompt)")
    print(f"{'=' * 70}")

    # 读取程序记忆
    response_prompt = procedural_memory["response_prompt"]
    print(f"\n  [程序记忆] 回复提示词:")
    print(f"    {response_prompt}")

    # 读取语义记忆
    print(f"\n  [语义记忆] 用户偏好:")
    pref_lines = []
    for key, val in semantic_memory.items():
        print(f"    {key}: {val}")
        pref_lines.append(f"- {val}")
    pref_text = "\n".join(pref_lines)

    # 构建回复 Prompt
    reply_prompt = (
        f"{response_prompt}\n\n"
        f"用户偏好:\n{pref_text}\n\n"
        f"请为以下邮件生成回复:\n"
        f"发件人: {email.get('author', '')}\n"
        f"主题: {email.get('subject', '')}\n"
        f"内容: {email.get('body', '')}\n\n"
        f"请用中文生成专业、友好的回复邮件内容。"
    )

    print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
    print(f"  {'-' * 60}")
    print(f"  {reply_prompt}")
    print(f"  {'-' * 60}")

    response = call_llm([{"role": "user", "content": reply_prompt}])

    print(f"\n  >>> LLM 响应（生成的回复）<<<")
    print(f"  {'-' * 60}")
    print(f"  {response}")
    print(f"  {'-' * 60}")


def main():
    print("=" * 70)
    print("  记忆增强邮件代理（透明调试版 - 无框架）")
    print("=" * 70)

    # 初始化三层记忆
    init_memory()

    print(f"\n  三层记忆已初始化:")
    print(f"    情景记忆: {len(episodic_memory)} 条历史案例")
    print(f"    语义记忆: {len(semantic_memory)} 条用户偏好")
    print(f"    程序记忆: {len(procedural_memory)} 条提示词")

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
        process_email(email, i)

    # ==============================================================
    # 阶段4：演示程序记忆更新（Prompt 自我优化）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【阶段4】程序记忆更新 — Prompt 自我优化")
    print(f"{'=' * 70}")
    print(f"  这是本课最关键的创新：Agent 可以根据反馈更新自己的提示词")
    print(f"  在 LangGraph 原版中，这通过 LangMem 的 create_multi_prompt_optimizer 实现")
    print(f"  本质就是：用 LLM 改写 Prompt → 存回程序记忆 → 下次使用新 Prompt")

    feedback = "API文档相关的邮件必须被视为高优先级并立即回复。回复时应该具体提到对方提到的问题，而不是泛泛而谈。"
    print(f"\n  用户反馈: {feedback}")

    old_triage = procedural_memory["triage_prompt"]
    old_response = procedural_memory["response_prompt"]

    optimize_prompt = (
        f"你是一个 Prompt 优化专家。根据用户反馈改进以下两个提示词。\n\n"
        f"用户反馈: {feedback}\n\n"
        f"当前分类提示词:\n{old_triage}\n\n"
        f"当前回复提示词:\n{old_response}\n\n"
        f"请分别输出改进后的两个提示词，用 === 分隔。\n"
        f"格式:\n[分类提示词]\n===\n[回复提示词]"
    )

    print(f"\n  >>> 调用 LLM 优化提示词...")
    result = call_llm([{"role": "user", "content": optimize_prompt}], temperature=0.3)

    parts = result.split("===")
    if len(parts) >= 2:
        procedural_memory["triage_prompt"] = parts[0].strip()
        procedural_memory["response_prompt"] = parts[1].strip()
        print(f"\n  程序记忆已更新!")
        print(f"  新分类提示词: {procedural_memory['triage_prompt'][:100]}...")
        print(f"  新回复提示词: {procedural_memory['response_prompt'][:100]}...")
    else:
        print(f"  优化结果格式不符预期")

    # 用更新后的提示词重新处理第3封邮件
    print(f"\n{'*' * 70}")
    print(f"  用更新后的程序记忆重新处理 API 文档邮件")
    print(f"{'*' * 70}")
    process_email(test_emails[2], 4)

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：LangGraph + LangMem 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. 三层记忆本质都是键值存储（InMemoryStore），按 namespace 隔离:")
    print(f"     - 情景记忆: (user_id, 'examples') → 历史案例，拼进 Prompt 做 Few-Shot")
    print(f"     - 语义记忆: (user_id, 'collection') → 用户偏好，通过工具读写")
    print(f"     - 程序记忆: (user_id, 'prompts') → 系统提示词，可被优化器更新")
    print(f"  2. LangMem 提供了 create_manage_memory_tool / create_search_memory_tool")
    print(f"     让 Agent 可以自主决定何时读写记忆（本课简化为直接读取）")
    print(f"  3. create_multi_prompt_optimizer 是程序记忆的核心:")
    print(f"     用 LLM 分析反馈 → 改写 Prompt → 存回 Store → 下次使用新版")
    print(f"  4. 本质: dict 存储 + Prompt 拼接 + LLM 调用")
    print(f"     三层记忆只是给不同用途的 Prompt 片段起了不同的名字")


if __name__ == "__main__":
    main()
