"""
第9课 - 智能客服路由系统内部机制（不使用 LangGraph）

目的：让你看清客服路由的本质：
  1. 分类 = 一次 LLM 调用
  2. 情感分析 = 一次 LLM 调用
  3. 路由 = if-elif-else（优先级判断）
  4. 处理 = 一次 LLM 调用（或固定文案）
  5. 升级 = 不调 LLM，直接返回固定文案

对比 main.py（LangGraph 框架版），理解：
  - add_conditional_edges → 就是 if sentiment=="消极": return "escalate"
  - 4个处理节点 → 就是 4 个 if 分支里的函数调用
  - escalate 不调 LLM → 就是直接 return 固定字符串
"""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ================================================================
#  完整流程
# ================================================================

def handle_customer_query(query: str) -> dict:
    """
    客服路由系统的完整流程。

    Java 类比：
        @Service
        public class CustomerSupportService {
            public SupportResult handle(String query) {
                // 1. 分类
                String category = classifyService.classify(query);
                // 2. 情感分析
                String sentiment = sentimentService.analyze(query);
                // 3. 路由 + 处理
                if ("消极".equals(sentiment)) {
                    return escalate(query, category, sentiment);
                }
                switch (category) {
                    case "技术": return techSupport.handle(query);
                    case "账单": return billingSupport.handle(query);
                    default: return generalSupport.handle(query);
                }
            }
        }
    """

    state = {"query": query, "category": "", "sentiment": "", "response": ""}

    # ==========================================
    # 第1步：分类
    # ==========================================
    print()
    print('=' * 60)
    print('【1 - 分类】')
    print('=' * 60)

    category = call_llm(
        query,
        "将用户工单分类为：技术、账单、一般。只返回分类名称。"
    )

    # 标准化
    for c in ["技术", "账单", "一般"]:
        if c in category:
            category = c
            break
    else:
        category = "一般"

    state["category"] = category
    print(f'>>> 工单: {query}')
    print(f'>>> 分类结果: {category}')

    # ==========================================
    # 第2步：情感分析
    # ==========================================
    print()
    print('=' * 60)
    print('【2 - 情感分析】')
    print('=' * 60)

    sentiment = call_llm(
        query,
        ("分析用户工单的情绪态度（不是事情本身好坏）。\n"
         "- 积极：表达感谢、满意\n"
         "- 中性：正常描述问题、寻求帮助\n"
         "- 消极：愤怒、抱怨、威胁、投诉\n"
         "只回复：积极、中性、消极（三选一）。")
    )

    for s in ["积极", "消极", "中性"]:
        if s in sentiment:
            sentiment = s
            break
    else:
        sentiment = "中性"

    state["sentiment"] = sentiment
    print(f'>>> 情感结果: {sentiment}')

    # ==========================================
    # 第3步：路由 — 这就是 add_conditional_edges 的本质
    # ==========================================
    print()
    print('=' * 60)
    print('【3 - 路由决策】')
    print('=' * 60)
    print(f'>>> 分类={category}, 情感={sentiment}')
    print()
    print('    路由逻辑（就是 if-elif-else）:')
    print('    if sentiment == "消极": → 升级人工（优先级最高）')
    print('    elif category == "技术": → 技术客服')
    print('    elif category == "账单": → 账单客服')
    print('    else: → 一般客服')
    print()

    # ==========================================
    # 第4步：处理
    # ==========================================

    if sentiment == "消极":
        # 升级 — 不调用 LLM
        print('=' * 60)
        print('【4 - 升级人工】（不调用 LLM！）')
        print('=' * 60)
        print('>>> 消极情绪 → 直接转人工，避免 AI 说错话')

        response = ("非常抱歉给您带来不好的体验。您的问题已被标记为紧急工单，"
                    "我们的资深客服专员将在15分钟内与您联系。"
                    f"\n\n工单摘要：{query[:100]}"
                    f"\n分类：{category}\n情感：{sentiment}")
    elif category == "技术":
        print('=' * 60)
        print('【4 - 技术客服】（调用 LLM）')
        print('=' * 60)
        response = call_llm(query, "你是技术客服专家。用专业但友好的语气回复。用中文回答。")
    elif category == "账单":
        print('=' * 60)
        print('【4 - 账单客服】（调用 LLM）')
        print('=' * 60)
        response = call_llm(query, "你是账单客服专家。用耐心、清晰的语气回复。用中文回答。")
    else:
        print('=' * 60)
        print('【4 - 一般客服】（调用 LLM）')
        print('=' * 60)
        response = call_llm(query, "你是客服代表。用友好的语气回复。用中文回答。")

    state["response"] = response
    print(f'>>> 回复: {response[:300]}')

    return state


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第9课 - 智能客服路由系统（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 多条件路由 = if-elif-else')
    print('  - 升级人工 = 不调 LLM，直接返回固定文案（安全兜底）')
    print('  - 整个系统 = 2次分析LLM + 1次处理LLM（或0次如果升级）')
    print()

    # 示例工单（覆盖4条路径）
    examples = [
        ("我的网络连接一直断断续续", "期望→技术客服"),
        ("我被多扣了两个月的费用", "期望→账单客服"),
        ("请问你们的营业时间？", "期望→一般客服"),
        ("你们这破服务什么玩意！我要投诉！", "期望→升级人工"),
    ]

    print('--- 示例工单 ---')
    for query, expected in examples:
        print()
        print('#' * 60)
        print(f'#  工单: {query}')
        print(f'#  {expected}')
        print('#' * 60)

        result = handle_customer_query(query)

        print()
        print('=' * 60)
        print('【最终结果】')
        print('=' * 60)
        print(f'  分类: {result["category"]}')
        print(f'  情感: {result["sentiment"]}')
        print(f'  回复: {result["response"]}')
        print()

    # 交互模式
    print('\n输入工单，输入 /quit 退出\n')
    while True:
        user_input = input('客户: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break

        result = handle_customer_query(user_input)

        print()
        print('=' * 60)
        print('【最终结果】')
        print('=' * 60)
        print(f'  分类: {result["category"]}')
        print(f'  情感: {result["sentiment"]}')
        print(f'  回复: {result["response"]}')
        print()
