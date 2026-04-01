"""
第25课 - 任务导向工具调用代理内部机制

目的：让你看清工具调用Agent的本质：
  1. 工具 = 普通函数 + 名称/描述/参数（元信息）
  2. Agent循环 = while True: LLM决策 → 执行函数 → 把结果反馈给LLM
  3. LLM不执行 = LLM只输出"调summarize"的文本指令，系统解析后调用
  4. 整个系统 = prompt里告诉LLM有哪些工具 + 循环解析LLM的输出

Java 类比：
  - 工具 = @RequestMapping 注解的Controller方法
  - Agent循环 = DispatcherServlet 的请求分发循环
  - LLM决策 = 路由器根据URL决定调哪个Controller
  - 工具执行 = Controller.method() 实际执行
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


def call_llm_simple(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
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
#  工具的本质：就是普通函数
# ================================================================

def summarize(text: str) -> str:
    """工具1：摘要"""
    return call_llm_simple(text, "用中文摘要以下文本，保留核心信息。")

def translate(text: str) -> str:
    """工具2：翻译成英语"""
    return call_llm_simple(text, "翻译成英语。只返回翻译结果。")

def word_count(text: str) -> str:
    """工具3：统计字数（不调LLM，纯Python）"""
    return f"共 {len(text)} 个字符"

def extract_keywords(text: str) -> str:
    """工具4：提取关键词"""
    return call_llm_simple(text, "提取5个核心关键词，逗号分隔。")

# 工具注册表（就是一个dict）
TOOLS = {
    "summarize": summarize,
    "translate": translate,
    "word_count": word_count,
    "extract_keywords": extract_keywords,
}


# ================================================================
#  完整流程：手动版Agent循环
# ================================================================

def run_task(task: str) -> str:
    """
    工具调用Agent的完整流程 — 不用任何框架。

    Java 类比：
        public class AgentService {
            Map<String, Tool> tools = Map.of(
                "summarize", new SummarizeTool(),
                "translate", new TranslateTool()
            );

            public String execute(String task) {
                // Agent循环（就是 while + switch-case）
                List<String> observations = new ArrayList<>();
                while (true) {
                    // 1. LLM决策：下一步调什么工具？
                    String action = llm.decide(task, tools, observations);
                    if (action.startsWith("FINAL")) return action;

                    // 2. 解析+执行工具
                    Tool tool = tools.get(action.toolName);
                    String result = tool.execute(action.args);

                    // 3. 把结果存起来，下一轮LLM能看到
                    observations.add(result);
                }
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：LLM决定执行计划
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：LLM制定计划】')
    print('=' * 60)

    plan_prompt = (f"你是任务规划器。分析以下任务，列出需要按顺序调用的工具。\n"
                   f"可用工具：summarize(摘要)、translate(翻译成英语)、word_count(字数统计)、extract_keywords(提取关键词)\n\n"
                   f"任务：{task}\n\n"
                   f"按顺序列出需要的工具（每行一个工具名）：")

    plan = call_llm_simple(plan_prompt)
    total_llm_calls += 1
    print(f'>>> 执行计划: {plan}')

    # 解析工具列表
    tool_sequence = []
    for name in TOOLS:
        if name in plan.lower():
            tool_sequence.append(name)

    if not tool_sequence:
        tool_sequence = ["summarize"]  # 兜底

    print(f'>>> 解析出工具序列: {tool_sequence}')

    # ==========================================
    # 第2步：顺序执行工具（Agent循环的本质）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：Agent循环 — 逐个执行工具】')
    print('=' * 60)

    # 提取原始文本
    current_text = task  # 初始输入
    # 尝试提取冒号后的文本
    if '：' in task:
        current_text = task.split('：', 1)[1].strip()
    elif ':' in task:
        current_text = task.split(':', 1)[1].strip()

    results = {}
    for i, tool_name in enumerate(tool_sequence, 1):
        print()
        print(f'    [{i}/{len(tool_sequence)}] 调用工具: {tool_name}')

        func = TOOLS[tool_name]

        # 确定输入：如果前面有摘要结果且当前是翻译，用摘要结果
        if tool_name == "translate" and "summarize" in results:
            input_text = results["summarize"]
            print(f'    输入: 上一步的摘要结果')
        else:
            input_text = current_text
            print(f'    输入: 原始文本（{len(input_text)}字）')

        # 执行！（这就是agent循环里的"工具执行"）
        result = func(input_text)
        if tool_name in ["summarize", "translate", "extract_keywords"]:
            total_llm_calls += 1

        results[tool_name] = result
        print(f'    结果: {result[:120]}...')

    # ==========================================
    # 第3步：组合最终结果
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：组合最终结果】')
    print('=' * 60)

    final_parts = []
    for tool_name, result in results.items():
        label = {"summarize": "摘要", "translate": "翻译", "word_count": "字数", "extract_keywords": "关键词"}.get(tool_name, tool_name)
        final_parts.append(f"{label}: {result}")

    final = "\n".join(final_parts)
    print(final)
    print()
    print(f'>>> LLM调用: {total_llm_calls}次（计划1 + 工具{total_llm_calls-1}）')

    return final


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第25课 - 任务导向工具调用代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 工具 = 普通函数，放在dict里注册')
    print('  - Agent循环 = for tool in plan: result = tool(input)')
    print('  - LLM决策 = 先问LLM"该调什么"，再按顺序执行')
    print('  - 和第7课区别：第7课LLM每步决策，本课一次规划后顺序执行')
    print()

    examples = [
        "请摘要并翻译成英语：人工智能正在改变软件开发方式，AI编程助手已成为开发者日常工具，效率提升30%到50%。",
        "提取关键词并统计字数：大语言模型核心技术包括Transformer架构、注意力机制、预训练和微调。",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex[:60]}...')
    print()

    task = input('任务（回车用示例1）: ').strip()
    if not task:
        task = examples[0]
        print(f'>>> 使用: {task[:60]}...')

    result = run_task(task)

    print()
    print('=' * 60)
    print('【最终结果】')
    print('=' * 60)
    print(result)
