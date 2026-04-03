"""
第42课：图检查器自动测试系统（透明调试版）

展示"用 Agent 测试 Agent"的元编程流程。
每个步骤打印完整的 Prompt 和 LLM 调用过程。
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


def call_llm(messages: list, temperature: float = 0.3) -> str:
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


# 模拟目标图
GRAPH_DESC = "一个 ReAct 算术计算 Agent。assistant节点调用add/multiply/divide工具，tools节点执行计算。"
NODES = [
    {"name": "assistant", "type": "LLM节点", "tools": ["add(a,b)", "multiply(a,b)", "divide(a,b)"],
     "input": '{"messages": [HumanMessage("3加4")]}', "output": '{"messages": [AIMessage(tool_calls=[add(3,4)])]}'},
    {"name": "tools", "type": "ToolNode", "tools": ["add", "multiply", "divide"],
     "input": '{"messages": [AIMessage(tool_calls=[add(3,4)])]}', "output": '{"messages": [ToolMessage("7")]}'}
]


def main():
    print("=" * 70)
    print("  图检查器自动测试系统（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  目标系统: {GRAPH_DESC}")

    # ==============================================================
    # 步骤1：静态分析
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤1】静态分析 — 提取图结构")
    print(f"{'=' * 70}")
    print(f"  原教程: 编译 LangGraph 图，用 NetworkX 提取节点/边/工具")
    print(f"  本质: 遍历图对象，提取每个节点的类型、工具列表、连接关系")
    for n in NODES:
        print(f"\n  节点: {n['name']} ({n['type']})")
        print(f"    工具: {n['tools']}")
        print(f"    输入: {n['input']}")
        print(f"    输出: {n['output']}")

    # ==============================================================
    # 步骤2：LLM 生成节点描述
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤2】LLM 生成节点描述")
    print(f"{'=' * 70}")
    print(f"  用 LLM 根据节点的输入/输出/工具推断其功能")

    descriptions = {}
    for n in NODES:
        prompt = (
            f"描述以下 LangGraph 节点的功能（45字以内）:\n"
            f"图: {GRAPH_DESC}\n节点: {n['name']} ({n['type']})\n"
            f"工具: {n['tools']}\n输入: {n['input']}\n输出: {n['output']}"
        )
        print(f"\n  >>> Prompt: {prompt[:150]}...")
        desc = call_llm([{"role": "user", "content": prompt}])
        descriptions[n["name"]] = desc
        print(f"  >>> 描述: {desc}")

    # ==============================================================
    # 步骤3：LLM 生成测试人设
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤3】LLM 生成测试人设")
    print(f"{'=' * 70}")
    print(f"  原教程创建3种测试角色:")
    print(f"    1. 功能测试者 — 验证正常功能")
    print(f"    2. 注入攻击测试者 — Prompt注入/越狱")
    print(f"    3. 漏洞猎手 — 边界情况/异常输入")

    tester_prompt = (
        f"为以下AI系统创建3个测试人设:\n系统: {GRAPH_DESC}\n"
        f"1. 功能测试者 2. 注入攻击测试者 3. 边界测试者\n"
        f"每个: 角色名+描述。中文。"
    )
    print(f"\n  >>> Prompt: {tester_prompt}")
    testers = call_llm([{"role": "user", "content": tester_prompt}])
    print(f"\n  >>> 测试人设:\n{testers}")

    # ==============================================================
    # 步骤4：生成测试用例
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤4】为每个(节点×测试者)生成测试用例")
    print(f"{'=' * 70}")
    print(f"  原教程: generate_pairs(nodes, testers) 生成所有组合")
    print(f"  每个组合生成3+个测试用例")

    # 只为 assistant 节点生成（简化）
    node = NODES[0]
    tc_prompt = (
        f"为以下节点生成3个测试用例:\n"
        f"节点: {node['name']} - {descriptions.get(node['name'], '')}\n"
        f"工具: {node['tools']}\n输入: {node['input']}\n\n"
        f"测试人设:\n{testers}\n\n"
        f"每个: 名称、描述、验收标准。中文。"
    )
    print(f"\n  >>> Prompt(部分): {tc_prompt[:200]}...")
    test_cases = call_llm([{"role": "user", "content": tc_prompt}])
    print(f"\n  >>> 测试用例:\n{test_cases}")

    # ==============================================================
    # 步骤5：模拟执行
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤5】执行测试（模拟）")
    print(f"{'=' * 70}")
    print(f"  原教程: 子图生成新输入 → eval() 转为实际输入 → 运行目标图 → 收集输出")
    print(f"  本课简化: LLM 模拟执行结果")

    exec_prompt = (
        f"模拟执行以下测试:\n节点: {node['name']}\n"
        f"测试用例:\n{test_cases[:500]}\n\n"
        f"对每个测试: 结果(通过/失败) + 简短说明。中文。"
    )
    results = call_llm([{"role": "user", "content": exec_prompt}])
    print(f"\n  >>> 测试结果:\n{results}")

    # ==============================================================
    # 步骤6：生成报告
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤6】测试报告")
    print(f"{'=' * 70}")

    report = f"# 测试报告\n\n**系统**: {GRAPH_DESC}\n\n## 测试结果\n\n{results}\n"
    filepath = os.path.join(os.path.dirname(__file__), "测试报告_debug.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  已保存: {filepath}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结")
    print(f"{'=' * 70}")
    print(f"  1. 这是'用 Agent 测试 Agent'的元编程系统")
    print(f"  2. 6步流水线:")
    print(f"     静态分析 → 节点描述 → 测试人设 → 测试用例 → 执行 → 分析")
    print(f"  3. 原教程的亮点:")
    print(f"     - NetworkX 静态分析图结构（节点类型/工具/边）")
    print(f"     - generate_pairs() 生成所有(节点×测试者)组合")
    print(f"     - 子图嵌套执行: 生成输入→eval()→运行目标图→收集输出")
    print(f"     - 多轮条件循环: 处理完所有组合和所有结果")
    print(f"  4. 安全测试: 包含注入攻击和越狱测试")
    print(f"  5. 本质: 多次 LLM 调用 + 组合遍历 + 图执行")
    print(f"     框架的价值: 让测试流程可追踪、可重复、可扩展")


if __name__ == "__main__":
    main()
