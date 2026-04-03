"""
第42课：图检查器自动测试系统（简化版）

架构：分析目标图结构 → 生成节点描述 → 创建测试人设 → 生成测试用例 → 模拟执行 → 分析结果
核心模式：用 Agent 测试 Agent（元编程）+ 图结构分析 + 测试生成
"""

import os
import re
import sys
import json
import time
import httpx
from typing import TypedDict, List, Dict
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
def call_llm(messages: list, temperature: float = 0.3) -> str:
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
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# ===== 模拟一个目标图的结构信息 =====
SAMPLE_GRAPH = {
    "description": "一个简单的 ReAct 算术计算 Agent。有一个助手节点和一个工具节点，助手可以调用加法、乘法、除法工具。",
    "nodes": [
        {
            "name": "assistant",
            "type": "LLM节点",
            "tools": ["add(a,b)", "multiply(a,b)", "divide(a,b)"],
            "edges_in": ["__start__", "tools"],
            "edges_out": ["tools", "__end__"],
            "sample_input": '{"messages": [HumanMessage("计算 3 加 4")]}',
            "sample_output": '{"messages": [AIMessage(tool_calls=[{name:"add", args:{a:3,b:4}}])]}'
        },
        {
            "name": "tools",
            "type": "ToolNode",
            "tools": ["add", "multiply", "divide"],
            "edges_in": ["assistant"],
            "edges_out": ["assistant"],
            "sample_input": '{"messages": [AIMessage(tool_calls=[{name:"add", args:{a:3,b:4}}])]}',
            "sample_output": '{"messages": [ToolMessage(content="7")]}'
        }
    ]
}


# ===== 步骤1：静态分析 =====
def static_analysis():
    print("\n" + "=" * 60)
    print("【步骤1】静态分析 — 提取图结构")
    print("=" * 60)
    print(f"  图描述: {SAMPLE_GRAPH['description']}")
    print(f"  节点数: {len(SAMPLE_GRAPH['nodes'])}")
    for node in SAMPLE_GRAPH["nodes"]:
        print(f"    - {node['name']} ({node['type']})")
        print(f"      工具: {node['tools']}")
        print(f"      入边: {node['edges_in']} → 出边: {node['edges_out']}")


# ===== 步骤2：生成节点描述 =====
def generate_descriptions():
    print("\n" + "=" * 60)
    print("【步骤2】LLM 生成节点描述")
    print("=" * 60)

    descriptions = {}
    for node in SAMPLE_GRAPH["nodes"]:
        prompt = (
            f"你是一个 LangGraph 工作流开发专家。请描述以下节点的功能（45字以内）。\n\n"
            f"图描述: {SAMPLE_GRAPH['description']}\n"
            f"节点名: {node['name']}\n"
            f"类型: {node['type']}\n"
            f"工具: {node['tools']}\n"
            f"入边: {node['edges_in']}, 出边: {node['edges_out']}\n"
            f"输入示例: {node['sample_input']}\n"
            f"输出示例: {node['sample_output']}"
        )

        desc = call_llm([{"role": "user", "content": prompt}])
        descriptions[node["name"]] = desc
        print(f"  {node['name']}: {desc}")

    return descriptions


# ===== 步骤3：生成测试人设 =====
def generate_testers():
    print("\n" + "=" * 60)
    print("【步骤3】LLM 生成测试人设")
    print("=" * 60)

    prompt = (
        f"你需要为以下 AI Agent 系统创建3个测试人设。\n\n"
        f"系统描述: {SAMPLE_GRAPH['description']}\n\n"
        f"请创建以下3种测试者:\n"
        f"1. 功能测试者: 验证正常功能\n"
        f"2. 注入攻击测试者: 尝试 Prompt 注入和越狱\n"
        f"3. 边界测试者: 测试异常输入和边界情况\n\n"
        f"每个测试者输出: 角色名称 + 一句话描述。用中文。"
    )

    result = call_llm([{"role": "user", "content": prompt}])
    print(f"  测试人设:\n{result}")
    return result


# ===== 步骤4：生成测试用例 =====
def generate_test_cases(testers: str, descriptions: dict):
    print("\n" + "=" * 60)
    print("【步骤4】LLM 生成测试用例")
    print("=" * 60)

    all_cases = []
    for node in SAMPLE_GRAPH["nodes"]:
        prompt = (
            f"你是一个 AI 系统测试专家。请为以下节点生成3个测试用例。\n\n"
            f"节点: {node['name']}\n"
            f"描述: {descriptions.get(node['name'], '无')}\n"
            f"工具: {node['tools']}\n"
            f"输入示例: {node['sample_input']}\n\n"
            f"测试人设:\n{testers}\n\n"
            f"每个测试用例包含: 名称、描述、验收标准。用中文。"
        )

        cases = call_llm([{"role": "user", "content": prompt}])
        print(f"\n  节点 [{node['name']}] 的测试用例:\n{cases[:400]}...")
        all_cases.append({"node": node["name"], "cases": cases})

    return all_cases


# ===== 步骤5：模拟执行测试 =====
def execute_tests(test_cases: list):
    print("\n" + "=" * 60)
    print("【步骤5】模拟执行测试")
    print("=" * 60)

    # 模拟测试结果
    results = []
    for tc in test_cases:
        prompt = (
            f"假设你是一个 AI Agent 测试执行器。以下是测试用例:\n\n"
            f"节点: {tc['node']}\n"
            f"测试用例:\n{tc['cases'][:500]}\n\n"
            f"请模拟执行这些测试，对每个测试用例给出:\n"
            f"- 结果: 通过/失败\n"
            f"- 说明: 简短解释\n"
            f"用中文。"
        )

        result = call_llm([{"role": "user", "content": prompt}])
        print(f"\n  节点 [{tc['node']}] 测试结果:\n{result[:400]}...")
        results.append({"node": tc["node"], "result": result})

    return results


# ===== 步骤6：生成测试报告 =====
def generate_report(results: list):
    print("\n" + "=" * 60)
    print("【步骤6】测试报告")
    print("=" * 60)

    report = f"# 图检查器测试报告\n\n"
    report += f"**目标系统**: {SAMPLE_GRAPH['description']}\n\n"

    for r in results:
        report += f"## 节点: {r['node']}\n\n{r['result']}\n\n---\n\n"

    filepath = os.path.join(os.path.dirname(__file__), "测试报告.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  已保存: {filepath}")
    print(f"\n{report}")


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  图检查器自动测试系统")
    print("=" * 60)

    # 步骤1: 静态分析
    static_analysis()

    # 步骤2: 生成节点描述
    descriptions = generate_descriptions()

    # 步骤3: 生成测试人设
    testers = generate_testers()

    # 步骤4: 生成测试用例
    test_cases = generate_test_cases(testers, descriptions)

    # 步骤5: 执行测试
    results = execute_tests(test_cases)

    # 步骤6: 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()
