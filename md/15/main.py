"""
第15课 - E2E测试代理（LangGraph 框架版）

核心概念：
  - 动作循环：用计数器逐个处理动作列表
  - 代码生成+验证：LLM生成代码，ast.parse()检查语法
  - 三路条件路由：通过+继续 / 通过+完成 / 失败
  - 代码执行：生成的测试代码会被实际运行

简化说明：
  原始notebook用Playwright控制浏览器，本版简化为API/函数测试代码生成。
  聚焦于"指令拆解→逐步生成→验证→执行"的核心模式。
"""

import os
import ast
import json
import re
import time
from typing import TypedDict, List, Annotated

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（带重试）"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.2},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                print(f'    ⚠ API错误，{wait}秒后重试: {e}')
                time.sleep(wait)
            else:
                raise


def extract_json(text: str) -> dict:
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        return {}


def extract_code(text: str) -> str:
    """从LLM回复中提取Python代码块"""
    match = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ========== State ==========

class State(TypedDict):
    query: str                 # 用户测试描述
    actions: List[str]         # 拆解后的动作列表
    current_action: int        # 当前动作索引
    code_blocks: List[str]     # 每个动作生成的代码块
    error_message: str         # 错误信息（空=无错误）
    test_script: str           # 最终组装的测试脚本
    test_result: str           # 测试执行结果
    report: str                # 最终报告


# ========== 节点：指令拆解 ==========

def parse_instructions_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 指令拆解】')
    print('=' * 60)

    system = ("你是测试专家。把用户的测试描述拆解为原子动作列表。\n"
              "每个动作应该是一个具体的、可独立测试的步骤。\n"
              "返回JSON：{\"actions\": [\"动作1描述\", \"动作2描述\"]}")
    result = extract_json(call_llm(state["query"], system))
    actions = result.get("actions", ["执行基本测试"])

    print(f'>>> 拆解为 {len(actions)} 个动作:')
    for i, a in enumerate(actions, 1):
        print(f'    {i}. {a}')

    return {"actions": actions, "current_action": 0, "code_blocks": [], "error_message": ""}


# ========== 节点：生成代码 ==========

def generate_code_node(state: State):
    idx = state["current_action"]
    action = state["actions"][idx]

    print()
    print('=' * 60)
    print(f'【2 - 生成代码】（动作 {idx + 1}/{len(state["actions"])}）')
    print('=' * 60)
    print(f'>>> 当前动作: {action}')

    # 已有代码作为上下文
    prev_code = "\n".join(state["code_blocks"]) if state["code_blocks"] else "（无）"

    system = ("你是Python测试代码生成专家。为指定的测试动作生成Python代码。\n"
              "要求：\n"
              "1. 只生成这一个动作的代码片段（不要完整函数）\n"
              "2. 使用 assert 语句做断言\n"
              "3. 用注释说明这一步在做什么\n"
              "4. 代码必须是合法的Python语法\n"
              "5. 只返回代码，用 ```python ``` 包裹")

    prompt = (f"测试描述: {state['query']}\n"
              f"当前动作: {action}\n"
              f"已生成的代码:\n{prev_code}\n\n"
              f"请为当前动作生成Python测试代码。")

    response = call_llm(prompt, system)
    code = extract_code(response)

    print(f'>>> 生成的代码:')
    for line in code.split('\n')[:10]:
        print(f'    {line}')
    if code.count('\n') > 10:
        print(f'    ... ({code.count(chr(10)) + 1}行)')

    return {"code_blocks": state["code_blocks"] + [code], "error_message": ""}


# ========== 节点：语法验证 ==========

def validate_code_node(state: State):
    idx = state["current_action"]
    code = state["code_blocks"][-1]  # 最新生成的代码

    print()
    print('=' * 60)
    print(f'【3 - 语法验证】（动作 {idx + 1}）')
    print('=' * 60)

    try:
        ast.parse(code)
        print(f'>>> ✓ 语法验证通过')
        return {"current_action": idx + 1, "error_message": ""}
    except SyntaxError as e:
        error_msg = f"语法错误（动作{idx + 1}）: {e}"
        print(f'>>> ✗ {error_msg}')
        return {"error_message": error_msg}


# ========== 节点：错误处理 ==========

def handle_error_node(state: State):
    print()
    print('=' * 60)
    print('【错误处理】')
    print('=' * 60)
    print(f'>>> 错误: {state["error_message"]}')

    report = (f"# 测试生成失败报告\n\n"
              f"## 错误信息\n{state['error_message']}\n\n"
              f"## 已完成的动作\n"
              + "\n".join(f"- {a}" for a in state["actions"][:state["current_action"]])
              + f"\n\n## 已生成的代码\n```python\n" + "\n".join(state["code_blocks"]) + "\n```")

    return {"report": report}


# ========== 节点：组装脚本 ==========

def post_process_node(state: State):
    print()
    print('=' * 60)
    print('【4 - 组装测试脚本】')
    print('=' * 60)

    # 拼接所有代码块到一个测试函数中
    all_code = "\n\n".join(state["code_blocks"])
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in all_code.split("\n"))

    test_script = (f"def test_generated():\n"
                   f"    \"\"\"自动生成的测试: {state['query'][:50]}\"\"\"\n"
                   f"{indented}\n"
                   f"    print('所有测试步骤通过！')\n")

    print(f'>>> 测试脚本:')
    for line in test_script.split('\n')[:15]:
        print(f'    {line}')

    return {"test_script": test_script}


# ========== 节点：执行测试 ==========

def execute_test_node(state: State):
    print()
    print('=' * 60)
    print('【5 - 执行测试】')
    print('=' * 60)

    try:
        # 编译并执行测试函数
        exec_namespace = {}
        exec(state["test_script"], exec_namespace)
        test_func = exec_namespace.get("test_generated")

        if test_func:
            test_func()
            result = "✓ 测试通过"
        else:
            result = "✗ 未找到测试函数"
    except AssertionError as e:
        result = f"✗ 断言失败: {e}"
    except Exception as e:
        result = f"✗ 执行错误: {type(e).__name__}: {e}"

    print(f'>>> 结果: {result}')
    return {"test_result": result}


# ========== 节点：生成报告 ==========

def generate_report_node(state: State):
    print()
    print('=' * 60)
    print('【6 - 生成测试报告】')
    print('=' * 60)

    actions_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(state["actions"]))

    report = "\n".join([
        "=" * 50,
        "        测试生成报告",
        "=" * 50,
        "",
        f"测试描述: {state['query']}",
        f"动作数量: {len(state['actions'])}",
        "",
        "【动作列表】",
        actions_str,
        "",
        "【测试结果】",
        f"  {state['test_result']}",
        "",
        "【生成的测试脚本】",
        f"```python",
        state["test_script"],
        f"```",
        "",
        "=" * 50,
    ])

    print(report)
    return {"report": report}


# ========== 路由函数 ==========

def decide_next(state: State) -> str:
    """三路条件路由：失败/完成/继续"""
    print()
    print(f'>>> 路由决策: error={bool(state["error_message"])}, '
          f'action={state["current_action"]}/{len(state["actions"])}')

    if state["error_message"]:
        print(f'>>> → 错误处理（语法验证失败）')
        return "handle_error"
    elif state["current_action"] >= len(state["actions"]):
        print(f'>>> → 组装脚本（所有动作完成）')
        return "post_process"
    else:
        print(f'>>> → 继续生成（还有动作未处理）')
        return "generate_code"


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("parse_instructions", parse_instructions_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("validate_code", validate_code_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("post_process", post_process_node)
    workflow.add_node("execute_test", execute_test_node)
    workflow.add_node("generate_report", generate_report_node)

    # 入口
    workflow.set_entry_point("parse_instructions")

    # 线性：拆解 → 第一次生成代码
    workflow.add_edge("parse_instructions", "generate_code")

    # 生成 → 验证
    workflow.add_edge("generate_code", "validate_code")

    # 验证后三路分支
    workflow.add_conditional_edges("validate_code", decide_next, {
        "generate_code": "generate_code",   # 继续循环
        "post_process": "post_process",     # 全部完成
        "handle_error": "handle_error",     # 出错
    })

    # 后处理 → 执行 → 报告
    workflow.add_edge("post_process", "execute_test")
    workflow.add_edge("execute_test", "generate_report")
    workflow.add_edge("generate_report", END)

    # 错误 → 结束
    workflow.add_edge("handle_error", END)

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
        print('\n>>> 粘贴到 https://mermaid.live 查看可视化')
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第15课 - E2E测试代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 示例
    examples = [
        "测试一个计算器函数：验证加法（1+2=3）、减法（5-3=2）、乘法（4*3=12），以及除以零的异常处理",
        "测试一个用户注册流程：验证用户名不能为空、密码长度至少8位、两次密码输入必须一致",
    ]

    print('示例测试描述:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    query = input('请输入测试描述（回车使用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    print()
    print('#' * 60)
    print('#  开始生成测试')
    print('#' * 60)

    result = app.invoke({
        "query": query,
        "actions": [],
        "current_action": 0,
        "code_blocks": [],
        "error_message": "",
        "test_script": "",
        "test_result": "",
        "report": "",
    })

    print()
    print(f'>>> 测试生成完成')
