"""
第30课：自愈代码代理（LangGraph 框架版）

架构：代码执行 → 捕获错误 → 生成Bug报告 → 搜索历史记忆(ChromaDB)
      → 更新/新增记忆 → LLM生成修复代码 → 热补丁(exec) → 再次执行

核心模式：自愈循环 + 向量数据库记忆 + 动态代码热补丁
"""

import os
import re
import sys
import json
import time
import uuid
import types
import inspect
import httpx
import chromadb

# 修复 Windows GBK 编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from typing import Callable, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3

# ===== ChromaDB 初始化 =====
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="bug-reports")


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    """标准 LLM 调用模板"""
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }

    for attempt in range(MAX_RETRIES):
        try:
            print(f"    [LLM调用] 第{attempt+1}次请求...")
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 状态定义 =====
class State(BaseModel):
    function: Callable               # 待测试的函数引用
    function_string: str             # 函数源码字符串
    arguments: list                  # 函数参数
    error: bool                      # 是否有错误
    error_description: str = ""      # 错误描述
    new_function_string: str = ""    # LLM生成的修复代码
    bug_report: str = ""             # Bug报告
    memory_search_results: list = [] # 记忆搜索结果
    memory_ids_to_update: list = []  # 需要更新的记忆ID


# ===== 节点函数 =====

def code_execution_node(state: State):
    """节点1：执行代码，捕获错误"""
    print("\n" + "=" * 60)
    print("【代码执行】运行目标函数")
    print("=" * 60)
    print(f"  函数: {state.function.__name__}")
    print(f"  参数: {state.arguments}")

    try:
        result = state.function(*state.arguments)
        print(f"  结果: {result}")
        print(f"  状态: 执行成功!")
    except Exception as e:
        print(f"  错误: {e}")
        state.error = True
        state.error_description = str(e)
    return state


def bug_report_node(state: State):
    """节点2：LLM 生成 Bug 报告"""
    print("\n" + "=" * 60)
    print("【Bug报告】LLM 生成错误分析")
    print("=" * 60)

    prompt = (
        f"你是一个 Bug 分析专家。请为以下 Python 函数生成一份 Bug 报告。\n"
        f"函数代码:\n{state.function_string}\n"
        f"错误信息: {state.error_description}\n"
        f"请用中文回答，包含：错误原因、触发条件、建议修复方向。"
    )

    print(f"  发送给LLM: 分析函数 {state.function.__name__} 的错误...")
    bug_report = call_llm([{"role": "user", "content": prompt}])
    print(f"  Bug报告: {bug_report[:200]}...")

    state.bug_report = bug_report
    return state


def memory_search_node(state: State):
    """节点3：在 ChromaDB 中搜索相似的历史 Bug"""
    print("\n" + "=" * 60)
    print("【记忆搜索】ChromaDB 查询历史 Bug")
    print("=" * 60)

    # 用 LLM 摘要 Bug 报告，提高搜索精度
    prompt = (
        f"请将以下 Bug 报告压缩为一行简洁摘要，用于存档检索。\n"
        f"Bug 报告: {state.bug_report}\n"
        f"格式: # 函数名 ## 错误描述 ### 分析"
    )

    summary = call_llm([{"role": "user", "content": prompt}])
    print(f"  Bug摘要: {summary[:100]}...")

    results = collection.query(query_texts=[summary])

    if results["ids"][0]:
        print(f"  找到 {len(results['ids'][0])} 条历史记录")
        state.memory_search_results = [
            {
                "id": results["ids"][0][idx],
                "memory": results["documents"][0][idx],
                "distance": results["distances"][0][idx]
            }
            for idx in range(len(results["ids"][0]))
        ]
        for mem in state.memory_search_results:
            print(f"    - 距离: {mem['distance']:.4f} | {mem['memory'][:80]}...")
    else:
        print("  未找到相似历史 Bug")

    return state


def memory_filter_node(state: State):
    """节点4：过滤记忆，只保留距离 < 0.3 的高相关记录"""
    print("\n" + "=" * 60)
    print("【记忆过滤】筛选高相关历史 Bug")
    print("=" * 60)

    for memory in state.memory_search_results:
        if memory["distance"] < 0.3:
            state.memory_ids_to_update.append(memory["id"])
            print(f"  选中: {memory['id'][:8]}... (距离: {memory['distance']:.4f})")

    if not state.memory_ids_to_update:
        print("  无高相关记录，将创建新记忆")

    return state


def memory_generation_node(state: State):
    """节点5：在 ChromaDB 中保存新的 Bug 记忆"""
    print("\n" + "=" * 60)
    print("【保存记忆】新 Bug 报告存入 ChromaDB")
    print("=" * 60)

    prompt = (
        f"请将以下 Bug 报告压缩为一行简洁摘要，用于存档。\n"
        f"Bug 报告: {state.bug_report}\n"
        f"格式: # 函数名 ## 错误描述 ### 分析"
    )

    summary = call_llm([{"role": "user", "content": prompt}])
    record_id = str(uuid.uuid4())
    collection.add(ids=[record_id], documents=[summary])
    print(f"  已保存: {record_id[:8]}...")
    print(f"  内容: {summary[:150]}...")

    return state


def memory_modification_node(state: State):
    """节点6：更新已有的 Bug 记忆"""
    print("\n" + "=" * 60)
    print("【更新记忆】合并新旧 Bug 报告")
    print("=" * 60)

    memory_id = state.memory_ids_to_update.pop(0)
    state.memory_search_results.pop(0)
    results = collection.get(ids=[memory_id])
    old_memory = results["documents"][0]

    prompt = (
        f"请合并以下两条 Bug 报告为一条更完整的记录。\n"
        f"当前 Bug: {state.bug_report}\n"
        f"历史 Bug: {old_memory}\n"
        f"格式: # 函数名 ## 错误描述 ### 分析"
    )

    updated = call_llm([{"role": "user", "content": prompt}])
    collection.update(ids=[memory_id], documents=[updated])

    print(f"  旧记忆: {old_memory[:100]}...")
    print(f"  新记忆: {updated[:100]}...")

    return state


def code_update_node(state: State):
    """节点7：LLM 生成修复代码"""
    print("\n" + "=" * 60)
    print("【代码修复】LLM 生成修复方案")
    print("=" * 60)

    prompt = (
        f"你需要修复一个有错误的 Python 函数。\n"
        f"函数代码:\n{state.function_string}\n"
        f"错误信息: {state.error_description}\n"
        f"要求:\n"
        f"1. 只修复当前错误，优雅处理异常情况（返回错误消息，不要 raise）\n"
        f"2. 函数名和参数必须完全相同\n"
        f"3. 只输出函数定义代码，不要任何额外文字、代码块标记或语言声明"
    )

    print(f"  有 Bug 的函数:\n{state.function_string}")

    new_code = call_llm([{"role": "user", "content": prompt}], temperature=0.3)

    # 清理可能的代码块标记
    new_code = re.sub(r'^```(?:python)?\s*', '', new_code)
    new_code = re.sub(r'\s*```$', '', new_code)

    print(f"\n  LLM 提出的修复:\n{new_code}")

    state.new_function_string = new_code
    return state


def code_patching_node(state: State):
    """节点8：热补丁 — 用 exec() 动态替换函数"""
    print("\n" + "=" * 60)
    print("【热补丁】exec() 动态替换函数")
    print("=" * 60)

    try:
        new_code = state.new_function_string
        namespace = {}
        exec(new_code, namespace)

        func_name = state.function.__name__
        new_function = namespace[func_name]

        state.function = new_function
        state.error = False

        # 测试修复后的函数
        result = state.function(*state.arguments)
        print(f"  补丁成功! 测试结果: {result}")

    except Exception as e:
        print(f"  补丁失败: {e}")

    return state


# ===== 路由函数 =====

def error_router(state: State):
    """路由1：有错误则进入修复流程，否则结束"""
    if state.error:
        print("  [路由] 检测到错误 → 进入 Bug 报告流程")
        return "bug_report_node"
    else:
        print("  [路由] 无错误 → 结束")
        return END


def memory_filter_router(state: State):
    """路由2：有搜索结果则过滤，否则直接保存新记忆"""
    if state.memory_search_results:
        return "memory_filter_node"
    else:
        return "memory_generation_node"


def memory_generation_router(state: State):
    """路由3：有需要更新的记忆则更新，否则保存新记忆"""
    if state.memory_ids_to_update:
        return "memory_modification_node"
    else:
        return "memory_generation_node"


def memory_update_router(state: State):
    """路由4：还有记忆要更新则继续，否则进入代码修复"""
    if state.memory_ids_to_update:
        return "memory_modification_node"
    else:
        return "code_update_node"


# ===== 构建 LangGraph 工作流 =====

def build_workflow():
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("code_execution_node", code_execution_node)
    builder.add_node("bug_report_node", bug_report_node)
    builder.add_node("memory_search_node", memory_search_node)
    builder.add_node("memory_filter_node", memory_filter_node)
    builder.add_node("memory_generation_node", memory_generation_node)
    builder.add_node("memory_modification_node", memory_modification_node)
    builder.add_node("code_update_node", code_update_node)
    builder.add_node("code_patching_node", code_patching_node)

    # 入口
    builder.set_entry_point("code_execution_node")

    # 边和条件路由
    builder.add_conditional_edges("code_execution_node", error_router)
    builder.add_edge("bug_report_node", "memory_search_node")
    builder.add_conditional_edges("memory_search_node", memory_filter_router)
    builder.add_conditional_edges("memory_filter_node", memory_generation_router)
    builder.add_edge("memory_generation_node", "code_update_node")
    builder.add_conditional_edges("memory_modification_node", memory_update_router)
    builder.add_edge("code_update_node", "code_patching_node")
    builder.add_edge("code_patching_node", "code_execution_node")  # 自愈循环！

    return builder.compile()


# ===== 主函数 =====

def execute_self_healing(function, arguments):
    """执行自愈代码系统"""
    graph = build_workflow()

    state = State(
        error=False,
        function=function,
        function_string=inspect.getsource(function),
        arguments=arguments,
    )

    return graph.invoke(state)


def main():
    print("=" * 60)
    print("  自愈代码代理系统（LangGraph 框架版）")
    print("=" * 60)

    # 测试函数1：除法（除以零）
    def divide_two_numbers(a, b):
        return a / b

    # 测试函数2：列表处理（索引越界）
    def process_list(lst, index):
        return lst[index] * 2

    # 测试函数3：日期解析（格式错误）
    def parse_date(date_string):
        year, month, day = date_string.split("-")
        return {"year": int(year), "month": int(month), "day": int(day)}

    print("\n" + "*" * 60)
    print("  测试1：除以零")
    print("*" * 60)
    execute_self_healing(divide_two_numbers, [10, 0])

    print("\n" + "*" * 60)
    print("  测试2：列表索引越界")
    print("*" * 60)
    execute_self_healing(process_list, [[1, 2, 3], 5])

    print("\n" + "*" * 60)
    print("  测试3：日期格式错误")
    print("*" * 60)
    execute_self_healing(parse_date, ["2024/01/01"])


if __name__ == "__main__":
    main()
