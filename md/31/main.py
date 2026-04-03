"""
第31课：数据库探索代理舰队（LangGraph 框架版）

架构：Supervisor 主管调度 4 个子 Agent
  - 分类器：判断用户输入类型（闲聊 or 数据库查询）
  - 发现者：探索数据库 Schema，生成关系图
  - 规划者：将问题拆解为可执行步骤
  - 推理者：执行 SQL 查询

核心模式：Supervisor 主管模式 + 多 Agent 舰队 + Schema 图缓存
"""

import os
import re
import sys
import json
import time
import sqlite3
import httpx
from typing import TypedDict, List, Optional, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "all_agents_tutorials", "data", "chinook.db")


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


# ===== SQLite 工具 =====
def run_sql(query: str) -> str:
    """执行 SQL 查询并返回结果"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        if not rows:
            return "查询无结果"
        result = " | ".join(columns) + "\n"
        for row in rows[:20]:  # 限制20行
            result += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 20:
            result += f"... 共 {len(rows)} 行\n"
        return result
    except Exception as e:
        return f"SQL错误: {e}"


def get_schema_info() -> str:
    """获取数据库完整 Schema 信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()

        col_info = []
        for col in columns:
            col_str = f"  - {col[1]} ({col[2]})"
            if col[5]:  # is primary key
                col_str += " [主键]"
            if col[3]:  # not null
                col_str += " [非空]"
            # 检查外键
            for fk in fks:
                if fk[3] == col[1]:
                    col_str += f" → {fk[2]}.{fk[4]}"
            col_info.append(col_str)

        schema_info.append(f"表: {table}\n" + "\n".join(col_info))

    conn.close()
    return "\n\n".join(schema_info)


# ===== 状态定义 =====
class ConversationState(TypedDict):
    question: str                     # 用户问题
    input_type: str                   # 输入分类
    schema_info: str                  # 数据库 Schema（缓存）
    plan: List[str]                   # 查询计划
    db_results: str                   # 数据库查询结果
    response: str                     # 最终回复


# ===== 节点函数 =====

def classify_input(state: ConversationState) -> dict:
    """分类器 Agent：判断用户输入类型"""
    print("\n" + "=" * 60)
    print("【分类器 Agent】判断输入类型")
    print("=" * 60)
    print(f"  用户输入: {state['question']}")

    prompt = (
        "你是一个输入分类器。将用户输入分为以下类别之一:\n"
        "- DATABASE_QUERY: 关于数据、需要查询数据库的问题\n"
        "- GREETING: 问候语\n"
        "- CHITCHAT: 闲聊\n"
        "- FAREWELL: 告别\n"
        "只输出类别名称，不要其他内容。"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["question"]}
    ]

    classification = call_llm(messages, temperature=0).strip().upper()
    # 规范化分类结果
    if "DATABASE" in classification or "QUERY" in classification:
        classification = "DATABASE_QUERY"
    elif "GREET" in classification:
        classification = "GREETING"
    elif "FAREWELL" in classification or "BYE" in classification:
        classification = "FAREWELL"
    else:
        classification = "CHITCHAT"

    print(f"  分类结果: {classification}")
    return {**state, "input_type": classification}


def discover_database(state: ConversationState) -> dict:
    """发现者 Agent：探索数据库 Schema（一次性，缓存）"""
    print("\n" + "=" * 60)
    print("【发现者 Agent】探索数据库 Schema")
    print("=" * 60)

    if state.get("schema_info"):
        print("  Schema 已缓存，跳过发现步骤")
        return state

    print(f"  数据库路径: {DB_PATH}")
    schema = get_schema_info()
    print(f"  发现的 Schema:\n{schema[:500]}...")
    return {**state, "schema_info": schema}


def create_plan(state: ConversationState) -> dict:
    """规划者 Agent：将问题拆解为执行步骤"""
    print("\n" + "=" * 60)
    print("【规划者 Agent】制定查询计划")
    print("=" * 60)

    prompt = (
        f"你是一个数据库查询规划专家。根据用户问题和数据库结构制定查询计划。\n\n"
        f"数据库结构:\n{state['schema_info'][:2000]}\n\n"
        f"用户问题: {state['question']}\n\n"
        f"要求:\n"
        f"1. 每个步骤一行\n"
        f"2. 以 'SQL:' 开头表示需要执行的 SQL 查询\n"
        f"3. 以 '总结:' 开头表示需要汇总回复\n"
        f"4. SQL 必须是合法的 SQLite 语法\n"
        f"5. 保持计划简洁，通常 1-2 个 SQL 步骤即可\n\n"
        f"示例:\n"
        f"SQL: SELECT Name FROM artists LIMIT 5\n"
        f"总结: 用友好的方式展示查询结果"
    )

    plan_text = call_llm([{"role": "user", "content": prompt}], temperature=0.3)
    steps = [s.strip() for s in plan_text.split("\n") if s.strip() and ("SQL:" in s or "总结:" in s)]

    if not steps:
        steps = ["总结: 用友好的方式回答用户问题"]

    print(f"  查询计划:")
    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")

    return {**state, "plan": steps}


def execute_plan(state: ConversationState) -> dict:
    """推理者 Agent：逐步执行 SQL 查询"""
    print("\n" + "=" * 60)
    print("【推理者 Agent】执行查询计划")
    print("=" * 60)

    results = []
    for step in state["plan"]:
        if step.startswith("SQL:"):
            sql = step[4:].strip()
            # 清理 SQL（去掉可能的代码块标记）
            sql = re.sub(r'^```(?:sql)?\s*', '', sql)
            sql = re.sub(r'\s*```$', '', sql)
            print(f"\n  执行 SQL: {sql}")
            result = run_sql(sql)
            print(f"  结果: {result[:200]}...")
            results.append(f"查询: {sql}\n结果:\n{result}")
        else:
            results.append(step)

    return {**state, "db_results": "\n\n".join(results)}


def generate_response(state: ConversationState) -> dict:
    """主管 Agent：生成最终回复"""
    print("\n" + "=" * 60)
    print("【主管 Agent】生成最终回复")
    print("=" * 60)

    is_chat = state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]

    if is_chat:
        prompt = (
            f"你是一个友好的 AI 助手。请自然地回复用户的消息，保持简短友好。\n"
            f"用户消息: {state['question']}"
        )
    else:
        prompt = (
            f"你是一个数据分析助手。根据查询结果回答用户问题。\n\n"
            f"用户问题: {state['question']}\n\n"
            f"查询结果:\n{state.get('db_results', '无结果')}\n\n"
            f"要求: 用中文、友好的方式展示结果，使用列表或表格格式。"
        )

    response = call_llm([{"role": "user", "content": prompt}])
    print(f"  最终回复: {response[:200]}...")
    return {**state, "response": response}


# ===== 路由函数 =====
def input_type_router(state: ConversationState) -> str:
    if state.get("input_type") == "DATABASE_QUERY":
        print("  [路由] 数据库查询 → 发现 Schema")
        return "discover_database"
    else:
        print("  [路由] 闲聊/问候 → 直接生成回复")
        return "generate_response"


def plan_router(state: ConversationState) -> str:
    if state.get("plan"):
        return "execute_plan"
    else:
        return "generate_response"


# ===== 构建工作流 =====
def build_workflow():
    builder = StateGraph(ConversationState)

    builder.add_node("classify_input", classify_input)
    builder.add_node("discover_database", discover_database)
    builder.add_node("create_plan", create_plan)
    builder.add_node("execute_plan", execute_plan)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "classify_input")
    builder.add_conditional_edges("classify_input", input_type_router)
    builder.add_edge("discover_database", "create_plan")
    builder.add_conditional_edges("create_plan", plan_router)
    builder.add_edge("execute_plan", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


# ===== 主函数 =====
def main():
    print("=" * 60)
    print("  数据库探索代理舰队（LangGraph 框架版）")
    print("=" * 60)
    print(f"  数据库: {DB_PATH}")

    # 测试数据库连接
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks")
    print(f"  曲目总数: {cursor.fetchone()[0]}")
    conn.close()

    graph = build_workflow()

    # 保持状态跨轮次（Schema 缓存）
    state = {
        "question": "",
        "input_type": "",
        "schema_info": "",
        "plan": [],
        "db_results": "",
        "response": ""
    }

    # 测试用例
    questions = [
        "你好，今天怎么样？",
        "曲目数量最多的前3位艺术家是谁？",
        "他们分别有哪些专辑？",
    ]

    for q in questions:
        print("\n" + "*" * 60)
        print(f"  用户: {q}")
        print("*" * 60)

        state["question"] = q
        result = graph.invoke(state)

        # 保留 schema 缓存
        state["schema_info"] = result.get("schema_info", "")

        print(f"\n  回复: {result['response']}")

    print("\n" + "=" * 60)
    print("  所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
