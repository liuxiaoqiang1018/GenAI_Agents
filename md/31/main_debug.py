"""
第31课：数据库探索代理舰队（透明调试版）

不使用任何框架，纯手写多 Agent 协作流程。
让你看清 Supervisor 主管模式 + 多 Agent 舰队的内部机制。
"""

import os
import re
import sys
import json
import time
import sqlite3
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
            print(f"    !! LLM调用失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== SQLite 工具 =====
def run_sql(query: str) -> str:
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
        for row in rows[:20]:
            result += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 20:
            result += f"... 共 {len(rows)} 行\n"
        return result
    except Exception as e:
        return f"SQL错误: {e}"


def get_schema_info() -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
            if col[5]:
                col_str += " [主键]"
            if col[3]:
                col_str += " [非空]"
            for fk in fks:
                if fk[3] == col[1]:
                    col_str += f" → {fk[2]}.{fk[4]}"
            col_info.append(col_str)

        schema_info.append(f"表: {table}\n" + "\n".join(col_info))

    conn.close()
    return "\n\n".join(schema_info)


def run_conversation(questions: list):
    """纯手写的多 Agent 协作流程"""

    print("=" * 70)
    print("  数据库探索代理舰队（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  数据库: {DB_PATH}")

    # 状态（跨轮次保持）
    schema_info = ""

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'*' * 70}")
        print(f"  第 {q_idx} 轮对话 — 用户: {question}")
        print(f"{'*' * 70}")

        # ==============================================================
        # 阶段1：分类器 Agent — 判断输入类型
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段1】分类器 Agent — 判断输入类型")
        print(f"{'=' * 70}")

        classify_prompt = (
            "你是一个输入分类器。将用户输入分为以下类别之一:\n"
            "- DATABASE_QUERY: 关于数据、需要查询数据库的问题\n"
            "- GREETING: 问候语\n"
            "- CHITCHAT: 闲聊\n"
            "- FAREWELL: 告别\n"
            "只输出类别名称，不要其他内容。"
        )

        classify_messages = [
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": question}
        ]

        print(f"\n  >>> 发送给 LLM 的 Prompt <<<")
        print(f"  {'-' * 60}")
        print(f"  [system]: {classify_prompt[:100]}...")
        print(f"  [user]: {question}")
        print(f"  {'-' * 60}")

        classification = call_llm(classify_messages, temperature=0).strip().upper()
        if "DATABASE" in classification or "QUERY" in classification:
            input_type = "DATABASE_QUERY"
        elif "GREET" in classification:
            input_type = "GREETING"
        elif "FAREWELL" in classification or "BYE" in classification:
            input_type = "FAREWELL"
        else:
            input_type = "CHITCHAT"

        print(f"\n  >>> LLM 响应: {classification}")
        print(f"  >>> 规范化结果: {input_type}")

        # ==============================================================
        # 路由决策
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【路由决策】input_type_router")
        print(f"{'=' * 70}")
        print(f"  input_type = {input_type}")
        print(f"  规则: DATABASE_QUERY → 发现Schema → 规划 → 执行")
        print(f"        其他 → 直接生成友好回复")

        if input_type != "DATABASE_QUERY":
            print(f"  结果: → 直接生成回复（跳过数据库流程）")

            chat_prompt = (
                f"你是一个友好的 AI 助手。请自然地回复用户的消息，保持简短友好。\n"
                f"用户消息: {question}"
            )
            response = call_llm([{"role": "user", "content": chat_prompt}])
            print(f"\n  回复: {response}")
            continue

        print(f"  结果: → 进入数据库查询流程")

        # ==============================================================
        # 阶段2：发现者 Agent — 探索数据库 Schema
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段2】发现者 Agent — 探索数据库 Schema")
        print(f"{'=' * 70}")

        if schema_info:
            print(f"  Schema 已缓存（上轮发现的），跳过")
            print(f"  这就是 LangGraph State 的 reducer 机制：保留旧值不覆盖")
        else:
            print(f"  首次查询，执行 Schema 发现...")
            print(f"  直接用 SQLite PRAGMA 命令获取表结构（不需要 LLM）")
            schema_info = get_schema_info()
            print(f"\n  发现的 Schema（前500字）:")
            print(f"  {schema_info[:500]}...")

        # ==============================================================
        # 阶段3：规划者 Agent — 制定查询计划
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段3】规划者 Agent — 制定查询计划")
        print(f"{'=' * 70}")

        plan_prompt = (
            f"你是一个数据库查询规划专家。根据用户问题和数据库结构制定查询计划。\n\n"
            f"数据库结构:\n{schema_info[:2000]}\n\n"
            f"用户问题: {question}\n\n"
            f"要求:\n"
            f"1. 每个步骤一行\n"
            f"2. 以 'SQL:' 开头表示需要执行的 SQL 查询\n"
            f"3. 以 '总结:' 开头表示需要汇总回复\n"
            f"4. SQL 必须是合法的 SQLite 语法\n"
            f"5. 保持计划简洁，通常 1-2 个 SQL 步骤即可\n\n"
            f"示例:\nSQL: SELECT Name FROM artists LIMIT 5\n总结: 用友好的方式展示查询结果"
        )

        print(f"\n  >>> 发送给 LLM: 规划查询步骤...")
        print(f"  [用户问题]: {question}")

        plan_text = call_llm([{"role": "user", "content": plan_prompt}], temperature=0.3)

        print(f"\n  >>> LLM 响应（原始计划）:")
        print(f"  {plan_text}")

        steps = [s.strip() for s in plan_text.split("\n") if s.strip() and ("SQL:" in s or "总结:" in s)]
        if not steps:
            steps = ["总结: 用友好的方式回答用户问题"]

        print(f"\n  解析后的步骤:")
        for i, step in enumerate(steps, 1):
            print(f"    {i}. {step}")

        # ==============================================================
        # 阶段4：推理者 Agent — 执行 SQL 查询
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段4】推理者 Agent — 执行查询计划")
        print(f"{'=' * 70}")

        results = []
        for step in steps:
            if step.startswith("SQL:"):
                sql = step[4:].strip()
                sql = re.sub(r'^```(?:sql)?\s*', '', sql)
                sql = re.sub(r'\s*```$', '', sql)
                print(f"\n  执行 SQL: {sql}")
                result = run_sql(sql)
                print(f"  结果:\n{result[:300]}")
                results.append(f"查询: {sql}\n结果:\n{result}")
            else:
                results.append(step)

        db_results = "\n\n".join(results)

        # ==============================================================
        # 阶段5：主管 Agent — 生成最终回复
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段5】主管 Agent — 生成最终回复")
        print(f"{'=' * 70}")

        response_prompt = (
            f"你是一个数据分析助手。根据查询结果回答用户问题。\n\n"
            f"用户问题: {question}\n\n"
            f"查询结果:\n{db_results}\n\n"
            f"要求: 用中文、友好的方式展示结果，使用列表或表格格式。"
        )

        print(f"\n  >>> 发送给 LLM: 汇总结果生成回复...")

        response = call_llm([{"role": "user", "content": response_prompt}])

        print(f"\n  >>> 最终回复:")
        print(f"  {response}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：LangGraph 在本课做了什么")
    print(f"{'=' * 70}")
    print(f"  1. StateGraph 定义了 ConversationState，跨轮次保持 schema_info")
    print(f"  2. 5 个节点代表 4 个 Agent 角色 + 1 个路由:")
    print(f"     - classify_input: 分类器，判断闲聊 or 数据库查询")
    print(f"     - discover_database: 发现者，一次性探索 Schema 并缓存")
    print(f"     - create_plan: 规划者，拆解问题为 SQL 步骤")
    print(f"     - execute_plan: 推理者，逐步执行 SQL")
    print(f"     - generate_response: 主管，生成最终回复")
    print(f"  3. 条件路由: 闲聊直接回复，查询走完整流程")
    print(f"  4. State 的 reducer: db_graph 用 reducer 保持旧值，避免重复发现")
    print(f"  5. 本质: if-else 分支 + 多个 LLM 调用 + SQL 执行")
    print(f"     Supervisor 模式就是一个 Agent 持有其他 Agent 的引用并调度")


def main():
    questions = [
        "你好，今天怎么样？",
        "曲目数量最多的前3位艺术家是谁？",
        "他们分别有哪些专辑？",
    ]
    run_conversation(questions)


if __name__ == "__main__":
    main()
