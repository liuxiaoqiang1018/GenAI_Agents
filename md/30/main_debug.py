"""
第30课：自愈代码代理（透明调试版）

不使用任何框架，纯手写自愈循环，打印每个阶段的完整 Prompt 和调用日志。
让你看清 LangGraph 自愈循环 + 向量数据库记忆的内部机制。
"""

import os
import re
import sys
import json
import time
import uuid
import inspect
import httpx
import chromadb

# 修复 Windows GBK 编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3
MAX_HEAL_ATTEMPTS = 3  # 最多修复次数，防止无限循环

# ===== ChromaDB 初始化 =====
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="bug-reports-debug")


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
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


def self_healing_code(function, arguments):
    """纯手写的自愈代码循环 — 展示 LangGraph 内部做了什么"""

    func = function
    func_string = inspect.getsource(function)
    func_name = function.__name__

    print("\n" + "=" * 70)
    print(f"  自愈代码系统 — 测试函数: {func_name}")
    print("=" * 70)
    print(f"  函数源码:\n{func_string}")
    print(f"  参数: {arguments}")

    for heal_round in range(MAX_HEAL_ATTEMPTS):

        # =============================================================
        # 阶段1：执行代码（对应 code_execution_node）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段1】执行代码（第{heal_round + 1}轮）")
        print(f"{'=' * 70}")
        print(f"  调用: {func_name}({', '.join(repr(a) for a in arguments)})")

        error = False
        error_description = ""

        try:
            result = func(*arguments)
            print(f"  结果: {result}")
            print(f"  执行成功！自愈完成。")
            return result
        except Exception as e:
            error = True
            error_description = str(e)
            print(f"  错误: {e}")

        # =============================================================
        # 阶段2：路由决策（对应 error_router）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【路由决策】error_router")
        print(f"{'=' * 70}")
        print(f"  error={error}")
        print(f"  规则: if error → bug_report_node, else → END")
        print(f"  结果: → 进入 Bug 报告流程")

        # =============================================================
        # 阶段3：生成 Bug 报告（对应 bug_report_node）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段3】生成 Bug 报告 — LLM 调用")
        print(f"{'=' * 70}")

        bug_prompt = (
            f"你是一个 Bug 分析专家。请为以下 Python 函数生成一份 Bug 报告。\n"
            f"函数代码:\n{func_string}\n"
            f"错误信息: {error_description}\n"
            f"请用中文回答，包含：错误原因、触发条件、建议修复方向。"
        )

        print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
        print(f"  {'-' * 60}")
        print(f"  [user]: {bug_prompt}")
        print(f"  {'-' * 60}")

        bug_report = call_llm([{"role": "user", "content": bug_prompt}])

        print(f"\n  >>> LLM 响应（Bug 报告）<<<")
        print(f"  {'-' * 60}")
        print(f"  {bug_report}")
        print(f"  {'-' * 60}")

        # =============================================================
        # 阶段4：搜索历史记忆（对应 memory_search_node）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段4】搜索历史 Bug 记忆（ChromaDB 向量搜索）")
        print(f"{'=' * 70}")

        summary_prompt = (
            f"请将以下 Bug 报告压缩为一行简洁摘要，用于存档检索。\n"
            f"Bug 报告: {bug_report}\n"
            f"格式: # 函数名 ## 错误描述 ### 分析"
        )

        print(f"  先用 LLM 摘要 Bug 报告，提高搜索精度...")
        summary = call_llm([{"role": "user", "content": summary_prompt}])
        print(f"  摘要: {summary[:100]}...")

        print(f"\n  执行 ChromaDB 向量查询: collection.query(query_texts=[摘要])")
        results = collection.query(query_texts=[summary])

        memory_search_results = []
        memory_ids_to_update = []

        if results["ids"][0]:
            print(f"  找到 {len(results['ids'][0])} 条历史记录:")
            memory_search_results = [
                {
                    "id": results["ids"][0][idx],
                    "memory": results["documents"][0][idx],
                    "distance": results["distances"][0][idx]
                }
                for idx in range(len(results["ids"][0]))
            ]
            for mem in memory_search_results:
                print(f"    - 距离: {mem['distance']:.4f} | {mem['memory'][:80]}...")
        else:
            print(f"  未找到相似历史 Bug（数据库为空或无相似记录）")

        # =============================================================
        # 阶段5：路由决策（对应 memory_filter_router）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【路由决策】memory_filter_router")
        print(f"{'=' * 70}")
        print(f"  memory_search_results 数量: {len(memory_search_results)}")

        if memory_search_results:
            print(f"  结果: → 进入记忆过滤")

            # 阶段5a：过滤记忆（对应 memory_filter_node）
            print(f"\n  【记忆过滤】筛选距离 < 0.3 的高相关记录")
            for mem in memory_search_results:
                if mem["distance"] < 0.3:
                    memory_ids_to_update.append(mem["id"])
                    print(f"    选中: {mem['id'][:8]}... (距离: {mem['distance']:.4f})")

            if not memory_ids_to_update:
                print(f"    无高相关记录")

            # 路由决策（memory_generation_router）
            print(f"\n  【路由决策】memory_generation_router")
            print(f"  需要更新的记忆: {len(memory_ids_to_update)}")

            if memory_ids_to_update:
                # 更新已有记忆
                while memory_ids_to_update:
                    mid = memory_ids_to_update.pop(0)
                    old_doc = collection.get(ids=[mid])["documents"][0]
                    merge_prompt = (
                        f"请合并以下两条 Bug 报告为一条更完整的记录。\n"
                        f"当前 Bug: {bug_report}\n"
                        f"历史 Bug: {old_doc}\n"
                        f"格式: # 函数名 ## 错误描述 ### 分析"
                    )
                    updated = call_llm([{"role": "user", "content": merge_prompt}])
                    collection.update(ids=[mid], documents=[updated])
                    print(f"    已更新记忆: {mid[:8]}...")
            else:
                # 保存新记忆
                record_id = str(uuid.uuid4())
                collection.add(ids=[record_id], documents=[summary])
                print(f"    已保存新记忆: {record_id[:8]}...")
        else:
            print(f"  结果: → 直接保存新记忆")
            record_id = str(uuid.uuid4())
            collection.add(ids=[record_id], documents=[summary])
            print(f"    已保存新记忆: {record_id[:8]}...")

        # =============================================================
        # 阶段6：LLM 生成修复代码（对应 code_update_node）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段6】LLM 生成修复代码")
        print(f"{'=' * 70}")

        fix_prompt = (
            f"你需要修复一个有错误的 Python 函数。\n"
            f"函数代码:\n{func_string}\n"
            f"错误信息: {error_description}\n"
            f"要求:\n"
            f"1. 只修复当前错误，优雅处理异常情况（返回错误消息，不要 raise）\n"
            f"2. 函数名和参数必须完全相同\n"
            f"3. 只输出函数定义代码，不要任何额外文字、代码块标记或语言声明"
        )

        print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
        print(f"  {'-' * 60}")
        print(f"  [user]: {fix_prompt}")
        print(f"  {'-' * 60}")

        new_code = call_llm([{"role": "user", "content": fix_prompt}], temperature=0.3)

        # 清理代码块标记
        new_code = re.sub(r'^```(?:python)?\s*', '', new_code)
        new_code = re.sub(r'\s*```$', '', new_code)

        print(f"\n  >>> LLM 响应（修复后的代码）<<<")
        print(f"  {'-' * 60}")
        print(f"  {new_code}")
        print(f"  {'-' * 60}")

        # =============================================================
        # 阶段7：热补丁（对应 code_patching_node）
        # =============================================================
        print(f"\n{'=' * 70}")
        print(f"【阶段7】热补丁 — exec() 动态替换函数")
        print(f"{'=' * 70}")
        print(f"  这就是 Python 的动态能力：exec() 将字符串编译为可执行函数")
        print(f"  Java 类比：类似于热部署 / 字节码增强（Javassist/ByteBuddy）")

        try:
            namespace = {}
            exec(new_code, namespace)
            func = namespace[func_name]
            func_string = new_code
            print(f"  补丁应用成功，即将重新执行...")
        except Exception as e:
            print(f"  补丁失败: {e}")
            print(f"  将在下一轮重试...")

        # 循环回到阶段1（code_execution_node）
        print(f"\n  → 循环回到【阶段1】重新执行（这就是 LangGraph 的自愈循环边）")

    print(f"\n  达到最大修复次数({MAX_HEAL_ATTEMPTS})，停止尝试。")
    return None


def main():
    print("=" * 70)
    print("  自愈代码代理系统（透明调试版 - 无框架）")
    print("=" * 70)

    # 测试函数1：除以零
    def divide_two_numbers(a, b):
        return a / b

    # 测试函数2：列表索引越界
    def process_list(lst, index):
        return lst[index] * 2

    # 测试函数3：日期格式错误
    def parse_date(date_string):
        year, month, day = date_string.split("-")
        return {"year": int(year), "month": int(month), "day": int(day)}

    print("\n" + "*" * 70)
    print("  测试1：除以零 — divide_two_numbers(10, 0)")
    print("*" * 70)
    self_healing_code(divide_two_numbers, [10, 0])

    print("\n" + "*" * 70)
    print("  测试2：列表越界 — process_list([1,2,3], 5)")
    print("*" * 70)
    self_healing_code(process_list, [[1, 2, 3], 5])

    print("\n" + "*" * 70)
    print("  测试3：日期格式 — parse_date('2024/01/01')")
    print("*" * 70)
    self_healing_code(parse_date, ["2024/01/01"])

    # 调试总结
    print("\n" + "=" * 70)
    print("  调试总结：LangGraph 在本课做了什么")
    print("=" * 70)
    print("  1. StateGraph 定义了 State，含函数引用、错误信息、Bug报告、记忆等")
    print("  2. 8 个节点组成自愈工作流：执行→报告→搜索→过滤→保存/更新→修复→补丁→执行")
    print("  3. 4 个条件路由控制分支：")
    print("     - error_router: 有错误 → Bug报告，无错误 → 结束")
    print("     - memory_filter_router: 有搜索结果 → 过滤，无 → 直接保存")
    print("     - memory_generation_router: 有需更新 → 更新记忆，无 → 保存新记忆")
    print("     - memory_update_router: 还有待更新 → 继续，无 → 进入修复")
    print("  4. code_patching_node → code_execution_node 形成自愈循环边")
    print("  5. ChromaDB 向量数据库用于存储和检索历史 Bug 报告")
    print("  6. 本质：while True + try/except + exec() + 向量搜索")
    print("     框架把这些组织成了一个可追踪、可视化的图结构")


if __name__ == "__main__":
    main()
