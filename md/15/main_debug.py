"""
第15课 - E2E测试代理内部机制（不使用 LangGraph）

目的：让你看清动作循环+代码生成验证的本质：
  1. 指令拆解 = 一次LLM调用，返回动作列表
  2. 动作循环 = for i in range(len(actions))
  3. 代码生成 = 每个动作一次LLM调用
  4. 语法验证 = ast.parse()
  5. 三路分支 = if error: break; elif done: 组装; else: continue
  6. 执行 = exec() 运行生成的代码

对比 main.py（LangGraph 框架版），理解：
  - current_action 计数器 → 就是 for 循环的 i
  - add_conditional_edges(三路) → 就是 if-elif-else
  - 验证后回到生成 → 就是循环体的 continue
  - 错误处理 → 就是 break + 输出报错
"""

import os
import ast
import json
import re
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
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
                print(f'    ⚠ 重试({attempt+1}/{MAX_RETRIES}): {e}')
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
    match = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ================================================================
#  完整流程
# ================================================================

def generate_tests(query: str) -> dict:
    """
    E2E测试代理的完整流程。

    Java 类比：
        @Service
        public class TestGeneratorService {
            public TestReport generate(String testDescription) {
                // 1. 拆解指令
                List<String> actions = parser.parse(testDescription);

                // 2. 逐个动作生成代码（带验证的for循环）
                List<String> codeBlocks = new ArrayList<>();
                for (int i = 0; i < actions.size(); i++) {
                    String code = codeGen.generate(actions.get(i), codeBlocks);
                    // 3. 验证
                    if (!syntaxChecker.isValid(code)) {
                        return TestReport.failed("语法错误", i);
                    }
                    codeBlocks.add(code);
                }

                // 4. 组装+执行+报告
                String script = assembler.assemble(codeBlocks);
                String result = executor.run(script);
                return TestReport.success(actions, script, result);
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：指令拆解
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：指令拆解】（LLM调用 #1）')
    print('=' * 60)

    system = ("你是测试专家。把测试描述拆解为原子动作列表。\n"
              "返回JSON：{\"actions\": [\"动作1\", \"动作2\"]}")
    result = extract_json(call_llm(query, system))
    actions = result.get("actions", ["执行基本测试"])
    total_llm_calls += 1

    print(f'>>> 拆解为 {len(actions)} 个动作:')
    for i, a in enumerate(actions, 1):
        print(f'    {i}. {a}')

    # ==========================================
    # 第2-3步：动作循环（这就是 LangGraph 图循环的本质）
    # ==========================================
    print()
    print('*' * 60)
    print('*  动作循环开始（就是 for + if-break）')
    print('*' * 60)

    code_blocks = []
    error_message = ""

    for idx in range(len(actions)):
        action = actions[idx]

        # 第2步：生成代码
        print()
        print(f'    【生成代码】动作 {idx + 1}/{len(actions)}: {action}')

        prev_code = "\n".join(code_blocks) if code_blocks else "（无）"
        system = ("你是Python测试代码生成专家。只生成当前动作的代码片段。\n"
                  "用assert做断言，用注释说明。只返回代码。\n"
                  "用 ```python ``` 包裹。")
        prompt = (f"测试描述: {query}\n当前动作: {action}\n已有代码:\n{prev_code}")

        response = call_llm(prompt, system)
        code = extract_code(response)
        total_llm_calls += 1

        print(f'    >>> 生成 {code.count(chr(10)) + 1} 行代码')

        # 第3步：语法验证
        print(f'    【语法验证】', end=' ')
        try:
            ast.parse(code)
            print('✓ 通过')
        except SyntaxError as e:
            error_message = f"动作{idx + 1}语法错误: {e}"
            print(f'✗ {error_message}')

            # 三路分支 → 错误：break退出
            print()
            print(f'    【三路分支】→ 错误，break退出循环')
            break

        code_blocks.append(code)

        # 三路分支 → 还有动作：continue
        if idx < len(actions) - 1:
            print(f'    【三路分支】→ 还有动作，continue继续循环')
        else:
            print(f'    【三路分支】→ 所有动作完成，退出循环')

    print()
    print(f'*  动作循环结束，生成了 {len(code_blocks)} 个代码块')
    print('*' * 60)

    # 如果有错误
    if error_message:
        print()
        print('=' * 60)
        print('【错误处理】')
        print('=' * 60)
        print(f'>>> {error_message}')
        return {
            "success": False,
            "error": error_message,
            "actions": actions,
            "code_blocks": code_blocks,
            "total_llm_calls": total_llm_calls,
        }

    # ==========================================
    # 第4步：组装测试脚本
    # ==========================================
    print()
    print('=' * 60)
    print('【第4步：组装测试脚本】')
    print('=' * 60)

    all_code = "\n\n".join(code_blocks)
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in all_code.split("\n"))
    test_script = (f"def test_generated():\n"
                   f"    \"\"\"自动生成: {query[:50]}\"\"\"\n"
                   f"{indented}\n"
                   f"    print('所有测试步骤通过！')\n")

    print(f'>>> 脚本:')
    for line in test_script.split('\n')[:12]:
        print(f'    {line}')
    if test_script.count('\n') > 12:
        print(f'    ... (共{test_script.count(chr(10)) + 1}行)')

    # ==========================================
    # 第5步：执行测试
    # ==========================================
    print()
    print('=' * 60)
    print('【第5步：执行测试】（exec运行生成的代码）')
    print('=' * 60)

    try:
        exec_namespace = {}
        exec(test_script, exec_namespace)
        test_func = exec_namespace.get("test_generated")
        if test_func:
            test_func()
            test_result = "✓ 测试通过"
        else:
            test_result = "✗ 未找到测试函数"
    except AssertionError as e:
        test_result = f"✗ 断言失败: {e}"
    except Exception as e:
        test_result = f"✗ 执行错误: {type(e).__name__}: {e}"

    print(f'>>> 结果: {test_result}')

    # ==========================================
    # 第6步：生成报告
    # ==========================================
    print()
    print('=' * 60)
    print('【第6步：测试报告】')
    print('=' * 60)
    print(f'  测试描述: {query}')
    print(f'  动作数: {len(actions)}')
    print(f'  代码块: {len(code_blocks)}个')
    print(f'  结果: {test_result}')
    print(f'  LLM调用: {total_llm_calls}次')

    return {
        "success": True,
        "actions": actions,
        "code_blocks": code_blocks,
        "test_script": test_script,
        "test_result": test_result,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第15课 - E2E测试代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 动作循环 = for i in range(len(actions))')
    print('  - 代码生成 = 每个动作一次LLM调用')
    print('  - 语法验证 = ast.parse()（Python内置）')
    print('  - 三路分支 = if error: break; elif done: 结束; else: continue')
    print('  - 执行测试 = exec() 运行生成的代码')
    print()

    examples = [
        "测试计算器：加法(1+2=3)、减法(5-3=2)、乘法(4*3=12)、除以零异常",
        "测试用户注册：用户名不能为空、密码至少8位、两次密码必须一致",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    query = input('测试描述（回车用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    result = generate_tests(query)

    print()
    print('#' * 60)
    if result["success"]:
        print(f'#  测试生成成功！LLM调用 {result["total_llm_calls"]} 次')
    else:
        print(f'#  测试生成失败: {result["error"]}')
    print('#' * 60)
