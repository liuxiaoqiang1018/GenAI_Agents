"""
第3课 - 数据分析代理内部机制演示（不使用任何框架）

目的：让你看清 Agent + 工具 + 重试 的完整内部流程：
  1. Agent 拼装 Prompt（包含工具描述 + DataFrame 信息）发给 LLM
  2. LLM 决策：生成查询代码
  3. Agent 执行查询，可能出错
  4. 出错时：把错误信息拼回 Prompt，让 LLM 重试（这就是 ModelRetry 的本质）
  5. 成功后：把结果拼回 Prompt，LLM 生成自然语言回答

对比 main.py（框架版），理解 PydanticAI 在这一步到底封装了什么。
"""

import os
import json
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化 API 客户端
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE_URL'),
    default_headers={"User-Agent": "Mozilla/5.0"},
)
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

np.random.seed(42)


# ================================================================
#  【数据层】生成示例数据
# ================================================================

def create_sample_data() -> pd.DataFrame:
    n_rows = 1000
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {
        '日期': dates,
        '品牌': np.random.choice(['丰田', '本田', '福特', '雪佛兰', '日产', '宝马', '奔驰', '奥迪', '现代', '起亚'], n_rows),
        '车型': np.random.choice(['轿车', 'SUV', '皮卡', '掀背车', '跑车', '面包车'], n_rows),
        '颜色': np.random.choice(['红色', '蓝色', '黑色', '白色', '银色', '灰色', '绿色'], n_rows),
        '年份': np.random.randint(2015, 2023, n_rows),
        '价格': np.random.uniform(100000, 500000, n_rows).round(2),
        '里程': np.random.uniform(0, 100000, n_rows).round(0),
        '排量': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
        '油耗': np.random.uniform(5, 15, n_rows).round(1),
        '销售员': np.random.choice(['张伟', '李娜', '王强', '刘洋', '陈静'], n_rows),
    }

    return pd.DataFrame(data).sort_values('日期')


# ================================================================
#  【Tool 层】—— 执行 DataFrame 查询
# ================================================================

def df_query_tool(df: pd.DataFrame, query: str) -> dict:
    """
    执行查询并返回结果或错误。

    Java 类比：
        public Result executeQuery(DataFrame df, String query) {
            try {
                return Result.success(df.eval(query));
            } catch (Exception e) {
                return Result.failure(e.getMessage());
            }
        }
    """
    try:
        result = str(pd.eval(query, target=df))
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ================================================================
#  【LLM 层】—— 调用大模型 API
# ================================================================

def call_llm(messages: list[dict], tools: list[dict] = None) -> dict:
    """调用 LLM API，支持工具调用"""
    kwargs = {
        'model': MODEL_NAME,
        'messages': messages,
        'temperature': 0,
    }
    if tools:
        kwargs['tools'] = tools

    response = client.chat.completions.create(**kwargs)
    return response


# ================================================================
#  【Agent 层】—— 完整的 Agent 运行逻辑
# ================================================================

# 工具定义（OpenAI function calling 格式）
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'df_query',
            'description': '用于查询 pandas DataFrame 的工具。query 会通过 pd.eval(query, target=df) 执行，必须是 pandas.eval 兼容的语法。',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': '要执行的 pandas 查询表达式',
                    }
                },
                'required': ['query'],
            }
        }
    }
]


def agent_run(question: str, df: pd.DataFrame, max_retries: int = 10) -> str:
    """
    Agent 的核心运行逻辑。

    这就是 PydanticAI 的 agent.run_sync() 内部做的事情：
    1. 拼装 messages（system + tools + user）
    2. 调用 LLM，看是否要调用工具
    3. 如果要调用工具：执行工具 → 结果或错误拼回 messages → 再调 LLM
    4. 重复直到 LLM 直接返回文本回答（或重试次数用完）
    """

    # 构建消息列表
    system_msg = f"""你是一个AI助手，帮助用户从 pandas DataFrame 中提取信息。
如果用户问到列名，请先检查实际的列名。回答要简洁。

DataFrame 信息：
- 行数: {len(df)}
- 列名: {list(df.columns)}
- 数据类型: {dict(df.dtypes.astype(str))}"""

    messages = [
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': question},
    ]

    # ==========================================
    # 第1步：打印初始 Prompt
    # ==========================================
    print()
    print('=' * 60)
    print('【1️⃣  初始 Prompt - 发给 LLM 的完整内容】')
    print('=' * 60)
    print()
    print('>>> System Prompt:')
    print('-' * 40)
    print(system_msg)
    print('-' * 40)
    print()
    print('>>> 工具定义（告诉 LLM 有哪些工具可用）:')
    print('-' * 40)
    print(json.dumps(TOOLS, indent=2, ensure_ascii=False))
    print('-' * 40)
    print()
    print(f'>>> 用户问题: {question}')
    print()

    # ==========================================
    # 循环：调用 LLM → 执行工具 → 重试
    # ==========================================
    for attempt in range(max_retries + 1):

        print('=' * 60)
        print(f'【2️⃣  调用 LLM（第 {attempt + 1} 次）】')
        print('=' * 60)
        print()

        response = call_llm(messages, tools=TOOLS)
        choice = response.choices[0]

        print(f'>>> finish_reason: {choice.finish_reason}')
        print(f'>>> token用量: prompt={response.usage.prompt_tokens}, '
              f'completion={response.usage.completion_tokens}')
        print()

        # 情况1：LLM 直接返回文本（不调用工具）→ 结束
        if choice.finish_reason == 'stop':
            final_answer = choice.message.content
            print('>>> LLM 直接返回了文本回答（不需要工具）:')
            print(f'    {final_answer}')
            print()
            return final_answer

        # 情况2：LLM 要调用工具
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            query = func_args.get('query', '')

            print(f'>>> LLM 决定调用工具: {func_name}')
            print(f'>>> LLM 生成的查询: `{query}`')
            print()

            # ==========================================
            # 第3步：执行工具
            # ==========================================
            print('=' * 60)
            print(f'【3️⃣  执行工具 df_query（第 {attempt + 1} 次尝试）】')
            print('=' * 60)
            print()

            tool_result = df_query_tool(df, query)

            if tool_result['success']:
                result_text = tool_result['result']
                print(f'>>> 执行成功！')
                print(f'>>> 结果: {result_text[:300]}{"..." if len(result_text) > 300 else ""}')
                print()

                # 把工具调用和结果拼回 messages
                messages.append(choice.message.model_dump())
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result_text,
                })

                print('=' * 60)
                print('【4️⃣  把工具结果拼回 Prompt，再调 LLM 生成最终回答】')
                print('=' * 60)
                print()
                print('>>> 新增的 messages:')
                print(f'    assistant: 调用 {func_name}({query})')
                print(f'    tool: {result_text[:200]}')
                print()

                # 再调一次 LLM 生成最终回答
                final_response = call_llm(messages, tools=TOOLS)
                final_choice = final_response.choices[0]

                # 检查是否还要调用工具（有些复杂问题需要多次工具调用）
                if final_choice.message.tool_calls:
                    # 继续循环处理
                    messages.append(final_choice.message.model_dump())
                    tc = final_choice.message.tool_calls[0]
                    next_query = json.loads(tc.function.arguments).get('query', '')
                    print(f'>>> LLM 还需要再调一次工具: `{next_query}`')

                    next_result = df_query_tool(df, next_query)
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': next_result['result'] if next_result['success'] else f"错误: {next_result['error']}",
                    })
                    continue

                final_answer = final_choice.message.content
                print(f'>>> LLM 最终回答: {final_answer}')
                print()
                return final_answer

            else:
                # ==========================================
                # 工具出错 → 这就是 ModelRetry 的本质！
                # ==========================================
                error_msg = tool_result['error']
                print(f'>>> 执行失败！错误: {error_msg}')
                print()
                print('=' * 60)
                print(f'【🔄 重试机制 - 这就是 ModelRetry 的本质！】')
                print(f'    框架做的事：把错误信息拼回 Prompt，让 LLM 换种写法')
                print(f'    第 {attempt + 1}/{max_retries} 次重试')
                print('=' * 60)
                print()

                # 把工具调用和错误结果拼回 messages（这就是 ModelRetry 做的事）
                messages.append(choice.message.model_dump())
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': f'查询 `{query}` 无效，原因: {error_msg}。请换种写法重试。',
                })

    return '重试次数用完，未能获取答案。'


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第3课 - 数据分析代理内部机制演示')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示 PydanticAI 的 Agent + Tool + ModelRetry 内部做了什么。')
    print()

    df = create_sample_data()
    print(f'已生成 {len(df)} 条汽车销售数据')
    print(f'列名: {list(df.columns)}')
    print()

    # 示例问题
    example_questions = [
        "这个数据集有哪些列？",
        "汽车的平均售价是多少？",
    ]

    print('--- 示例问题 ---')
    for q in example_questions:
        print()
        print('#' * 60)
        print(f'#  用户提问: {q}')
        print('#' * 60)

        result = agent_run(q, df)

        print('=' * 60)
        print('【5️⃣  最终返回给用户的回复】')
        print('=' * 60)
        print(f'AI: {result}')
        print()

    # 交互模式
    print('\n输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        print()
        print('#' * 60)
        print(f'#  用户提问: {user_input}')
        print('#' * 60)

        result = agent_run(user_input, df)

        print('=' * 60)
        print('【5️⃣  最终返回给用户的回复】')
        print('=' * 60)
        print(f'AI: {result}\n')
