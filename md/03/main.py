"""
第3课 - 简单数据分析代理（使用 PydanticAI 框架）

核心概念：Agent + 自定义工具 + 依赖注入 + 重试机制
Agent 自己生成 Pandas 查询代码，执行后返回自然语言回答。
"""

import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 加载环境变量
load_dotenv()
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

# 固定随机种子，保证每次生成的数据一致
np.random.seed(42)


# ========== 生成示例数据 ==========

def create_sample_data() -> pd.DataFrame:
    """生成 1000 条汽车销售模拟数据"""
    n_rows = 1000

    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    brands = ['丰田', '本田', '福特', '雪佛兰', '日产', '宝马', '奔驰', '奥迪', '现代', '起亚']
    models = ['轿车', 'SUV', '皮卡', '掀背车', '跑车', '面包车']
    colors = ['红色', '蓝色', '黑色', '白色', '银色', '灰色', '绿色']
    salespersons = ['张伟', '李娜', '王强', '刘洋', '陈静']

    data = {
        '日期': dates,
        '品牌': np.random.choice(brands, n_rows),
        '车型': np.random.choice(models, n_rows),
        '颜色': np.random.choice(colors, n_rows),
        '年份': np.random.randint(2015, 2023, n_rows),
        '价格': np.random.uniform(100000, 500000, n_rows).round(2),  # 人民币
        '里程': np.random.uniform(0, 100000, n_rows).round(0),
        '排量': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
        '油耗': np.random.uniform(5, 15, n_rows).round(1),  # L/100km
        '销售员': np.random.choice(salespersons, n_rows),
    }

    return pd.DataFrame(data).sort_values('日期')


# ========== 依赖注入：把 DataFrame 传给工具 ==========

@dataclass
class Deps:
    """工具的依赖，包含要查询的 DataFrame"""
    df: pd.DataFrame


# ========== 初始化模型和 Agent ==========

model = OpenAIChatModel(
    os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    provider=OpenAIProvider(
        base_url=os.getenv('API_BASE_URL'),
        api_key=os.getenv('API_KEY'),
    ),
)

agent = Agent(
    model=model,
    deps_type=Deps,
    retries=10,  # 允许重试 10 次（LLM 生成的查询可能出错）
)


@agent.system_prompt
def build_system_prompt(ctx: RunContext[Deps]) -> str:
    """动态生成 system prompt，把 DataFrame 的信息告诉 LLM"""
    df = ctx.deps.df
    return f"""你是一个数据分析AI助手，帮助用户从 pandas DataFrame（变量名为 df）中提取信息。

重要规则：
- 必须使用 df_query 工具来查询数据，不要凭空回答
- 查询语法必须是 pandas 表达式（不是 SQL！）
- 例如：df.shape[0]、df['价格'].mean()、df.columns.tolist()

当前 DataFrame 信息：
- 行数: {len(df)}
- 列名: {list(df.columns)}
- 数据类型:
{df.dtypes.to_string()}

前3行示例数据:
{df.head(3).to_string()}

回答要简洁，用中文。"""


# ========== 自定义工具：查询 DataFrame ==========

@agent.tool
async def df_query(ctx: RunContext[Deps], query: str) -> str:
    """用于查询 pandas DataFrame 的工具。
    query 会通过 pd.eval(query, target=df) 执行，必须是 pandas.eval 兼容的语法。
    """
    print(f'  [工具调用] 执行查询: `{query}`')
    try:
        result = str(pd.eval(query, target=ctx.deps.df))
        print(f'  [工具返回] {result[:200]}{"..." if len(result) > 200 else ""}')
        return result
    except Exception as e:
        print(f'  [工具出错] {e} → 触发重试')
        raise ModelRetry(f'查询 `{query}` 无效，原因: `{e}`') from e


# ========== 问答函数 ==========

def ask_agent(question: str, df: pd.DataFrame) -> str:
    """向数据分析代理提问"""
    deps = Deps(df=df)

    print()
    print('=' * 50)
    print(f'问题: {question}')
    print('=' * 50)

    response = agent.run_sync(question, deps=deps)

    # 提取最终回答
    answer = response.output
    print(f'\n回答: {answer}')
    print('-' * 50)

    return answer


# ========== 运行 ==========

if __name__ == '__main__':
    print(f'当前模型: {os.getenv("MODEL_NAME")}')
    print(f'API地址: {os.getenv("API_BASE_URL")}')
    print()

    # 生成示例数据
    df = create_sample_data()
    print(f'已生成 {len(df)} 条汽车销售数据')
    print(f'列名: {list(df.columns)}')
    print()

    # 示例问题
    example_questions = [
        "这个数据集有哪些列？",
        "数据集有多少行？",
        "汽车的平均售价是多少？",
    ]

    print('--- 示例问题 ---\n')
    for q in example_questions:
        ask_agent(q, df)

    # 交互模式
    print('\n输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        ask_agent(user_input, df)
