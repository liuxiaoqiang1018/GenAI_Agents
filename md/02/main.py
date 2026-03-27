"""
第2课 - 简单问答代理（使用 LangChain 框架）

核心概念：PromptTemplate + LLM = 最简单的 Agent
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

# ========== 初始化语言模型 ==========

llm = ChatOpenAI(
    model=os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    base_url=os.getenv('API_BASE_URL'),
    api_key=os.getenv('API_KEY'),
    max_tokens=1000,
    temperature=0,
    use_responses_api=False,
    default_headers={"User-Agent": "Mozilla/5.0"},  # 第三方API可能屏蔽SDK默认UA
)

# ========== 定义提示模板 ==========

template = """
你是一个有帮助的AI助手。请尽你所能回答用户的问题。

用户的问题：{question}

请提供清晰简洁的回答：
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# ========== 创建 Chain ==========
# prompt | llm 相当于：先格式化模板，再发给 LLM
qa_chain = prompt | llm


# ========== 问答函数 ==========

def get_answer(question: str) -> str:
    """调用 Chain 获取回答，并打印中间过程"""
    input_variables = {"question": question}

    # 1. 打印模板格式化后的完整 Prompt（这就是实际发给大模型的内容）
    formatted = prompt.format(**input_variables)
    print()
    print('=' * 50)
    print('【发给大模型的完整 Prompt】')
    print('=' * 50)
    print(formatted)
    print('=' * 50)

    # 2. 调用 Chain，拿到响应
    response = qa_chain.invoke(input_variables)

    # 3. 打印大模型返回的原始响应
    print()
    print('=' * 50)
    print('【大模型返回的原始响应】')
    print('=' * 50)
    print(f'  类型: {type(response).__name__}')
    print(f'  content: {response.content}')
    print(f'  token用量: {response.usage_metadata}')
    print('=' * 50)
    print()

    return response.content


# ========== 运行 ==========

if __name__ == '__main__':
    print(f'当前模型: {os.getenv("MODEL_NAME")}')
    print(f'API地址: {os.getenv("API_BASE_URL")}')
    print('输入 /quit 退出\n')

    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        answer = get_answer(user_input)
        print(f'AI: {answer}\n')
