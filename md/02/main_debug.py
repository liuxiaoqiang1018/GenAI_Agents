"""
第2课 - 简单问答代理内部机制演示（不使用任何框架）

目的：让你看清最简单的 Agent 内部做了什么：
  1. PromptTemplate 本质就是字符串格式化
  2. Chain 本质就是 "格式化 → 调用API → 提取结果"
  3. 没有工具、没有历史记录，就是最纯粹的 "拼Prompt → 调API"

对比 main.py（框架版），理解 LangChain 在这一步到底封装了什么。
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()


# ================================================================
#  【第1层】Prompt Template —— 本质就是 str.format()
# ================================================================

TEMPLATE = """你是一个有帮助的AI助手。请尽你所能回答用户的问题。

用户的问题：{question}

请提供清晰简洁的回答："""


def format_prompt(question: str) -> str:
    """
    PromptTemplate 的本质：字符串替换。

    Java 类比：
        String template = "...{question}...";
        String prompt = template.replace("{question}", question);
    """
    return TEMPLATE.format(question=question)


# ================================================================
#  【第2层】LLM 调用 —— 本质就是一个 HTTP POST 请求
# ================================================================

# 初始化 OpenAI 客户端（兼容所有 OpenAI 接口格式的服务）
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE_URL'),
    default_headers={"User-Agent": "Mozilla/5.0"},
)

MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str) -> dict:
    """
    直接调用 LLM API，返回原始响应。

    Java 类比：
        HttpResponse response = httpClient.send(
            HttpRequest.newBuilder()
                .uri(URI.create(apiUrl + "/chat/completions"))
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build(),
            HttpResponse.BodyHandlers.ofString()
        );
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0,
    )
    return response


# ================================================================
#  【第3层】Chain —— 本质就是把上面两步串起来
# ================================================================

def qa_chain(question: str) -> str:
    """
    LangChain 的 chain = prompt | llm 到底做了什么？
    就是下面这三步：

    1. 格式化 Prompt
    2. 调用 LLM API
    3. 从响应中提取文本内容

    Java 类比：
        public String qaChain(String question) {
            String prompt = formatPrompt(question);           // step 1
            ApiResponse response = callLlm(prompt);           // step 2
            return response.getChoices()[0].getMessage();      // step 3
        }
    """

    # ==========================================
    # Step 1: PromptTemplate.format()
    # ==========================================
    print()
    print('=' * 60)
    print('【Step 1: PromptTemplate 格式化】')
    print('  框架代码: prompt = PromptTemplate(template=..., input_variables=["question"])')
    print('  本质操作: template.format(question=用户输入)')
    print('=' * 60)

    formatted_prompt = format_prompt(question)

    print()
    print('>>> 格式化后的完整 Prompt:')
    print('-' * 40)
    print(formatted_prompt)
    print('-' * 40)

    # ==========================================
    # Step 2: LLM.invoke()
    # ==========================================
    print()
    print('=' * 60)
    print('【Step 2: 调用 LLM API】')
    print(f'  框架代码: llm = ChatOpenAI(model="{MODEL_NAME}")')
    print('  本质操作: HTTP POST /chat/completions')
    print('=' * 60)

    # 打印实际发送的请求体
    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": formatted_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0,
    }
    print()
    print('>>> 发送给 API 的请求体:')
    print('-' * 40)
    print(json.dumps(request_body, indent=2, ensure_ascii=False))
    print('-' * 40)

    print()
    print('>>> 正在调用 LLM API...')
    response = call_llm(formatted_prompt)

    # 打印原始响应
    print()
    print('>>> LLM API 原始响应:')
    print('-' * 40)
    print(f'  model: {response.model}')
    print(f'  usage: prompt_tokens={response.usage.prompt_tokens}, '
          f'completion_tokens={response.usage.completion_tokens}, '
          f'total_tokens={response.usage.total_tokens}')
    print(f'  finish_reason: {response.choices[0].finish_reason}')
    print(f'  content: {response.choices[0].message.content}')
    print('-' * 40)

    # ==========================================
    # Step 3: 提取 .content
    # ==========================================
    print()
    print('=' * 60)
    print('【Step 3: 提取回答内容】')
    print('  框架代码: response = qa_chain.invoke(input).content')
    print('  本质操作: response.choices[0].message.content')
    print('=' * 60)

    answer = response.choices[0].message.content

    print()
    print(f'>>> 最终提取的回答: {answer}')
    print()

    return answer


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第2课 - 简单问答代理内部机制演示')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示 LangChain 的 PromptTemplate + Chain 内部到底做了什么。')
    print('输入 /quit 退出\n')

    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        result = qa_chain(user_input)

        print('=' * 60)
        print('【最终返回给用户的回复】')
        print('=' * 60)
        print(f'AI: {result}\n')
