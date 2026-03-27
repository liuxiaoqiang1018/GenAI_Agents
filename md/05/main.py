"""
第5课 - MCP 协议（框架版：使用 mcp 库 + OpenAI 兼容 API）

核心概念：MCP Server 暴露工具，MCP Client 发现和调用工具，Host 用 LLM 决策
本文件实现 Host + Client，连接 mcp_server.py（需要先确保它在同目录下）。

运行方式：python main.py
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE_URL'),
    default_headers={"User-Agent": "Mozilla/5.0"},
)
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

# MCP Server 路径
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcp_server.py')


# ========== 第1步：工具发现（MCP Client → Server） ==========

async def discover_tools() -> List[Dict]:
    """
    连接 MCP Server，发现可用的工具。

    MCP 协议流程：
    1. 启动 Server 子进程
    2. 通过 stdio 建立连接
    3. 发送 initialize 握手
    4. 调用 list_tools() 获取工具列表
    """
    server_params = StdioServerParameters(
        command=sys.executable,  # 用当前 Python 解释器
        args=[MCP_SERVER_PATH],
    )

    print('=' * 50)
    print('【工具发现阶段】连接 MCP Server...')
    print('=' * 50)
    print(f'>>> Server 路径: {MCP_SERVER_PATH}')
    print(f'>>> 启动命令: {sys.executable} {MCP_SERVER_PATH}')
    print()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print('>>> 发送 initialize 握手...')
            await session.initialize()
            print('>>> 握手成功！')
            print()

            print('>>> 调用 list_tools() 发现工具...')
            tools_result = await session.list_tools()

            tool_info = []
            for tool_type, tool_list in tools_result:
                if tool_type == "tools":
                    for tool in tool_list:
                        info = {
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema,
                        }
                        tool_info.append(info)
                        print(f'>>> 发现工具: {tool.name}')
                        print(f'    描述: {tool.description[:80]}...')
                        print(f'    参数: {json.dumps(tool.inputSchema, ensure_ascii=False, indent=2)[:200]}')
                        print()

            print(f'>>> 共发现 {len(tool_info)} 个工具')
            print()
            return tool_info


# ========== 第2步：工具执行（MCP Client → Server） ==========

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    通过 MCP 协议调用 Server 上的工具。

    每次调用都新建连接（简单但不高效，生产环境应复用连接）。
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
    )

    print(f'>>> [MCP 调用] 工具: {tool_name}')
    print(f'>>> [MCP 调用] 参数: {json.dumps(arguments, ensure_ascii=False)}')

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            result_text = str(result)
            print(f'>>> [MCP 返回] {result_text[:300]}')
            print()
            return result_text


# ========== 第3步：LLM 推理 + 工具调用（Host 层） ==========

def build_tools_for_llm(tool_info: List[Dict]) -> list:
    """把 MCP 发现的工具转换成 OpenAI function calling 格式"""
    tools = []
    for info in tool_info:
        tools.append({
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["schema"],
            }
        })
    return tools


async def query_llm(prompt: str, tool_info: List[Dict], messages: list = None) -> tuple:
    """
    Host 层：发送用户问题给 LLM，处理工具调用。

    流程：
    1. 把 MCP 工具描述 → 转成 OpenAI tools 格式 → 发给 LLM
    2. LLM 决定是否调用工具
    3. 如果调用工具 → 通过 MCP Client 执行 → 把结果返回给 LLM
    4. LLM 生成最终回答
    """
    if messages is None:
        messages = []

    openai_tools = build_tools_for_llm(tool_info)

    system_prompt = """你是一个AI助手，可以通过 MCP 协议使用外部工具。
当用户问到城市信息或单位换算时，请调用对应的工具来获取准确数据。
回答要简洁，用中文。"""

    print()
    print('=' * 50)
    print('【LLM 推理阶段】')
    print('=' * 50)
    print(f'>>> 用户问题: {prompt}')
    print(f'>>> 可用工具: {[t["name"] for t in tool_info]}')
    print()

    # 构建消息
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)
    full_messages.append({"role": "user", "content": prompt})

    # 第一次调用 LLM
    print('>>> 第1次调用 LLM（让它决定是否使用工具）...')
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=full_messages,
        tools=openai_tools,
        temperature=0,
    )

    choice = response.choices[0]
    print(f'>>> finish_reason: {choice.finish_reason}')
    print(f'>>> Token 用量: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}')

    # 如果 LLM 不需要工具，直接返回
    if choice.finish_reason == 'stop':
        answer = choice.message.content
        print(f'>>> LLM 直接回答（不需要工具）: {answer}')
        full_messages.append({"role": "assistant", "content": answer})
        return answer, full_messages

    # 如果 LLM 要调用工具
    if choice.message.tool_calls:
        # 把 assistant 消息加入历史
        full_messages.append(choice.message.model_dump())

        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print()
            print(f'>>> LLM 决定调用工具: {func_name}')
            print(f'>>> 参数: {json.dumps(func_args, ensure_ascii=False)}')
            print()

            # 通过 MCP 执行工具
            print('--- MCP 工具执行 ---')
            tool_result = await execute_tool(func_name, func_args)

            # 把工具结果加入消息
            full_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # 第二次调用 LLM，让它根据工具结果生成回答
        print('>>> 第2次调用 LLM（根据工具结果生成回答）...')
        final_response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            tools=openai_tools,
            temperature=0,
        )

        final_answer = final_response.choices[0].message.content
        print(f'>>> LLM 最终回答: {final_answer}')
        full_messages.append({"role": "assistant", "content": final_answer})
        return final_answer, full_messages

    # fallback
    answer = choice.message.content or "（无响应）"
    return answer, full_messages


# ========== 运行 ==========

async def main():
    print('第5课 - MCP 协议演示')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()

    # 第1步：发现工具
    tool_info = await discover_tools()

    # 第2步：示例问题
    example_questions = [
        "北京有哪些著名景点？",
        "100 人民币等于多少美元？",
        "深圳的人口有多少？",
    ]

    print('\n--- 示例问题 ---')
    for q in example_questions:
        answer, _ = await query_llm(q, tool_info)
        print()
        print('-' * 50)
        print(f'问: {q}')
        print(f'答: {answer}')
        print('-' * 50)
        print()

    # 第3步：交互模式
    print('\n输入问题，输入 /quit 退出\n')
    messages = []
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break

        answer, messages = await query_llm(user_input, tool_info, messages)
        print()
        print(f'AI: {answer}')
        print()


if __name__ == '__main__':
    asyncio.run(main())
