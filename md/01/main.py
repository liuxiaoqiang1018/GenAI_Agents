import os
from dotenv import load_dotenv
from itertools import chain

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.agent import AgentRunResult

# 加载环境变量并初始化语言模型
load_dotenv()
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

model = OpenAIChatModel(
    os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    provider=OpenAIProvider(
        base_url=os.getenv('API_BASE_URL'),
        api_key=os.getenv('API_KEY'),
    ),
)

agent = Agent(
    model=model,
    system_prompt='你是一个有帮助的AI助手，请用中文回答。',
)


# ========== Tool：航线计算 ==========

@agent.tool_plain
def calculate_course(distance_meters: int) -> str:
    """根据起航线距离计算航线方案。当用户提到起航线或航线距离时调用此工具。

    Args:
        distance_meters: 起航线距离，单位为米
    """
    return f'[航线计算结果] 起航线距离: {distance_meters}米，建议航向: 185°，预计用时: {distance_meters / 100:.1f}分钟'


# ========== 聊天历史存储 ==========

store: dict[str, list[bytes]] = {}


def create_session_if_not_exists(session_id: str) -> None:
    """确保 session_id 在聊天存储中存在。"""
    # 检查 session_id 是否已存在，不存在则创建空列表
    if session_id not in store:
        store[session_id]: list[ModelMessage] = []


def get_chat_history(session_id: str) -> list[ModelMessage]:
    """返回已有的聊天历史。"""
    create_session_if_not_exists(session_id)
    return list(chain.from_iterable(
        ModelMessagesTypeAdapter.validate_json(msg_group)
        for msg_group in store[session_id]
    ))


def store_messages_in_history(session_id: str, run_result: AgentRunResult[ModelMessage]) -> None:
    """将新消息存储到聊天历史中。"""
    create_session_if_not_exists(session_id)
    store[session_id].append(run_result.new_messages_json())


# ========== 带历史记录的问答 ==========

def ask_with_history(user_message: str, user_session_id: str) -> AgentRunResult[ModelMessage]:
    """向聊天机器人提问并将新消息存储到聊天历史中。"""
    chat_history = get_chat_history(user_session_id)
    chat_response = agent.run_sync(user_message, message_history=chat_history)
    store_messages_in_history(user_session_id, chat_response)
    return chat_response


# ========== 运行示例 ==========

if __name__ == '__main__':
    print(f'当前模型: {os.getenv("MODEL_NAME")}')
    print(f'API地址: {os.getenv("API_BASE_URL")}')
    print('输入 /quit 退出，输入 /history 查看对话历史\n')

    session_id = 'user_123'

    while True:
        user_input = input('你: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        if user_input == '/history':
            print('\n对话历史:')
            for message in get_chat_history(session_id):
                print(f'  {message.parts[-1].part_kind}: {message.parts[-1].content}')
            print()
            continue

        result = ask_with_history(user_input, session_id)
        print(f'AI: {result.output}\n')
