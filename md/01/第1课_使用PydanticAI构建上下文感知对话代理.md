# 第1课：使用 PydanticAI 构建具有上下文感知能力的对话代理

**本教程基于 LangChain 教程《构建具有上下文感知能力的对话代理》，使用 PydanticAI 作为代理框架来演示相同的概念。**

## PydanticAI 简介

[PydanticAI](https://ai.pydantic.dev/) 是一个全新的 Python 代理框架，旨在降低使用生成式 AI 构建生产级应用的难度。它由 **Pydantic** 背后的团队开发，将 Pydantic 中强大的数据验证和类型安全原则引入 AI 领域——正是这些特性使 Pydantic 成为了众多 LLM 库（包括 OpenAI SDK、Anthropic SDK、LangChain、LlamaIndex 等）的基石。

使用 PydanticAI，控制流和代理组合都通过**原生 Python** 来处理，让你可以像在任何其他（非 AI）项目中一样应用相同的开发最佳实践。

主要特性包括：

- 由 Pydantic 驱动的**[数据验证](https://ai.pydantic.dev/results/#structured-result-validation)**和**[类型安全](https://ai.pydantic.dev/agents/#static-type-checking)**
- **[依赖注入系统](https://ai.pydantic.dev/dependencies/)**，用于定义工具，后续课程中会有详细演示
- **[Logfire](https://ai.pydantic.dev/logfire/)**，用于增强可观测性的调试和监控工具
- 以及更多功能！

## 概述

本教程介绍如何创建一个能够在多轮交互中保持上下文的对话代理。我们将使用现代 AI 框架构建一个能够进行更自然、更连贯对话的代理。

## 动机

许多简单的聊天机器人缺乏保持上下文的能力，导致对话断裂、用户体验不佳。本教程旨在通过实现一个能够记住并引用之前对话内容的对话代理来解决这个问题，从而提升整体交互质量。

## 关键组件

1. **语言模型 (Language Model)**：生成回复的核心 AI 组件
2. **提示模板 (Prompt Template)**：定义对话的结构
3. **历史记录管理器 (History Manager)**：管理对话历史和上下文
4. **消息存储 (Message Store)**：为每个对话会话存储消息

## 方法详解

### 环境搭建

首先设置必要的 AI 框架并确保能够访问合适的语言模型，这构成了对话代理的基础。

### 创建聊天历史存储

实现一个管理多个对话会话的系统。每个会话都应有唯一标识符，并关联自己的消息历史。

### 定义对话结构

创建一个模板，包含：
- 定义 AI 角色的系统消息
- 对话历史的占位符
- 用户的输入

这个结构引导 AI 的回复并在整个对话过程中保持一致性。

### 构建对话代理

将提示模板与语言模型结合，创建一个基本的对话代理。用历史记录管理组件包装代理，自动处理对话历史的插入和检索。

### 与代理交互

使用代理时，需要传入用户输入和会话标识符。历史记录管理器负责检索相应的对话历史、将其插入到提示中，并在每次交互后存储新消息。

---

## 实现

### 安装依赖

```bash
pip install 'pydantic-ai-slim[openai]'
```

### 导入所需的库

```python
import os

from dotenv import load_dotenv
from itertools import chain

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.agent import AgentRunResult
```

### 处理 Jupyter Notebook 中的异步事件循环

```python
# 在 Jupyter Notebook 中运行异步代码时需要这个设置。
# 否则会报错：在已有事件循环运行的情况下尝试启动新的事件循环。
import nest_asyncio
nest_asyncio.apply()
```

### 加载环境变量并初始化语言模型

```python
load_dotenv()
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

# 使用自定义 API 地址和模型，兼容所有 OpenAI 接口格式的服务
# （如 DeepSeek、通义千问、智谱等）
model = OpenAIModel(
    os.getenv('MODEL_NAME', 'gpt-4o-mini'),       # 模型名称
    base_url=os.getenv('API_BASE_URL'),            # 自定义 API 地址
    api_key=os.getenv('API_KEY'),                  # API Key
)

agent = Agent(
    model=model,
    system_prompt='You are a helpful AI assistant.',  # 你是一个有帮助的 AI 助手
)
```

### 创建简单的内存聊天历史存储

```python
# 简单的内存存储。在实际应用中，这通常会是一个数据库。
# 注意：我们在存储消息前会将 Pydantic 的 `Message` 类型转换为 `bytes`，
# 以模拟实际应用中的使用方式。
store: dict[str, list[bytes]] = {}

def create_session_if_not_exists(session_id: str) -> None:
    """确保 session_id 在聊天存储中存在。"""
    if session_id not in store:
        store[session_id]: list[ModelMessage] = []

def get_chat_history(session_id: str) -> list[ModelMessage]:
    """返回已有的聊天历史。"""

    create_session_if_not_exists(session_id)

    # 将 bytes 转换为 Message 列表并返回历史记录。
    return list(chain.from_iterable(
        ModelMessagesTypeAdapter.validate_json(msg_group)
        for msg_group in store[session_id]
    ))

def store_messages_in_history(session_id: str, run_result: AgentRunResult[ModelMessage]) -> None:
    """将最近一次与模型交互产生的所有新消息存储到本地存储中。

    接收一个会话 ID 和模型返回的结果，获取所有新消息的 bytes 格式
    并存储到本地存储中。
    """
    create_session_if_not_exists(session_id)

    store[session_id].append(run_result.new_messages_json())
```

### 封装带历史记录的问答功能

```python
def ask_with_history(user_message: str, user_session_id: str) -> AgentRunResult[ModelMessage]:
    """向聊天机器人提问并将新消息存储到聊天历史中。"""

    # 获取已有的历史记录发送给模型
    chat_history = get_chat_history(user_session_id)

    # 发送用户问题和聊天历史
    chat_response: AgentRunResult[ModelMessage] = agent.run_sync(
        user_message, message_history=chat_history
    )

    # 将新消息存储到聊天历史中
    store_messages_in_history(user_session_id, chat_response)

    return chat_response
```

### 使用示例

```python
session_id = 'user_123'

result1 = ask_with_history('Hello! How are you?', session_id)
print('AI:', result1.data)

result2 = ask_with_history('What was my previous message?', session_id)
print('AI:', result2.data)
```

**输出：**

```
AI: Hello! I'm just a program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?
AI: Your previous message was: "Hello! How are you?" How can I assist you further?
```

### 打印对话历史

```python
print('\nConversation History:')
for message in get_chat_history(session_id):
    print(f'{message.parts[-1].part_kind}: {message.parts[-1].content}')
```

**输出：**

```
Conversation History:
user-prompt: Hello! How are you?
text: Hello! I'm just a program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?
user-prompt: What was my previous message?
text: Your previous message was: "Hello! How are you?" How can I assist you further?
```

---

## 总结

这种创建对话代理的方法具有以下优势：

- **上下文感知**：代理可以引用之前的对话内容，实现更自然的交互
- **简洁性**：模块化设计使实现保持简单直观
- **灵活性**：可以轻松修改对话结构或切换不同的语言模型
- **可扩展性**：基于会话的方法支持管理多个独立对话

在此基础上，你可以进一步增强代理的能力：

- 实现更复杂的提示工程
- 集成外部知识库
- 为特定领域添加专业能力
- 加入错误处理和对话修复策略

通过专注于上下文管理，这种对话代理设计显著改进了基本聊天机器人的功能，为构建更具吸引力和实用性的 AI 助手铺平了道路。
