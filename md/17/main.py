"""
第17课 - TTS诗歌朗读代理（LangGraph 框架版）

核心概念：
  - 菱形结构：分类→4选1分支→汇聚到TTS
  - 风格改写：同一文本，不同system prompt产生不同风格
  - 多模态输出：文本→语音（本课模拟TTS步骤）

说明：因无 OpenAI TTS API，语音合成步骤用文字模拟。
     生产中替换为 TTS API 即可生成真实语音。
"""

import os
import time
import re
from typing import TypedDict

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（带重试和空响应防护）"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not choices[0].get("message"):
                raise ValueError(f"API返回空响应")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                print(f'    ⚠ 重试({attempt+1}): {e}')
                time.sleep(wait)
            else:
                raise


# ========== State ==========

class State(TypedDict):
    input_text: str         # 用户输入
    content_type: str       # 分类结果
    processed_text: str     # 风格改写后的文本
    voice_type: str         # 语音类型
    tts_result: str         # TTS结果（模拟）


# ========== 节点：内容分类 ==========

def classify_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 内容分类】')
    print('=' * 60)

    system = ("将以下文本分类为四种类型之一，只回复类型名：\n"
              "- 一般：普通文本\n"
              "- 诗歌：诗词、抒情文字\n"
              "- 新闻：新闻报道、时事\n"
              "- 笑话：幽默、搞笑内容\n\n"
              "只回复：一般、诗歌、新闻、笑话")

    content_type = call_llm(state["input_text"], system)

    # 标准化
    for t in ["诗歌", "新闻", "笑话", "一般"]:
        if t in content_type:
            content_type = t
            break
    else:
        content_type = "一般"

    print(f'>>> 输入: {state["input_text"][:50]}...')
    print(f'>>> 分类: {content_type}')
    return {"content_type": content_type}


# ========== 4个处理节点 ==========

def process_general_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 一般文本处理】（原样输出）')
    print('=' * 60)
    print(f'>>> 不改写，直接传递给TTS')
    return {"processed_text": state["input_text"], "voice_type": "标准"}


def process_poem_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 诗歌风格改写】')
    print('=' * 60)

    result = call_llm(state["input_text"],
                       "把以下文字改写为一首优美的中文诗歌，注意韵律和意境。")
    print(f'>>> 改写结果:\n{result}')
    return {"processed_text": result, "voice_type": "柔和（诗歌朗诵）"}


def process_news_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 新闻播报改写】')
    print('=' * 60)

    result = call_llm(state["input_text"],
                       "把以下文字改写为正式的中文新闻播报稿，语气严肃专业。")
    print(f'>>> 改写结果:\n{result}')
    return {"processed_text": result, "voice_type": "沉稳（新闻播音）"}


def process_joke_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 笑话风格改写】')
    print('=' * 60)

    result = call_llm(state["input_text"],
                       "把以下文字改写为一个幽默搞笑的中文笑话或段子。")
    print(f'>>> 改写结果:\n{result}')
    return {"processed_text": result, "voice_type": "活泼（相声口吻）"}


# ========== 节点：语音合成（汇聚点） ==========

def tts_node(state: State):
    print()
    print('=' * 60)
    print('【3 - 语音合成】（所有分支汇聚到这里）')
    print('=' * 60)

    print(f'>>> 内容类型: {state["content_type"]}')
    print(f'>>> 语音风格: {state["voice_type"]}')
    print(f'>>> 待朗读文本: {state["processed_text"][:100]}...')
    print()

    # 模拟TTS
    print(f'>>> 生产环境中的TTS代码:')
    print(f'>>>   voice_map = {{"一般": "alloy", "诗歌": "nova", "新闻": "onyx", "笑话": "shimmer"}}')
    print(f'>>>   response = client.audio.speech.create(')
    print(f'>>>       model="tts-1",')
    print(f'>>>       voice=voice_map[content_type],')
    print(f'>>>       input=processed_text')
    print(f'>>>   )')
    print()

    tts_result = (f"[模拟语音输出]\n"
                  f"语音风格: {state['voice_type']}\n"
                  f"朗读内容: {state['processed_text']}")

    print(f'>>> 语音已生成（模拟）')
    return {"tts_result": tts_result}


# ========== 路由函数 ==========

def route_content(state: State) -> str:
    ct = state["content_type"]
    print(f'>>> 路由: {ct}')
    if ct == "诗歌":
        return "process_poem"
    elif ct == "新闻":
        return "process_news"
    elif ct == "笑话":
        return "process_joke"
    else:
        return "process_general"


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("classify", classify_node)
    workflow.add_node("process_general", process_general_node)
    workflow.add_node("process_poem", process_poem_node)
    workflow.add_node("process_news", process_news_node)
    workflow.add_node("process_joke", process_joke_node)
    workflow.add_node("tts", tts_node)

    workflow.set_entry_point("classify")

    # 分类→4选1分支
    workflow.add_conditional_edges("classify", route_content, {
        "process_general": "process_general",
        "process_poem": "process_poem",
        "process_news": "process_news",
        "process_joke": "process_joke",
    })

    # 4条分支汇聚到TTS（菱形结构）
    workflow.add_edge("process_general", "tts")
    workflow.add_edge("process_poem", "tts")
    workflow.add_edge("process_news", "tts")
    workflow.add_edge("process_joke", "tts")

    workflow.add_edge("tts", END)

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
        print('\n>>> 粘贴到 https://mermaid.live 查看可视化')
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第17课 - TTS诗歌朗读代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 示例
    examples = [
        ("春天来了，花儿开了，鸟儿在枝头歌唱", "预期→诗歌"),
        ("今日A股三大指数集体高开，沪指涨0.5%站上3200点", "预期→新闻"),
        ("程序员最怕什么？怕产品经理说：需求变了", "预期→笑话"),
        ("今天天气不错，我打算出去走走", "预期→一般"),
    ]

    print('示例文本:')
    for i, (text, expected) in enumerate(examples, 1):
        print(f'  {i}. {text} ({expected})')
    print()

    # 运行所有示例
    for text, expected in examples:
        print()
        print('#' * 60)
        print(f'#  输入: {text}')
        print(f'#  {expected}')
        print('#' * 60)

        result = app.invoke({
            "input_text": text,
            "content_type": "",
            "processed_text": "",
            "voice_type": "",
            "tts_result": "",
        })

    # 交互模式
    print('\n输入文本，输入 /quit 退出\n')
    while True:
        user_input = input('输入文本: ').strip()
        if user_input == '/quit':
            print('再见！')
            break
        if not user_input:
            continue

        result = app.invoke({
            "input_text": user_input,
            "content_type": "",
            "processed_text": "",
            "voice_type": "",
            "tts_result": "",
        })
