"""
第16课 - GIF动画生成器（LangGraph 框架版）

核心概念：
  - 提示词链式传递：每步输出作为下一步输入，上下文不断丰富
  - 一致性约束：角色描述传递到所有帧，保证跨帧一致性
  - 多模态流水线：文本→提示词→图像描述（生产中接图像API）
  - 5节点线性流水线

说明：因无 DALL-E API，图像生成步骤用LLM生成详细的场景描述代替。
     生产环境中替换为真实图像API即可。
"""

import os
import json
import re
import time
from typing import TypedDict, List

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（带重试）"""
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
                raise ValueError(f"API返回空响应: {data}")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                print(f'    ⚠ API异常，{wait}秒后重试({attempt+1}/{MAX_RETRIES}): {e}')
                time.sleep(wait)
            else:
                raise


# ========== State ==========

class State(TypedDict):
    query: str                       # 用户输入的场景描述
    character_description: str       # 角色详细描述
    plot: str                        # 5帧剧情
    image_prompts: List[str]         # 5个图像提示词
    scene_descriptions: List[str]    # 5个场景详细描述（模拟图像）
    summary: str                     # 最终摘要


# ========== 节点1：角色描述 ==========

def character_description_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 角色描述生成】')
    print('=' * 60)

    prompt = (f"根据以下描述，创建一个详细的角色/场景描述。\n"
              f"包含外观细节、特征、颜色、材质等，用于后续保持一致性。\n\n"
              f"用户描述：{state['query']}\n\n"
              f"请用中文给出详细的角色描述（150字以内）。")

    desc = call_llm(prompt)
    print(f'>>> 角色描述: {desc}')
    return {"character_description": desc}


# ========== 节点2：剧情生成 ==========

def plot_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 剧情生成】（接收角色描述作为输入）')
    print('=' * 60)

    # 提示词链：包含前面的角色描述
    prompt = (f"根据以下信息，创建一个5帧GIF动画的剧情。\n"
              f"每帧一句话描述动作。保持故事连贯、有趣。\n\n"
              f"用户描述：{state['query']}\n"
              f"角色特征：{state['character_description']}\n\n"
              f"请用中文输出5帧剧情，格式：\n"
              f"1. 第一帧动作\n2. 第二帧动作\n...")

    plot = call_llm(prompt)
    print(f'>>> 剧情:\n{plot}')
    return {"plot": plot}


# ========== 节点3：提示词生成 ==========

def image_prompts_node(state: State):
    print()
    print('=' * 60)
    print('【3 - 图像提示词生成】（接收角色描述+剧情）')
    print('=' * 60)

    # 提示词链：包含角色描述+剧情
    prompt = (f"根据以下角色和剧情，为每一帧生成具体的图像提示词。\n\n"
              f"角色特征：{state['character_description']}\n"
              f"剧情：{state['plot']}\n\n"
              f"要求：\n"
              f"1. 每个提示词必须包含角色的关键特征（保持一致性）\n"
              f"2. 描述具体的动作、表情、背景\n"
              f"3. 适合图像生成模型使用\n\n"
              f"输出5个提示词，格式：\n"
              f"1. 提示词1\n2. 提示词2\n...")

    response = call_llm(prompt)

    # 解析提示词列表
    prompts = []
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            prompt_text = line.split('.', 1)[1].strip()
            if prompt_text:
                prompts.append(prompt_text)

    if not prompts:
        prompts = [response]  # 兜底

    print(f'>>> 生成 {len(prompts)} 个提示词:')
    for i, p in enumerate(prompts, 1):
        print(f'    帧{i}: {p[:60]}...')

    return {"image_prompts": prompts}


# ========== 节点4：图像生成（模拟） ==========

def create_images_node(state: State):
    print()
    print('=' * 60)
    print('【4 - 图像生成】（模拟 — 生产中接入 DALL-E / Midjourney）')
    print('=' * 60)

    print(f'>>> 生产环境中，这里会为每个提示词调用图像生成API:')
    print(f'>>>   response = client.images.generate(')
    print(f'>>>       model="dall-e-3",')
    print(f'>>>       prompt=image_prompt,')
    print(f'>>>       size="1024x1024"')
    print(f'>>>   )')

    # 模拟：直接用提示词作为场景描述（生产中这里调图像API）
    scenes = []
    for i, prompt in enumerate(state["image_prompts"], 1):
        scenes.append(prompt)
        print(f'    帧{i}: {prompt[:80]}...')

    return {"scene_descriptions": scenes}


# ========== 节点5：合成GIF（模拟） ==========

def create_gif_node(state: State):
    print()
    print('=' * 60)
    print('【5 - 合成GIF】（模拟 — 生产中用 PIL 拼帧）')
    print('=' * 60)

    print(f'>>> 生产环境中，这里会:')
    print(f'>>>   1. 下载所有图片 URL')
    print(f'>>>   2. 用 PIL 打开每张图')
    print(f'>>>   3. images[0].save(format="GIF", save_all=True, append_images=images[1:])')
    print()

    # 展示最终效果
    summary = f"GIF动画：{state['query']}\n\n"
    summary += f"角色：{state['character_description'][:100]}\n\n"
    summary += "动画帧序列：\n"
    for i, scene in enumerate(state["scene_descriptions"], 1):
        summary += f"  帧{i}: {scene}\n"

    print(summary)
    return {"summary": summary}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("character_description", character_description_node)
    workflow.add_node("plot", plot_node)
    workflow.add_node("image_prompts", image_prompts_node)
    workflow.add_node("create_images", create_images_node)
    workflow.add_node("create_gif", create_gif_node)

    # 纯线性流水线
    workflow.set_entry_point("character_description")
    workflow.add_edge("character_description", "plot")
    workflow.add_edge("plot", "image_prompts")
    workflow.add_edge("image_prompts", "create_images")
    workflow.add_edge("create_images", "create_gif")
    workflow.add_edge("create_gif", END)

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
    print('第16课 - GIF动画生成器')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    examples = [
        "一只戴礼帽的猫坐在书桌前用鹅毛笔写信",
        "一个机器人在厨房里学做中国菜",
        "一只小狗在雪地里追蝴蝶",
    ]

    print('示例场景:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    query = input('请输入GIF场景描述（回车使用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    print()
    print('#' * 60)
    print('#  开始生成GIF动画')
    print('#' * 60)

    result = app.invoke({
        "query": query,
        "character_description": "",
        "plot": "",
        "image_prompts": [],
        "scene_descriptions": [],
        "summary": "",
    })

    print()
    print('#' * 60)
    print('#  GIF动画生成完成')
    print('#' * 60)
    print(f'  帧数: {len(result["scene_descriptions"])}')
    print(f'  提示: 接入 DALL-E 或 Midjourney API 即可生成真实图像')
