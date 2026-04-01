"""
第19课 - 多平台内容生成代理（LangGraph 框架版）

核心概念：
  - 线性+并行+汇聚：前处理流水线→4平台并行→合并输出
  - 平台适配：同一摘要+调研，不同prompt适配不同平台
  - 条件跳过：未选平台返回空内容
  - operator.add：并行结果自动拼接
"""

import os
import json
import re
import time
import operator
from typing import TypedDict, List, Annotated

import httpx
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
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
                raise ValueError("API返回空响应")
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
    text: str                                              # 原始文章
    platforms: List[str]                                   # 目标平台列表
    summary: str                                           # 摘要
    research: str                                          # 调研结果
    contents: Annotated[List[str], operator.add]           # 各平台内容（并行汇聚）
    final_output: str                                      # 最终合并输出


# ========== 节点：文本摘要 ==========

def summary_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 文本摘要】')
    print('=' * 60)

    summary = call_llm(
        state["text"],
        "你是内容分析专家。提取以下文章的核心要点，生成200字以内的摘要。用中文。"
    )
    print(f'>>> 摘要: {summary[:150]}...')
    return {"summary": summary}


# ========== 节点：内容调研 ==========

def research_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 内容调研】（用LLM补充背景知识）')
    print('=' * 60)

    research = call_llm(
        f"文章摘要：{state['summary']}\n\n"
        f"请补充以下内容：\n"
        f"1. 相关行业趋势\n"
        f"2. 目标受众关注的热点\n"
        f"3. 可以引用的数据或案例",
        "你是行业研究分析师。为社交媒体内容创作提供研究支持。用中文。"
    )
    print(f'>>> 调研: {research[:150]}...')
    return {"research": research}


# ========== 节点：意图匹配 ==========

def intent_node(state: State):
    print()
    print('=' * 60)
    print('【3 - 意图匹配】')
    print('=' * 60)

    platforms = state["platforms"]
    print(f'>>> 目标平台: {platforms}')
    print(f'>>> 即将为 {len(platforms)} 个平台并行生成内容')
    return {}


# ========== 4个平台节点 ==========

def weibo_node(state: State):
    """微博：140字+话题标签"""
    if "微博" not in state["platforms"]:
        return {"contents": []}

    print(f'    ✓ 微博: 生成中...')
    content = call_llm(
        f"摘要：{state['summary']}\n调研：{state['research']}",
        ("你是微博运营专家。生成一条微博内容：\n"
         "1. 正文不超过140字\n"
         "2. 加入2-3个话题标签（#标签#格式）\n"
         "3. 语气简洁有力，适合传播\n"
         "用中文。")
    )
    print(f'    ✓ 微博: 完成')
    return {"contents": [f"【微博】\n{content}"]}


def gongzhonghao_node(state: State):
    """公众号：长文"""
    if "公众号" not in state["platforms"]:
        return {"contents": []}

    print(f'    ✓ 公众号: 生成中...')
    content = call_llm(
        f"摘要：{state['summary']}\n调研：{state['research']}",
        ("你是公众号内容编辑。生成一篇公众号文章：\n"
         "1. 吸引人的标题\n"
         "2. 开头引入话题\n"
         "3. 正文分段，有小标题\n"
         "4. 结尾总结+引导互动\n"
         "用中文。")
    )
    print(f'    ✓ 公众号: 完成')
    return {"contents": [f"【公众号】\n{content}"]}


def xiaohongshu_node(state: State):
    """小红书：种草风格"""
    if "小红书" not in state["platforms"]:
        return {"contents": []}

    print(f'    ✓ 小红书: 生成中...')
    content = call_llm(
        f"摘要：{state['summary']}\n调研：{state['research']}",
        ("你是小红书博主。生成一篇小红书笔记：\n"
         "1. 吸睛标题（可用emoji）\n"
         "2. 分享体验式写法，亲切自然\n"
         "3. 关键信息用emoji标注\n"
         "4. 结尾加10个相关标签\n"
         "用中文。")
    )
    print(f'    ✓ 小红书: 完成')
    return {"contents": [f"【小红书】\n{content}"]}


def zhihu_node(state: State):
    """知乎：深度分析"""
    if "知乎" not in state["platforms"]:
        return {"contents": []}

    print(f'    ✓ 知乎: 生成中...')
    content = call_llm(
        f"摘要：{state['summary']}\n调研：{state['research']}",
        ("你是知乎优质答主。生成一篇知乎回答：\n"
         "1. 先给结论\n"
         "2. 分点论述，有理有据\n"
         "3. 引用数据或案例\n"
         "4. 专业但不枯燥\n"
         "用中文。")
    )
    print(f'    ✓ 知乎: 完成')
    return {"contents": [f"【知乎】\n{content}"]}


# ========== 节点：内容合并 ==========

def combine_node(state: State):
    print()
    print('=' * 60)
    print('【5 - 内容合并】')
    print('=' * 60)

    valid_contents = [c for c in state["contents"] if c.strip()]
    final = "\n\n" + "=" * 40 + "\n\n".join(valid_contents)

    print(f'>>> 合并了 {len(valid_contents)} 个平台的内容')
    return {"final_output": final}


# ========== 扇出函数 ==========

def fan_out_platforms(state: State):
    """并行分发到4个平台节点"""
    print()
    print('=' * 60)
    print(f'【4 - 平台内容生成】（并行: {len(state["platforms"])} 个平台）')
    print('=' * 60)

    # 所有4个平台节点都会被调用，内部判断是否跳过
    return [
        Send("weibo", state),
        Send("gongzhonghao", state),
        Send("xiaohongshu", state),
        Send("zhihu", state),
    ]


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    # 线性前处理
    workflow.add_node("summary", summary_node)
    workflow.add_node("research", research_node)
    workflow.add_node("intent", intent_node)

    # 4个平台节点
    workflow.add_node("weibo", weibo_node)
    workflow.add_node("gongzhonghao", gongzhonghao_node)
    workflow.add_node("xiaohongshu", xiaohongshu_node)
    workflow.add_node("zhihu", zhihu_node)

    # 合并节点
    workflow.add_node("combine", combine_node)

    # 线性: 摘要→调研→意图
    workflow.set_entry_point("summary")
    workflow.add_edge("summary", "research")
    workflow.add_edge("research", "intent")

    # 并行: 意图→4个平台（Send扇出）
    workflow.add_conditional_edges("intent", fan_out_platforms,
                                   ["weibo", "gongzhonghao", "xiaohongshu", "zhihu"])

    # 汇聚: 4个平台→合并
    workflow.add_edge("weibo", "combine")
    workflow.add_edge("gongzhonghao", "combine")
    workflow.add_edge("xiaohongshu", "combine")
    workflow.add_edge("zhihu", "combine")

    workflow.add_edge("combine", END)

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 示例文章 ==========

EXAMPLE_ARTICLE = """
大语言模型（LLM）正在深刻改变软件开发的方式。GitHub Copilot、Cursor等AI编程助手已经成为
许多开发者的日常工具。最新数据显示，使用AI辅助编程的开发者效率提升了30%-50%。

然而，AI编程并不意味着"不需要程序员"。相反，AI更像是一个强大的副驾驶——它能快速生成代码、
解释错误、提供建议，但最终的架构设计、代码审查和质量把控仍然需要人类开发者来完成。

对于程序员来说，学会与AI协作而非被AI替代，是未来最重要的技能之一。掌握prompt工程、理解
AI的能力边界、培养系统设计思维，这些才是AI时代程序员的核心竞争力。
"""


# ========== 运行 ==========

if __name__ == '__main__':
    print('第19课 - 多平台内容生成代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    # 获取文章
    print('请输入文章内容（回车使用默认示例，多行输入以 END 结束）:')
    first_line = input().strip()
    if not first_line:
        article = EXAMPLE_ARTICLE
        print('>>> 使用默认示例文章')
    else:
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        article = '\n'.join(lines)

    # 选择平台
    print('\n可选平台: 微博、公众号、小红书、知乎')
    platform_input = input('选择平台（逗号分隔，回车默认全部）: ').strip()
    if platform_input:
        platforms = [p.strip() for p in platform_input.split(',') if p.strip()]
    else:
        platforms = ["微博", "公众号", "小红书", "知乎"]
        print(f'>>> 使用全部平台: {platforms}')

    print()
    print('#' * 60)
    print('#  开始生成多平台内容')
    print('#' * 60)

    result = app.invoke({
        "text": article,
        "platforms": platforms,
        "summary": "",
        "research": "",
        "contents": [],
        "final_output": "",
    })

    print()
    print('#' * 60)
    print('#  生成完成')
    print('#' * 60)
    print(result["final_output"])
