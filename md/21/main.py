"""
第21课 - 侦探推理游戏（LangGraph 框架版）

核心概念：
  - 子图（Sub-Graph）：对话循环是独立的图，嵌入到外层游戏图中
  - 多角色人格：每个嫌疑人有独立的system prompt
  - 游戏状态管理：外层管游戏进度，内层管对话历史
  - 用户交互：选择嫌疑人、提问、猜凶手
"""

import os
import json
import re
import time
import random
from typing import TypedDict, List, Optional

import httpx
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
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
                time.sleep((attempt + 1) * 3)
            else:
                raise


def call_llm_with_history(messages: list) -> str:
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
                time.sleep((attempt + 1) * 3)
            else:
                raise


def extract_json(text: str):
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        return {}


# ========== 外层 State（游戏主循环） ==========

class GameState(TypedDict):
    environment: str              # 案件地点
    characters: list              # 嫌疑人列表 [{name, role, backstory, is_killer}]
    story: str                    # 案件故事
    killer_name: str              # 凶手名字
    guesses_left: int             # 剩余猜测次数
    current_character_idx: int    # 当前审问的角色索引
    game_over: bool               # 游戏是否结束
    game_result: str              # 游戏结果


# ========== 内层 State（对话子循环） ==========

class ConversationState(TypedDict):
    character: dict               # 当前审问的角色
    story: str                    # 案件故事（上下文）
    messages: list                # 对话历史
    done: bool                    # 对话是否结束


# ========== 外层节点：角色生成 ==========

def create_characters_node(state: GameState):
    print()
    print('=' * 60)
    print('【1 - 角色生成】')
    print('=' * 60)

    system = (f"你是一个推理小说作家。为发生在「{state['environment']}」的谋杀案创建4个嫌疑人。\n"
              f"其中一个是凶手。每个角色需要：名字、职业、背景故事和动机。\n"
              f"返回JSON：{{\"characters\": [\n"
              f"  {{\"name\": \"张三\", \"role\": \"厨师\", \"backstory\": \"背景和动机\", \"is_killer\": false}},\n"
              f"  ...\n"
              f"]}}\n"
              f"注意：恰好1个角色的 is_killer 为 true。")

    result = extract_json(call_llm("生成角色", system))
    characters = result.get("characters", [])

    if not characters:
        characters = [
            {"name": "王厨师", "role": "厨师", "backstory": "在厨房工作多年，与受害者有财务纠纷", "is_killer": False},
            {"name": "李管家", "role": "管家", "backstory": "忠心耿耿的老管家，知道所有人的秘密", "is_killer": True},
            {"name": "赵秘书", "role": "秘书", "backstory": "新来的秘书，表面温顺实际野心勃勃", "is_killer": False},
            {"name": "陈医生", "role": "家庭医生", "backstory": "受害者的私人医生，经常出入豪宅", "is_killer": False},
        ]

    killer = next((c for c in characters if c.get("is_killer")), characters[0])
    killer_name = killer["name"]

    print(f'>>> 生成 {len(characters)} 个嫌疑人:')
    for c in characters:
        marker = " (凶手)" if c.get("is_killer") else ""
        print(f'    - {c["name"]}（{c["role"]}）{marker}')
    print(f'>>> 凶手是: {killer_name}（玩家不知道）')

    return {"characters": characters, "killer_name": killer_name}


# ========== 外层节点：故事生成 ==========

def create_story_node(state: GameState):
    print()
    print('=' * 60)
    print('【2 - 故事生成】')
    print('=' * 60)

    chars_desc = "\n".join(f"- {c['name']}（{c['role']}）：{c['backstory']}" for c in state["characters"])
    system = (f"你是推理小说作家。根据以下角色创建谋杀案故事。\n"
              f"地点：{state['environment']}\n"
              f"嫌疑人：\n{chars_desc}\n\n"
              f"包含：受害者是谁、发现经过、已知线索。不要透露凶手。200字以内。")

    story = call_llm("创建谋杀案故事", system)
    print(f'>>> 故事: {story[:200]}...')
    return {"story": story}


# ========== 外层节点：叙述开场 ==========

def narrator_node(state: GameState):
    print()
    print('=' * 60)
    print('  侦探推理游戏')
    print('=' * 60)
    print()
    print(f'案件地点: {state["environment"]}')
    print()
    print(state["story"])
    print()
    print(f'嫌疑人:')
    for i, c in enumerate(state["characters"], 1):
        print(f'  {i}. {c["name"]}（{c["role"]}）')
    print()
    print(f'你有 {state["guesses_left"]} 次猜测机会。')
    print(f'你可以审问嫌疑人来收集线索，然后猜凶手。')
    print()
    return {}


# ========== 外层节点：选择行动 ==========

def choose_action_node(state: GameState):
    print()
    print('-' * 40)
    print('请选择行动:')
    for i, c in enumerate(state["characters"], 1):
        print(f'  {i}. 审问 {c["name"]}')
    print(f'  G. 猜测凶手（剩余 {state["guesses_left"]} 次）')
    print()

    choice = input('你的选择: ').strip()

    if choice.upper() == 'G':
        return {"current_character_idx": -1}  # -1 表示猜测

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(state["characters"]):
            print(f'>>> 选择审问: {state["characters"][idx]["name"]}')
            return {"current_character_idx": idx}
    except ValueError:
        pass

    print('>>> 无效选择，默认审问第一个嫌疑人')
    return {"current_character_idx": 0}


# ========== 外层路由：审问还是猜测 ==========

def route_action(state: GameState) -> str:
    if state["current_character_idx"] == -1:
        return "guesser"
    else:
        return "conversation"


# ========== 内层子图：对话循环 ==========

def build_conversation_subgraph():
    """构建对话子图"""

    def intro_node(state: ConversationState):
        char = state["character"]
        system = (f"你是{char['name']}，{char['role']}。{char['backstory']}\n"
                  f"案件背景：{state['story'][:300]}\n\n"
                  f"你被侦探叫来审问。先做自我介绍（50字以内）。"
                  + ("\n注意：你是凶手，要隐瞒真相，但可以微妙地露出马脚。" if char.get("is_killer") else ""))

        intro = call_llm("请自我介绍", system)
        print(f'\n{char["name"]}: {intro}\n')

        messages = [
            {"role": "system", "content": (
                f"你是{char['name']}，{char['role']}。{char['backstory']}\n"
                f"案件背景：{state['story'][:300]}\n"
                f"你正在被侦探审问，要回答侦探的提问。保持角色性格。50字以内回答。"
                + ("\n你是凶手，要隐瞒真相但偶尔不自觉地露出破绽。" if char.get("is_killer") else "")
            )},
            {"role": "assistant", "content": intro},
        ]
        return {"messages": messages}

    def ask_node(state: ConversationState):
        question = input(f'侦探（输入 exit 结束审问）: ').strip()
        if question.lower() == 'exit' or not question:
            return {"done": True}
        return {"messages": state["messages"] + [{"role": "user", "content": question}], "done": False}

    def answer_node(state: ConversationState):
        response = call_llm_with_history(state["messages"])
        char_name = state["character"]["name"]
        print(f'\n{char_name}: {response}\n')
        return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

    def should_continue(state: ConversationState) -> str:
        if state.get("done"):
            return END
        return "answer"

    # 构建子图
    conv = StateGraph(ConversationState)
    conv.add_node("intro", intro_node)
    conv.add_node("ask", ask_node)
    conv.add_node("answer", answer_node)

    conv.set_entry_point("intro")
    conv.add_edge("intro", "ask")
    conv.add_conditional_edges("ask", should_continue, {"answer": "answer", END: END})
    conv.add_edge("answer", "ask")  # 循环：回答→继续提问

    return conv.compile()


# ========== 外层节点：对话（调用子图） ==========

def conversation_wrapper(state: GameState):
    """把子图包装成外层节点"""
    idx = state["current_character_idx"]
    character = state["characters"][idx]

    print()
    print('=' * 60)
    print(f'【审问 {character["name"]}（{character["role"]}）】')
    print('=' * 60)

    # 调用子图
    conv_graph = build_conversation_subgraph()
    conv_graph.invoke({
        "character": character,
        "story": state["story"],
        "messages": [],
        "done": False,
    })

    return {}


# ========== 外层节点：猜凶手 ==========

def guesser_node(state: GameState):
    print()
    print('=' * 60)
    print(f'【猜测凶手】（剩余 {state["guesses_left"]} 次）')
    print('=' * 60)

    print('嫌疑人:')
    for i, c in enumerate(state["characters"], 1):
        print(f'  {i}. {c["name"]}（{c["role"]}）')

    choice = input('你认为凶手是（输入编号）: ').strip()
    try:
        idx = int(choice) - 1
        guessed_name = state["characters"][idx]["name"]
    except (ValueError, IndexError):
        guessed_name = ""
        print('>>> 无效选择')

    guesses_left = state["guesses_left"] - 1

    if guessed_name == state["killer_name"]:
        print()
        print('*' * 60)
        print(f'  恭喜！你猜对了！凶手就是 {state["killer_name"]}！')
        print('*' * 60)
        return {"guesses_left": guesses_left, "game_over": True, "game_result": "胜利"}
    else:
        print(f'\n>>> 不对！{guessed_name} 不是凶手。剩余 {guesses_left} 次机会。')
        if guesses_left <= 0:
            print()
            print('*' * 60)
            print(f'  游戏结束！你没有猜出凶手。')
            print(f'  凶手是: {state["killer_name"]}')
            print('*' * 60)
            return {"guesses_left": 0, "game_over": True, "game_result": "失败"}
        return {"guesses_left": guesses_left, "game_over": False}


# ========== 外层路由：游戏是否结束 ==========

def check_game_over(state: GameState) -> str:
    if state.get("game_over"):
        return END
    return "choose_action"


# ========== 构建外层图 ==========

def build_game():
    workflow = StateGraph(GameState)

    workflow.add_node("create_characters", create_characters_node)
    workflow.add_node("create_story", create_story_node)
    workflow.add_node("narrator", narrator_node)
    workflow.add_node("choose_action", choose_action_node)
    workflow.add_node("conversation", conversation_wrapper)
    workflow.add_node("guesser", guesser_node)

    # 线性开头
    workflow.set_entry_point("create_characters")
    workflow.add_edge("create_characters", "create_story")
    workflow.add_edge("create_story", "narrator")
    workflow.add_edge("narrator", "choose_action")

    # 选择行动
    workflow.add_conditional_edges("choose_action", route_action, {
        "conversation": "conversation",
        "guesser": "guesser",
    })

    # 审问后回到选择
    workflow.add_edge("conversation", "choose_action")

    # 猜测后检查是否结束
    workflow.add_conditional_edges("guesser", check_game_over, {
        "choose_action": "choose_action",
        END: END,
    })

    return workflow.compile()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第21课 - 侦探推理游戏')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    env = input('案件地点（回车默认"上海外滩的百年老洋房"）: ').strip()
    if not env:
        env = "上海外滩的百年老洋房"
        print(f'>>> 使用: {env}')

    game = build_game()

    game.invoke({
        "environment": env,
        "characters": [],
        "story": "",
        "killer_name": "",
        "guesses_left": 3,
        "current_character_idx": 0,
        "game_over": False,
        "game_result": "",
    })
