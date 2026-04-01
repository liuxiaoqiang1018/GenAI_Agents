"""
第21课 - 侦探推理游戏内部机制（不使用 LangGraph）

目的：让你看清子图（Sub-Graph）的本质：
  1. 子图 = 一个独立的函数，内部有自己的循环
  2. 外层图 = while游戏循环，调用子函数
  3. 多角色 = 同一个函数，传入不同的system prompt
  4. 凶手说谎 = system prompt里加"隐藏真相"的指令

对比 main.py（LangGraph 框架版），理解：
  - 子图compile后作为节点 → 就是函数调用
  - 外层conditional_edges → 就是 while + if-elif
  - ConversationState独立 → 就是函数有自己的局部变量
"""

import os
import json
import re
import time
import httpx
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


# ================================================================
#  子图的本质：一个独立的函数
# ================================================================

def interrogate(character: dict, story: str):
    """
    对话子图的本质：一个函数，内部有自己的 while 循环和局部变量。

    Java 类比：
        // 子图 = 独立的 Service 方法
        public class ConversationService {
            public void interrogate(Character character, String story) {
                // 局部变量（= ConversationState）
                List<Message> messages = new ArrayList<>();
                messages.add(new SystemMessage(character.getPrompt()));

                // 自我介绍
                String intro = llm.generate(messages);
                System.out.println(character.getName() + ": " + intro);

                // 对话循环（= 子图的循环边）
                while (true) {
                    String question = scanner.nextLine();
                    if ("exit".equals(question)) break;
                    messages.add(new UserMessage(question));
                    String answer = llm.generate(messages);
                    messages.add(new AssistantMessage(answer));
                }
            }
        }
    """

    print()
    print(f'    === 进入对话子图（就是一个函数调用） ===')
    print()

    # 子图的局部状态（= ConversationState）
    killer_hint = "\n你是凶手，要隐瞒真相但偶尔不自觉地露出破绽。" if character.get("is_killer") else ""
    system_msg = (f"你是{character['name']}，{character['role']}。{character['backstory']}\n"
                  f"案件背景：{story[:300]}\n"
                  f"你正在被侦探审问，保持角色性格。50字以内回答。{killer_hint}")

    messages = [{"role": "system", "content": system_msg}]

    # 自我介绍
    intro = call_llm_with_history(messages + [{"role": "user", "content": "请自我介绍"}])
    messages.append({"role": "assistant", "content": intro})
    print(f'{character["name"]}: {intro}\n')

    # 对话循环（= 子图的 ask → answer → ask 循环边）
    while True:
        question = input('侦探（输入 exit 结束审问）: ').strip()
        if question.lower() == 'exit' or not question:
            print(f'    === 退出对话子图 ===')
            break

        messages.append({"role": "user", "content": question})

        # 裁剪历史（保留system + 最近10条）
        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

        response = call_llm_with_history(messages)
        messages.append({"role": "assistant", "content": response})
        print(f'\n{character["name"]}: {response}\n')


# ================================================================
#  完整流程
# ================================================================

def play_game(environment: str, guesses: int = 3):
    """
    外层图的本质：一个 while 游戏循环。

    Java 类比：
        public class GameService {
            @Autowired ConversationService conversationService;

            public void play(String environment) {
                // 线性开头
                List<Character> characters = createCharacters(environment);
                String story = createStory(characters, environment);
                narrate(story, characters);

                // 游戏循环（= 外层图的条件边循环）
                int guessesLeft = 3;
                while (guessesLeft > 0) {
                    String action = chooseAction(characters);
                    if ("guess".equals(action)) {
                        if (guess(characters, killer)) break;
                        guessesLeft--;
                    } else {
                        conversationService.interrogate(character, story); // 调子图
                    }
                }
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：角色生成
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：角色生成】')
    print('=' * 60)

    system = (f"为发生在「{environment}」的谋杀案创建4个嫌疑人（1个凶手）。\n"
              f"返回JSON：{{\"characters\": [{{\"name\":\"张三\",\"role\":\"厨师\",\"backstory\":\"背景\",\"is_killer\":false}}]}}")
    result = extract_json(call_llm("生成角色", system))
    characters = result.get("characters", [
        {"name": "王厨师", "role": "厨师", "backstory": "与受害者有财务纠纷", "is_killer": False},
        {"name": "李管家", "role": "管家", "backstory": "知道所有秘密", "is_killer": True},
        {"name": "赵秘书", "role": "秘书", "backstory": "表面温顺实际野心勃勃", "is_killer": False},
        {"name": "陈医生", "role": "医生", "backstory": "经常出入豪宅", "is_killer": False},
    ])
    total_llm_calls += 1

    killer = next((c for c in characters if c.get("is_killer")), characters[0])
    killer_name = killer["name"]
    print(f'>>> {len(characters)} 个嫌疑人，凶手: {killer_name}（玩家不知道）')

    # ==========================================
    # 第2步：故事生成
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：故事生成】')
    print('=' * 60)

    chars_desc = "\n".join(f"- {c['name']}（{c['role']}）" for c in characters)
    story = call_llm("创建谋杀案故事",
                      f"地点：{environment}\n嫌疑人：\n{chars_desc}\n创建案件故事，不透露凶手。200字。")
    total_llm_calls += 1
    print(f'>>> 故事: {story[:200]}...')

    # ==========================================
    # 第3步：叙述开场
    # ==========================================
    print()
    print('=' * 60)
    print('  侦探推理游戏')
    print('=' * 60)
    print()
    print(f'案件地点: {environment}')
    print()
    print(story)
    print()
    print('嫌疑人:')
    for i, c in enumerate(characters, 1):
        print(f'  {i}. {c["name"]}（{c["role"]}）')
    print(f'\n你有 {guesses} 次猜测机会。')

    # ==========================================
    # 游戏循环（= 外层图的条件边循环）
    # ==========================================
    guesses_left = guesses

    while guesses_left > 0:
        print()
        print('-' * 40)
        print('选择行动:')
        for i, c in enumerate(characters, 1):
            print(f'  {i}. 审问 {c["name"]}')
        print(f'  G. 猜测凶手（剩余 {guesses_left} 次）')

        choice = input('\n你的选择: ').strip()

        if choice.upper() == 'G':
            # 猜测分支
            print()
            print('嫌疑人:')
            for i, c in enumerate(characters, 1):
                print(f'  {i}. {c["name"]}')

            guess = input('凶手是（编号）: ').strip()
            try:
                guessed = characters[int(guess) - 1]["name"]
            except (ValueError, IndexError):
                guessed = ""

            guesses_left -= 1

            if guessed == killer_name:
                print()
                print('*' * 60)
                print(f'  恭喜！凶手就是 {killer_name}！')
                print('*' * 60)
                return {"result": "胜利", "llm_calls": total_llm_calls}
            else:
                print(f'\n不对！剩余 {guesses_left} 次。')
                if guesses_left <= 0:
                    print()
                    print('*' * 60)
                    print(f'  游戏结束！凶手是: {killer_name}')
                    print('*' * 60)
                    return {"result": "失败", "llm_calls": total_llm_calls}
        else:
            # 审问分支 → 调用子图（就是调函数）
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(characters):
                    print()
                    print('=' * 60)
                    print(f'【审问 {characters[idx]["name"]}】')
                    print('=' * 60)
                    interrogate(characters[idx], story)  # ← 子图 = 函数调用
                else:
                    print('无效选择')
            except ValueError:
                print('无效选择')


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第21课 - 侦探推理游戏（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 子图 = 一个独立的函数（有自己的局部变量和循环）')
    print('  - 外层图 = while 游戏循环 + if-elif 选择')
    print('  - 多角色 = 同一函数，不同 system prompt')
    print('  - 凶手说谎 = prompt 里加"隐藏真相"指令')
    print()

    env = input('案件地点（回车默认"上海外滩的百年老洋房"）: ').strip()
    if not env:
        env = "上海外滩的百年老洋房"

    result = play_game(env)
    print(f'\n游戏结果: {result["result"]}')
