"""
第22课 - 记忆增强对话代理内部机制

目的：让你看清双层记忆的本质：
  1. 短期记忆 = messages 列表，每轮 append，超长 slice 裁剪
  2. 长期记忆 = 从对话提取的事实列表，注入 system prompt
  3. 会话隔离 = dict[session_id] → 各自的 messages 和 facts
  4. 记忆注入 = 把事实字符串塞进 system prompt

Java 类比：
  - 短期记忆 = HttpSession（请求间共享，会话结束清空）
  - 长期记忆 = Redis/数据库（持久化，跨会话可用）
  - 会话隔离 = @SessionScope 的 Bean
"""

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(messages: list) -> str:
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


# ================================================================
#  记忆的本质：两个 dict
# ================================================================

# 短期记忆 = dict[session_id] → list[message]
#   就是 Java 的 Map<String, List<Message>>
short_term_memory = {}

# 长期记忆 = dict[session_id] → list[fact_string]
#   就是 Java 的 Map<String, List<String>>
long_term_memory = {}

MAX_SHORT = 20  # 短期最多20条消息
MAX_LONG = 10   # 长期最多10条事实


# ================================================================
#  完整流程
# ================================================================

def chat(session_id: str, user_input: str) -> str:
    """
    带双层记忆的对话。

    Java 类比：
        @Service
        @SessionScope
        public class ChatService {
            // 短期记忆 = 当前会话的消息列表
            private List<Message> shortTermMemory = new ArrayList<>();
            // 长期记忆 = 持久化的事实（可以存Redis）
            @Autowired private RedisTemplate<String, List<String>> longTermMemory;

            public String chat(String sessionId, String input) {
                // 1. 组装prompt（注入两种记忆）
                List<Message> prompt = new ArrayList<>();
                prompt.add(new SystemMessage("你是AI助手"));
                prompt.add(new SystemMessage("长期记忆: " + getLongTerm(sessionId)));
                prompt.addAll(shortTermMemory);  // ← 短期记忆
                prompt.add(new UserMessage(input));

                // 2. 调用LLM
                String response = llm.generate(prompt);

                // 3. 更新短期记忆（append + trim）
                shortTermMemory.add(new UserMessage(input));
                shortTermMemory.add(new AssistantMessage(response));
                if (shortTermMemory.size() > MAX) {
                    shortTermMemory = shortTermMemory.subList(size - MAX, size);
                }

                // 4. 更新长期记忆（提取事实）
                extractFacts(sessionId, input, response);

                return response;
            }
        }
    """

    # 初始化该会话的记忆
    if session_id not in short_term_memory:
        short_term_memory[session_id] = []
    if session_id not in long_term_memory:
        long_term_memory[session_id] = []

    # ==========================================
    # 第1步：组装 prompt（注入双层记忆）
    # ==========================================
    long_facts = long_term_memory[session_id]
    long_text = "。".join(long_facts) if long_facts else "暂无"

    messages = [
        # system prompt 包含长期记忆
        {"role": "system", "content": f"你是友好的AI助手。用中文回答。\n长期记忆：{long_text}"},
    ]
    # 加入短期记忆（对话历史）
    messages.extend(short_term_memory[session_id])
    # 加入当前输入
    messages.append({"role": "user", "content": user_input})

    print(f'    [prompt组装] system含长期记忆({len(long_facts)}条) + 短期历史({len(short_term_memory[session_id])}条) + 当前输入')

    # ==========================================
    # 第2步：调用LLM
    # ==========================================
    response = call_llm(messages)

    # ==========================================
    # 第3步：更新短期记忆（append + 裁剪）
    # ==========================================
    short_term_memory[session_id].append({"role": "user", "content": user_input})
    short_term_memory[session_id].append({"role": "assistant", "content": response})

    # 裁剪：就是 list slice
    if len(short_term_memory[session_id]) > MAX_SHORT:
        short_term_memory[session_id] = short_term_memory[session_id][-MAX_SHORT:]
        print(f'    [短期记忆裁剪] 保留最近 {MAX_SHORT} 条')

    # ==========================================
    # 第4步：更新长期记忆（简化版：长度>10的输入存为事实）
    # ==========================================
    if len(user_input) > 10:
        # 简化提取：直接存用户输入作为事实
        fact = f"用户说过：{user_input}"
        if not any(user_input in f for f in long_term_memory[session_id]):
            long_term_memory[session_id].append(fact)
            print(f'    [长期记忆更新] 新增: {fact[:40]}...')

        # 裁剪
        if len(long_term_memory[session_id]) > MAX_LONG:
            long_term_memory[session_id] = long_term_memory[session_id][-MAX_LONG:]

    return response


def show_memory(session_id: str):
    """展示记忆状态"""
    print()
    print('=' * 40)
    print(f'  记忆状态 (session: {session_id})')
    print('=' * 40)

    short = short_term_memory.get(session_id, [])
    print(f'  短期记忆: {len(short)} 条消息')
    if short:
        for m in short[-4:]:  # 只显示最近4条
            print(f'    [{m["role"]}] {m["content"][:50]}...')

    long = long_term_memory.get(session_id, [])
    print(f'  长期记忆: {len(long)} 条事实')
    for f in long:
        print(f'    - {f}')
    print('=' * 40)


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第22课 - 记忆增强对话代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 短期记忆 = list.append() + list[-N:] 裁剪')
    print('  - 长期记忆 = 提取事实存到另一个list，注入system prompt')
    print('  - 会话隔离 = dict[session_id] 各存各的')
    print('  - 记忆注入 = 事实字符串拼进system prompt')
    print()
    print('命令：/memory 查看记忆，/new 新会话，/quit 退出')
    print()

    session_id = "user_001"
    session_count = 1
    print(f'=== 会话 {session_count} ===')

    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break
        if user_input == '/memory':
            show_memory(session_id)
            continue
        if user_input == '/new':
            session_count += 1
            session_id = f"user_{session_count:03d}"
            print(f'\n=== 新会话 {session_count}（短期清空，长期保留） ===\n')
            continue

        response = chat(session_id, user_input)
        print(f'\nAI: {response}\n')
