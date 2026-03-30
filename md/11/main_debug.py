"""
第11课 - 旅行规划器内部机制（不使用 LangGraph）

目的：让你看清"交互式收集→LLM生成"的本质：
  1. 输入城市 = input() 存到变量
  2. 输入兴趣 = input() + split 存到列表
  3. 生成行程 = 一次 LLM 调用，把城市+兴趣填入 prompt
  4. 整个系统 = 两次 input() + 一次 API 调用

对比 main.py（LangGraph 框架版），理解：
  - 3个节点 → 就是3个函数顺序调用
  - State 累积 → 就是往 dict 里存值
  - add_edge(A, B) → 就是先调 A 再调 B
  - 前两步不调 LLM → 就是普通的 input()
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


def call_llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ================================================================
#  完整流程
# ================================================================

def plan_trip() -> dict:
    """
    旅行规划的完整流程。

    Java 类比：
        @Service
        public class TravelPlannerService {
            public TripPlan plan() {
                // 第1步：收集城市（纯用户输入，不调用任何服务）
                String city = scanner.nextLine();

                // 第2步：收集兴趣（纯用户输入）
                List<String> interests = Arrays.asList(scanner.nextLine().split(","));

                // 第3步：调用 LLM 生成行程（唯一的外部调用）
                String itinerary = llmService.generate(city, interests);

                return new TripPlan(city, interests, itinerary);
            }
        }
    """

    state = {"city": "", "interests": [], "itinerary": ""}

    # ==========================================
    # 第1步：输入城市（不调用 LLM！）
    # ==========================================
    print()
    print('=' * 60)
    print('【1 - 输入城市】（不调用 LLM）')
    print('=' * 60)
    print('>>> 这一步就是 input()，没有任何 AI 参与')
    print()

    city = input('请输入你想去的城市: ').strip()
    if not city:
        city = "北京"
        print(f'>>> 未输入，默认: {city}')
    else:
        print(f'>>> 目标城市: {city}')

    state["city"] = city
    print(f'>>> 当前状态: {{"city": "{city}", "interests": [], "itinerary": ""}}')

    # ==========================================
    # 第2步：输入兴趣（不调用 LLM！）
    # ==========================================
    print()
    print('=' * 60)
    print('【2 - 输入兴趣】（不调用 LLM）')
    print('=' * 60)
    print('>>> 这一步也是 input() + split，纯字符串操作')
    print()

    print(f'你要去 {city}，请输入你的兴趣（逗号分隔）:')
    print('例如: 美食,历史,购物,自然风光')
    raw = input('你的兴趣: ').strip()

    if not raw:
        interests = ["美食", "历史"]
        print(f'>>> 未输入，默认: {interests}')
    else:
        interests = [i.strip() for i in raw.split(',') if i.strip()]
        print(f'>>> 兴趣列表: {interests}')

    state["interests"] = interests
    print(f'>>> 当前状态: {{"city": "{city}", "interests": {interests}, "itinerary": ""}}')

    # ==========================================
    # 第3步：生成行程（唯一的 LLM 调用！）
    # ==========================================
    print()
    print('=' * 60)
    print('【3 - 生成行程】（调用 LLM — 整个流程唯一的一次）')
    print('=' * 60)

    interests_str = "、".join(interests)
    system = ("你是一个专业的旅行规划师。根据用户提供的城市和兴趣，"
              "生成一份详细的一日游行程。用中文回答。"
              "包含时间安排、具体地点、交通建议和小贴士。")
    prompt = f"请为我规划一天的{city}之旅。我的兴趣是：{interests_str}。"

    print(f'>>> System: {system[:60]}...')
    print(f'>>> Prompt: {prompt}')
    print(f'>>> 正在调用 LLM...')
    print()

    itinerary = call_llm(prompt, system)
    state["itinerary"] = itinerary

    print(itinerary)

    return state


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第11课 - 旅行规划器（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 前两步不调 LLM，就是 input() 收集用户输入')
    print('  - 只有第三步调一次 LLM，生成行程')
    print('  - State 累积 = 往 dict 里存值')
    print('  - 整个系统 = 2次input() + 1次API调用')
    print()

    while True:
        print()
        print('#' * 60)
        print('#  开始规划旅行')
        print('#' * 60)

        result = plan_trip()

        print()
        print('=' * 60)
        print('【规划完成】')
        print('=' * 60)
        print(f'  城市: {result["city"]}')
        print(f'  兴趣: {", ".join(result["interests"])}')
        print(f'  LLM调用次数: 1次（只有生成行程那一步）')
        print()

        again = input('\n再规划一次？(y/n): ').strip().lower()
        if again != 'y':
            print('再见，祝旅途愉快！')
            break
