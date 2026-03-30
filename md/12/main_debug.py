"""
第12课 - AI职业助手内部机制（不使用 LangGraph）

目的：让你看清两级路由+多轮对话的本质：
  1. 一级分类 = 一次 LLM 调用，返回数字 1/2/3/4
  2. 二级分类 = 又一次 LLM 调用，返回子类别
  3. 路由 = if-elif 匹配数字/关键词
  4. 多轮对话 = while 循环 + messages 列表不断追加
  5. 整个系统 = 2次分类调用 + N次对话调用

对比 main.py（LangGraph 框架版），理解：
  - add_conditional_edges(两层) → 就是两层 if-elif
  - 9个节点 → 就是9个函数
  - 多轮对话 → while True + input() + messages.append()
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_HISTORY = 10


def call_llm(prompt: str = "", system: str = "", messages: list = None) -> str:
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def trim_messages(messages: list) -> list:
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY * 2:
        non_system = non_system[-(MAX_HISTORY * 2):]
    return system_msgs + non_system


# ================================================================
#  多轮对话通用函数
# ================================================================

def multi_turn_chat(system_prompt: str, first_message: str,
                    user_label: str = "你", ai_label: str = "AI") -> str:
    """
    多轮对话的本质：while循环 + messages列表

    Java 类比：
        // 像 WebSocket 会话
        @OnMessage
        public void onMessage(Session session, String message) {
            messages.add(new UserMessage(message));
            String reply = llm.chat(messages);
            messages.add(new AIMessage(reply));
            session.getBasicRemote().sendText(reply);
        }
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_message},
    ]

    last_response = ""
    turn = 0

    while True:
        turn += 1
        messages = trim_messages(messages)

        print(f'    [第{turn}轮对话，messages长度={len(messages)}]')

        response = call_llm(messages=messages)
        messages.append({"role": "assistant", "content": response})
        last_response = response

        print(f'\n{ai_label}: {response}\n')

        user_input = input(f'{user_label}: ').strip()
        if user_input.lower() == 'exit':
            print('>>> 对话结束')
            break
        messages.append({"role": "user", "content": user_input})

    return last_response


# ================================================================
#  6个功能处理函数
# ================================================================

def handle_tutorial(query: str) -> str:
    """教程生成（单轮）"""
    print()
    print('    【教程生成】（单轮LLM调用）')
    response = call_llm(
        query,
        ("你是资深AI开发者和技术博主。根据用户需求生成高质量的中文教程，"
         "包含清晰的解释、Python代码示例和参考资源链接。")
    )
    print()
    print(response)
    return response


def handle_qa(query: str) -> str:
    """问答对话（多轮）"""
    print()
    print('    【问答对话】（多轮，输入 exit 退出）')
    return multi_turn_chat(
        system_prompt=("你是资深AI工程师，擅长解答各种AI技术问题。"
                       "用中文回答，给出有深度的专业解答。"),
        first_message=query,
    )


def handle_resume(query: str) -> str:
    """简历制作（多轮）"""
    print()
    print('    【简历制作】（多轮，输入 exit 退出）')
    return multi_turn_chat(
        system_prompt=("你是专业的简历顾问，擅长为AI/技术岗位制作简历。"
                       "分步骤向用户收集信息（技能、经验、项目），"
                       "4-5轮对话后生成完整的简历。用中文交流。"),
        first_message=query,
    )


def handle_interview_questions(query: str) -> str:
    """面试题库（多轮）"""
    print()
    print('    【面试题库】（多轮，输入 exit 退出）')
    return multi_turn_chat(
        system_prompt=("你是AI面试专家。根据用户需求提供面试题目列表，"
                       "包含题目、参考答案要点。可以追问用户想深入哪个方向。用中文回答。"),
        first_message=query,
    )


def handle_mock_interview(query: str) -> str:
    """模拟面试（多轮）"""
    print()
    print('    【模拟面试】（多轮，输入 exit 退出）')
    response = multi_turn_chat(
        system_prompt=("你是AI面试官，正在对候选人进行AI岗位的模拟面试。"
                       "每次只问一个问题，等候选人回答后再追问或出下一题。"
                       "面试结束时给出整体评价和改进建议。用中文交流。"),
        first_message="我准备好了，请开始面试。",
        user_label="候选人",
        ai_label="面试官",
    )
    return response


def handle_job_search(query: str) -> str:
    """求职建议（单轮）"""
    print()
    print('    【求职建议】（单轮LLM调用）')
    response = call_llm(
        query,
        ("你是AI行业求职顾问。根据用户需求提供求职建议，"
         "包括热门岗位、技能要求、薪资范围、求职策略等。用中文回答。")
    )
    print()
    print(response)
    return response


# ================================================================
#  完整流程
# ================================================================

def process_query(query: str) -> str:
    """
    AI职业助手的完整流程。

    Java 类比：
        @RestController
        public class CareerAssistant {

            @PostMapping("/query")
            public Response handle(String query) {
                // 一级路由（就是 switch-case）
                int category = classifier.classify(query);  // 1次LLM
                switch (category) {
                    case 1:
                        // 二级路由（又一个 switch-case）
                        String sub = subClassifier.classify(query);  // 又1次LLM
                        if ("教程".equals(sub)) return tutorialService.generate(query);
                        else return qaService.chat(query);
                    case 2:
                        return resumeService.create(query);
                    case 3:
                        String sub2 = subClassifier.classify(query);
                        if ("模拟".equals(sub2)) return interviewService.mock();
                        else return interviewService.questions(query);
                    case 4:
                        return jobService.search(query);
                }
            }
        }
    """

    llm_calls = 0

    # ==========================================
    # 第1步：一级分类（LLM调用 #1）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：一级分类】（LLM调用 #1）')
    print('=' * 60)

    system = ("将用户请求分类到以下类别之一，只回复数字：\n"
              "1: 学习AI技术\n"
              "2: 简历制作\n"
              "3: 面试准备\n"
              "4: 求职搜索\n\n"
              "示例：\n"
              "- \"什么是RAG？怎么实现？\" → 1\n"
              "- \"帮我写一份AI工程师简历\" → 2\n"
              "- \"面试一般会问什么？\" → 3\n"
              "- \"有什么AI相关的岗位？\" → 4\n\n"
              "只回复数字。")

    category = call_llm(query, system)
    llm_calls += 1

    print(f'>>> 用户输入: {query}')
    print(f'>>> 一级分类结果: {category}')
    print()

    category_names = {"1": "学习AI技术", "2": "简历制作", "3": "面试准备", "4": "求职搜索"}
    cat_num = next((k for k in category_names if k in category), "1")
    print(f'    一级路由（就是 if-elif）:')
    print(f'    if "{cat_num}" in "{category}":')
    print(f'        → {category_names[cat_num]}')

    # ==========================================
    # 第2步：根据一级分类，决定走哪条路
    # ==========================================

    if cat_num == "1":
        # 学习 → 需要二级分类
        print()
        print('=' * 60)
        print('【第2步：二级分类 - 学习】（LLM调用 #2）')
        print('=' * 60)

        sub_system = ("将用户请求分类为以下之一，只回复类别名：\n"
                      "- 教程：想要教程、文档、博客等\n"
                      "- 问答：有具体问题想问\n"
                      "默认为「问答」。只回复：教程 或 问答")

        sub = call_llm(query, sub_system)
        llm_calls += 1
        print(f'>>> 二级分类结果: {sub}')
        print()
        print(f'    二级路由（又一个 if-else）:')

        if '教程' in sub:
            print(f'    → 教程生成')
            result = handle_tutorial(query)
        else:
            print(f'    → 问答对话')
            result = handle_qa(query)

    elif cat_num == "2":
        # 简历 → 直接处理（无二级分类）
        print()
        print('>>> 无需二级分类，直接进入简历制作')
        result = handle_resume(query)

    elif cat_num == "3":
        # 面试 → 需要二级分类
        print()
        print('=' * 60)
        print('【第2步：二级分类 - 面试】（LLM调用 #2）')
        print('=' * 60)

        sub_system = ("将用户请求分类为以下之一，只回复类别名：\n"
                      "- 题库：想了解面试题目\n"
                      "- 模拟：想进行模拟面试\n"
                      "默认为「题库」。只回复：题库 或 模拟")

        sub = call_llm(query, sub_system)
        llm_calls += 1
        print(f'>>> 二级分类结果: {sub}')
        print()
        print(f'    二级路由（又一个 if-else）:')

        if '模拟' in sub:
            print(f'    → 模拟面试')
            result = handle_mock_interview(query)
        else:
            print(f'    → 面试题库')
            result = handle_interview_questions(query)

    else:
        # 求职 → 直接处理（无二级分类）
        print()
        print('>>> 无需二级分类，直接进入求职建议')
        result = handle_job_search(query)

    print()
    print(f'>>> 分类LLM调用次数: {llm_calls}（一级1次 + 二级0或1次）')
    return result


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第12课 - AI职业助手（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 两级路由 = 两层 if-elif，每层一次LLM分类调用')
    print('  - 多轮对话 = while True + messages.append() + input()')
    print('  - 6个功能模块 = 6个函数（4个多轮 + 2个单轮）')
    print('  - 整个系统 = 2次分类 + N次对话（N取决于用户聊多久）')
    print()
    print('功能：')
    print('  1. 学习 → 教程生成(单轮) / 问答对话(多轮)')
    print('  2. 简历制作(多轮)')
    print('  3. 面试 → 面试题库(多轮) / 模拟面试(多轮)')
    print('  4. 求职建议(单轮)')
    print()

    while True:
        print()
        print('#' * 60)
        query = input('请输入你的需求（输入 /quit 退出）: ').strip()
        if query == '/quit':
            print('再见，祝职业发展顺利！')
            break

        result = process_query(query)

        print()
        print('=' * 60)
        print('【处理完成】')
        print('=' * 60)
