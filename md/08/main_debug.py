"""
第8课 - 费曼学习导师内部机制演示（不使用 LangGraph）

目的：让你看清"人机交互学习循环"的本质：
  1. 生成检查点 = 一次 LLM 调用
  2. 检查点循环 = for 循环
  3. 出题 = 一次 LLM 调用
  4. 等用户回答 = input()
  5. 验证 = 一次 LLM 调用 + 条件判断
  6. 费曼教学 = 一次 LLM 调用
  7. 路由决策 = if-else

对比 main.py（LangGraph 框架版），理解：
  - Human-in-the-Loop → 就是 input()
  - 检查点循环 → 就是 for 循环
  - 条件路由 → 就是 if understanding < 0.7
"""

import os
import json
import re
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
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_from_text(text: str, default=None):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```json\s*([\s\S]*?)```', text)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        match = re.search(r'[\[{][\s\S]*[\]}]', text)
        if match:
            try: return json.loads(match.group())
            except: pass
    return default


# ================================================================
#  完整流程
# ================================================================

def run_learning_session(topic: str, goals: str, context: str = ""):
    """
    费曼学习导师的完整流程。

    Java 类比：
        public void learnTopic(String topic, String goals) {
            List<Checkpoint> checkpoints = generateCheckpoints(topic, goals);

            for (int i = 0; i < checkpoints.size(); i++) {
                String question = generateQuestion(checkpoints.get(i));
                String answer = getUserInput(question);  // Human-in-the-Loop
                VerifyResult result = verifyAnswer(answer, checkpoints.get(i));

                if (result.getLevel() < 0.7) {
                    feynmanTeach(checkpoints.get(i), result);
                }
            }
        }
    """

    # ==========================================
    # 第1步：生成检查点
    # ==========================================
    print()
    print('=' * 60)
    print('【1 - 生成学习检查点】')
    print('=' * 60)

    system = """你是学习导师。根据主题和目标生成3个学习检查点。
用JSON数组格式：[{"description":"描述","criteria":["标准1","标准2"],"verification":"验证方式"}]"""

    prompt = f"主题：{topic}\n目标：{goals}"
    if context:
        prompt += f"\n参考材料：{context[:500]}"

    response = call_llm(prompt, system)
    checkpoints = parse_json_from_text(response, [])
    if isinstance(checkpoints, dict):
        checkpoints = checkpoints.get("checkpoints", [checkpoints])
    if not checkpoints:
        checkpoints = [{"description": "理解核心概念", "criteria": ["能用自己的话解释"], "verification": "简述"}]

    print(f'>>> 生成了 {len(checkpoints)} 个检查点:')
    for i, cp in enumerate(checkpoints):
        print(f'    {i+1}. {cp.get("description", "?")}')
        for c in cp.get("criteria", []):
            print(f'       - {c}')
    print()

    # ==========================================
    # 第2步：检查点循环 — 这就是 LangGraph 的循环边
    # ==========================================
    print('=' * 60)
    print('【2 - 开始检查点循环】')
    print(f'    这就是 LangGraph 中 next_checkpoint → generate_question 的循环')
    print(f'    本质就是 for i in range(len(checkpoints)):')
    print('=' * 60)

    for i, cp in enumerate(checkpoints):
        description = cp.get("description", "")
        criteria = cp.get("criteria", [])
        verification = cp.get("verification", "")

        print()
        print(f'  {"=" * 54}')
        print(f'  检查点 {i+1}/{len(checkpoints)}: {description}')
        print(f'  {"=" * 54}')

        # ==========================================
        # 第3步：出题
        # ==========================================
        print()
        print(f'  【3 - 出题】')

        q_system = "你是学习导师。根据检查点出一道验证理解的题。只输出题目。"
        q_prompt = f"检查点：{description}\n标准：{json.dumps(criteria, ensure_ascii=False)}\n验证方式：{verification}"
        question = call_llm(q_prompt, q_system)
        print(f'  >>> 题目: {question}')

        # ==========================================
        # 第4步：等用户回答 — 这就是 Human-in-the-Loop
        # ==========================================
        print()
        print(f'  【4 - 等待回答（Human-in-the-Loop）】')
        print(f'      LangGraph: interrupt_before=["user_answer"]')
        print(f'      本质就是: answer = input()')
        print()
        print(f'  题目: {question}')
        answer = input('  你的回答: ').strip()
        if not answer:
            answer = "不知道"

        # ==========================================
        # 第5步：验证
        # ==========================================
        print()
        print(f'  【5 - 验证回答】')

        v_system = """评估学生回答。用JSON回复：{"understanding_level": 0.0~1.0, "feedback": "反馈"}"""
        v_prompt = f"题目：{question}\n回答：{answer}\n检查点：{description}\n标准：{json.dumps(criteria, ensure_ascii=False)}"
        if context:
            v_prompt += f"\n参考材料：{context[:300]}"

        v_response = call_llm(v_prompt, v_system)
        v_result = parse_json_from_text(v_response, {"understanding_level": 0.5, "feedback": v_response})

        level = float(v_result.get("understanding_level", 0.5))
        feedback = v_result.get("feedback", "")

        print(f'  >>> 理解度: {level:.0%}')
        print(f'  >>> 反馈: {feedback[:200]}')

        # ==========================================
        # 第6步：条件路由 — 这就是 route_verification
        # ==========================================
        print()
        print(f'  【6 - 路由决策】')
        print(f'      if understanding_level < 0.7: → 费曼教学')
        print(f'      else: → 下一个检查点')

        if level < 0.7:
            # ==========================================
            # 第7步：费曼教学
            # ==========================================
            print()
            print(f'  【7 - 费曼教学】理解度不足，用简单方式重新解释')

            t_system = """用费曼学习法重新教学。用JSON回复：
{"explanation":"简单解释","analogies":["类比1"],"key_concepts":["概念1"]}"""
            t_prompt = f"检查点：{description}\n学生回答：{answer}\n反馈：{feedback}"
            if context:
                t_prompt += f"\n材料：{context[:300]}"

            t_response = call_llm(t_prompt, t_system)
            teaching = parse_json_from_text(t_response, {"explanation": t_response})

            print(f'  >>> 简化解释: {teaching.get("explanation", "")[:300]}')
            if teaching.get("analogies"):
                print(f'  >>> 类比:')
                for a in teaching["analogies"]:
                    print(f'      - {a}')
            if teaching.get("key_concepts"):
                print(f'  >>> 必记概念:')
                for c in teaching["key_concepts"]:
                    print(f'      - {c}')
        else:
            print(f'  >>> 通过！理解度 {level:.0%} ≥ 70%')

        # 继续下一个检查点（for 循环自动处理）

    # ==========================================
    # 完成
    # ==========================================
    print()
    print('=' * 60)
    print('【学习完成！】')
    print('=' * 60)
    print(f'主题: {topic}')
    print(f'完成检查点: {len(checkpoints)}/{len(checkpoints)}')
    print()


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第8课 - 费曼学习导师内部机制演示（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('这个程序展示"人机交互学习循环"的本质：')
    print('  - Human-in-the-Loop = input()')
    print('  - 检查点循环 = for 循环')
    print('  - 条件路由 = if understanding < 0.7')
    print()

    topic = input('学习主题（回车用默认"Python装饰器"）: ').strip() or "Python装饰器"
    goals = input('学习目标（回车用默认）: ').strip() or "理解装饰器的本质，能手写一个带参数的装饰器"
    context = input('学习材料（可选，回车跳过）: ').strip()

    print()
    print('#' * 60)
    print(f'#  开始学习: {topic}')
    print('#' * 60)

    run_learning_session(topic, goals, context)
