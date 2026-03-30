"""
第10课 - 作文评分系统内部机制（不使用 LangGraph）

目的：让你看清条件提前终止的本质：
  1. 每个维度评分 = 一次 LLM 调用 + 正则提取分数
  2. 条件终止 = if score <= 门槛: break
  3. 最终得分 = 加权平均（未评的维度=0分）
  4. 整个系统 = 最多4次LLM调用 + 一个循环

对比 main.py（LangGraph 框架版），理解：
  - add_conditional_edges → 就是 if score <= threshold: 跳到最终计算
  - 5个节点 → 就是5个函数按顺序调用
  - 提前终止 → 就是 break 跳出评分循环
"""

import os
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
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extract_score(content: str) -> float:
    """从 LLM 回复中提取分数"""
    match = re.search(r'分数[:：]\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    match = re.search(r'Score[:：]\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"无法提取分数: {content[:100]}")


# ================================================================
#  完整流程
# ================================================================

def grade_essay(essay: str) -> dict:
    """
    作文评分的完整流程。

    Java 类比：
        @Service
        public class EssayGradingService {
            // 评分管道：每一步有门槛，不达标就短路
            public GradingResult grade(String essay) {
                double relevance = relevanceChecker.check(essay);
                if (relevance <= 0.5) return calculateFinal(relevance, 0, 0, 0);

                double grammar = grammarChecker.check(essay);
                if (grammar <= 0.6) return calculateFinal(relevance, grammar, 0, 0);

                double structure = structureAnalyzer.analyze(essay);
                if (structure <= 0.7) return calculateFinal(relevance, grammar, structure, 0);

                double depth = depthEvaluator.evaluate(essay);
                return calculateFinal(relevance, grammar, structure, depth);
            }
        }
    """

    scores = {
        "relevance_score": 0.0,
        "grammar_score": 0.0,
        "structure_score": 0.0,
        "depth_score": 0.0,
    }

    # 评分流水线：每个步骤定义 (名称, system_prompt, 分数key, 门槛)
    pipeline = [
        (
            "内容相关性",
            ("你是作文评分专家。评估以下作文的内容相关性：是否切题、是否围绕主题展开。\n"
             "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。"),
            "relevance_score",
            0.5,   # 门槛：> 0.5 才继续
        ),
        (
            "语法检查",
            ("你是语言专家。评估以下作文的语法和语言表达质量。\n"
             "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。"),
            "grammar_score",
            0.6,   # 门槛：> 0.6 才继续
        ),
        (
            "结构分析",
            ("你是写作结构专家。评估以下作文的组织架构、段落逻辑和行文流畅度。\n"
             "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。"),
            "structure_score",
            0.7,   # 门槛：> 0.7 才继续
        ),
        (
            "深度分析",
            ("你是学术评审专家。评估以下作文的分析深度、批判性思维和独到见解。\n"
             "给出 0 到 1 之间的分数。回复必须以「分数: 」开头，后跟数字，然后给出解释。"),
            "depth_score",
            None,  # 最后一步，无门槛
        ),
    ]

    terminated_at = None
    llm_calls = 0

    for i, (name, system, key, threshold) in enumerate(pipeline, 1):
        # ==========================================
        print()
        print('=' * 60)
        print(f'【{i} - {name}】')
        print('=' * 60)

        result = call_llm(essay, system)
        llm_calls += 1

        try:
            score = extract_score(result)
        except ValueError as e:
            print(f'>>> 提取分数失败: {e}')
            score = 0.0

        scores[key] = score
        print(f'>>> 得分: {score}')
        print(f'>>> LLM评语: {result[:200]}')

        # 检查门槛 —— 这就是 add_conditional_edges 的本质
        if threshold is not None:
            print()
            print(f'    门槛判断（就是 if-else）:')
            print(f'    if {key} ({score}) > {threshold}:')

            if score <= threshold:
                print(f'        → 不达标！提前终止，跳到最终评分')
                terminated_at = name
                break
            else:
                print(f'        → 达标，继续下一步')

    # ==========================================
    # 最终评分
    # ==========================================
    print()
    print('=' * 60)
    print('【最终评分计算】')
    print('=' * 60)

    if terminated_at:
        print(f'>>> 提前终止于: {terminated_at}')
    else:
        print(f'>>> 完成全部4个维度评分')
    print(f'>>> LLM调用次数: {llm_calls}（最多4次，提前终止可节省调用）')
    print()

    final = (
        scores["relevance_score"] * 0.3 +
        scores["grammar_score"]   * 0.2 +
        scores["structure_score"] * 0.2 +
        scores["depth_score"]     * 0.3
    )

    print(f'    加权计算（就是乘法加法）:')
    print(f'    相关性 {scores["relevance_score"]:.2f} × 0.3 = {scores["relevance_score"] * 0.3:.2f}')
    print(f'    语法   {scores["grammar_score"]:.2f} × 0.2 = {scores["grammar_score"] * 0.2:.2f}')
    print(f'    结构   {scores["structure_score"]:.2f} × 0.2 = {scores["structure_score"] * 0.2:.2f}')
    print(f'    深度   {scores["depth_score"]:.2f} × 0.3 = {scores["depth_score"] * 0.3:.2f}')
    print(f'    ──────────────────────')
    print(f'    最终得分: {final:.2f}')

    return {
        "essay": essay,
        **scores,
        "final_score": final,
        "terminated_at": terminated_at,
        "llm_calls": llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第10课 - 作文评分系统（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 条件提前终止 = for循环里的 if + break')
    print('  - 4个评分节点 = 4次LLM调用（提前终止可能只需1-3次）')
    print('  - 加权平均 = 乘法+加法，未评项=0自动拉低总分')
    print('  - 整个系统去掉框架就是一个带短路的循环')
    print()

    # 示例作文
    examples = [
        {
            "name": "优秀作文（预期完成全部评分）",
            "text": """
            人工智能对现代社会的影响

            人工智能已经成为我们日常生活的重要组成部分，在医疗、金融和交通等多个领域
            引发了深刻变革。本文将探讨人工智能对现代社会的深远影响。

            在医疗领域，AI驱动的诊断工具能够以极高的准确率分析医学影像，使得疾病
            能够被更早发现。AI算法能处理海量医学数据，有望推动药物研发的突破。

            在金融领域，机器学习算法能实时检测欺诈行为，智能投顾让更多人享受到
            专业的理财服务。交通方面，自动驾驶有望减少人为事故。

            然而，AI也带来了岗位替代、隐私侵犯和算法偏见等挑战。我们需要在技术
            进步与伦理考量之间取得平衡。
            """
        },
        {
            "name": "跑题作文（预期在相关性阶段终止）",
            "text": """
            我最喜欢的食物

            我最喜欢吃火锅。冬天和朋友一起涮火锅，热气腾腾特别暖和。
            毛肚涮七上八下，鸭肠要卷着涮，配上麻酱简直完美。

            除了火锅还有烧烤，夏天撸串喝啤酒也很开心。

            总之，美食让生活更美好。
            """
        },
    ]

    for ex in examples:
        print()
        print('#' * 60)
        print(f'#  测试: {ex["name"]}')
        print('#' * 60)

        result = grade_essay(ex["text"])

        print()
        print('=' * 60)
        print('【最终结果】')
        print('=' * 60)
        print(f'  相关性: {result["relevance_score"]:.2f}')
        print(f'  语法:   {result["grammar_score"]:.2f}')
        print(f'  结构:   {result["structure_score"]:.2f}')
        print(f'  深度:   {result["depth_score"]:.2f}')
        print(f'  最终得分: {result["final_score"]:.2f}')
        if result["terminated_at"]:
            print(f'  提前终止于: {result["terminated_at"]}')
        print(f'  LLM调用: {result["llm_calls"]}次')
        print()

    # 交互模式
    print('\n输入作文内容（多行输入，单独一行输入 END 结束作文，输入 /quit 退出）\n')
    while True:
        print('请输入作文:')
        lines = []
        while True:
            line = input()
            if line.strip() == '/quit':
                print('再见！')
                exit()
            if line.strip() == 'END':
                break
            lines.append(line)
        essay = '\n'.join(lines)
        if essay.strip():
            result = grade_essay(essay)
            print()
            print('=' * 60)
            print('【最终结果】')
            print('=' * 60)
            print(f'  相关性: {result["relevance_score"]:.2f}')
            print(f'  语法:   {result["grammar_score"]:.2f}')
            print(f'  结构:   {result["structure_score"]:.2f}')
            print(f'  深度:   {result["depth_score"]:.2f}')
            print(f'  最终得分: {result["final_score"]:.2f}')
            if result["terminated_at"]:
                print(f'  提前终止于: {result["terminated_at"]}')
            print(f'  LLM调用: {result["llm_calls"]}次')
            print()
