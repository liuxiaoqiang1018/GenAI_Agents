"""
第20课 - 企业梗图生成器内部机制（不使用 LangGraph）

目的：让你看清结构化流水线的本质：
  1. 采集 = 用户输入（或网页抓取）
  2. 分析 = call_llm(企业信息) → JSON结构化
  3. 概念 = call_llm(分析结果JSON) → 概念列表JSON
  4. 模板 = call_llm(概念+模板库) → 匹配结果JSON
  5. 文案 = for概念: call_llm(模板+概念) → 文案JSON
  6. 输出 = 格式化打印

对比 main.py，理解：
  - 6个节点 → 6个函数
  - 结构化输出 → JSON在函数间传递（像Java的DTO）
  - 线性流水线 → 顺序调用，没有分支和循环
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
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.8},
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


# 梗图模板库（同main.py）
MEME_TEMPLATES = [
    {"id": "drake", "name": "德雷克嫌弃/喜欢", "lines": 2, "description": "上面不喜欢，下面喜欢"},
    {"id": "distracted", "name": "分心男友", "lines": 2, "description": "被新事物吸引"},
    {"id": "change_my_mind", "name": "改变我的想法", "lines": 1, "description": "大胆观点"},
    {"id": "this_is_fine", "name": "一切正常（着火狗）", "lines": 1, "description": "假装没事"},
    {"id": "expanding_brain", "name": "脑洞扩展", "lines": 4, "description": "从普通到高级递进"},
    {"id": "two_buttons", "name": "两难选择", "lines": 2, "description": "两个都想要"},
]


# ================================================================
#  完整流程
# ================================================================

def generate_memes(company_info: str) -> dict:
    """
    企业梗图生成的完整流程。

    Java 类比：
        @Service
        public class MemeGeneratorService {
            public List<Meme> generate(String companyInfo) {
                // DTO在Service层间传递（结构化数据）
                CompanyAnalysis analysis = analyzer.analyze(companyInfo);  // → DTO
                List<MemeConcept> concepts = conceptGen.generate(analysis); // → List<DTO>
                List<TemplateMatch> matches = matcher.match(concepts, templates); // → List<DTO>
                List<MemeText> texts = textGen.generate(matches);           // → List<DTO>
                return texts;
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：采集（不调LLM）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：信息采集】')
    print('=' * 60)
    print(f'>>> 企业信息: {company_info[:80]}...')

    # ==========================================
    # 第2步：企业分析（→ 结构化JSON）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：企业分析】→ 输出结构化JSON')
    print('=' * 60)

    system = ("分析企业信息。返回JSON：\n"
              '{"tone":"调性","target_audience":"受众",'
              '"value_proposition":"卖点","key_products":["产品"],"brand_personality":"个性"}')
    analysis = extract_json(call_llm(company_info, system))
    total_llm_calls += 1

    if not analysis:
        analysis = {"tone": "专业", "target_audience": "用户", "value_proposition": "优质",
                     "key_products": ["产品"], "brand_personality": "可靠"}
    print(f'>>> 分析结果（JSON）: {json.dumps(analysis, ensure_ascii=False)[:200]}')

    # ==========================================
    # 第3步：概念生成（JSON → JSON）
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：概念生成】JSON输入 → JSON输出')
    print('=' * 60)

    system = ('生成3个梗图创意概念。返回JSON：\n'
              '{"concepts":[{"message":"核心点","emotion":"情绪","audience_relevance":"受众关联"}]}')
    result = extract_json(call_llm(json.dumps(analysis, ensure_ascii=False), system))
    concepts = result.get("concepts", [{"message": "默认", "emotion": "幽默", "audience_relevance": "通用"}])
    total_llm_calls += 1

    for i, c in enumerate(concepts, 1):
        print(f'    {i}. {c.get("message","")} ({c.get("emotion","")})')

    # ==========================================
    # 第4步：模板匹配（JSON → JSON）
    # ==========================================
    print()
    print('=' * 60)
    print('【第4步：模板匹配】概念JSON + 模板库 → 匹配JSON')
    print('=' * 60)

    system = (f"可用模板：{json.dumps(MEME_TEMPLATES, ensure_ascii=False)}\n"
              '为每个概念匹配模板。返回JSON：'
              '{"matches":[{"concept_index":0,"template_id":"drake"}]}')
    result = extract_json(call_llm(json.dumps(concepts, ensure_ascii=False), system))
    matches = result.get("matches", [{"concept_index": i, "template_id": MEME_TEMPLATES[i % len(MEME_TEMPLATES)]["id"]}
                                     for i in range(len(concepts))])
    total_llm_calls += 1

    selected = []
    for m in matches:
        idx = m.get("concept_index", 0)
        tid = m.get("template_id", "drake")
        template = next((t for t in MEME_TEMPLATES if t["id"] == tid), MEME_TEMPLATES[0])
        concept = concepts[idx] if idx < len(concepts) else concepts[0]
        selected.append({"template": template, "concept": concept})
        print(f'    "{concept.get("message","")[:20]}" → {template["name"]}')

    # ==========================================
    # 第5步：文案生成（for循环，每个概念一次LLM）
    # ==========================================
    print()
    print('=' * 60)
    print('【第5步：文案生成】for循环，每个概念一次LLM调用')
    print('=' * 60)

    meme_texts = []
    for i, item in enumerate(selected, 1):
        t = item["template"]
        c = item["concept"]
        system = (f"为「{t['name']}」模板（{t['description']}，{t['lines']}行）"
                  f"生成文案。概念：{c.get('message','')}。"
                  f'返回JSON：{{"lines":["第一行","第二行"]}}')
        result = extract_json(call_llm("生成文案", system))
        lines = result.get("lines", [c.get("message", "梗图")])
        total_llm_calls += 1

        meme_texts.append({"template": t["name"], "lines": lines, "concept": c})
        print(f'    梗图{i} [{t["name"]}]: {" / ".join(lines)}')

    # ==========================================
    # 第6步：输出
    # ==========================================
    print()
    print('=' * 60)
    print('【第6步：最终输出】')
    print('=' * 60)

    for i, m in enumerate(meme_texts, 1):
        print(f'\n  梗图 #{i} [{m["template"]}]')
        for j, line in enumerate(m["lines"]):
            print(f'    {"上" if j == 0 else "下"}文: {line}')

    print(f'\n>>> LLM调用: {total_llm_calls}次（分析1+概念1+匹配1+文案{len(selected)}）')

    return {"meme_texts": meme_texts, "total_llm_calls": total_llm_calls}


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第20课 - 企业梗图生成器（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 结构化流水线 = JSON在函数间传递（像Java DTO）')
    print('  - 6个节点 = 6个函数顺序调用')
    print('  - 创意在约束中产生（模板限定了格式）')
    print()

    examples = [
        "AI编程助手公司CodePilot，目标用户程序员，主打提效50%",
        "奶茶品牌「茶里有料」，目标年轻人，主打真材实料高颜值",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    info = input('企业信息（回车用示例1）: ').strip()
    if not info:
        info = examples[0]
        print(f'>>> 使用: {info}')

    result = generate_memes(info)

    print()
    print('#' * 60)
    print(f'#  梗图生成完成！LLM调用 {result["total_llm_calls"]} 次')
    print('#' * 60)
