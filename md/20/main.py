"""
第20课 - 企业梗图生成器（LangGraph 框架版）

核心概念：
  - 外部数据采集：用户输入企业信息作为流水线输入
  - 结构化中间数据：每步输出JSON，下一步精确解析
  - 创意生成流水线：分析→概念→模板→文案
  - 6节点线性流水线
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


def extract_json(text: str) -> dict:
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


# ========== 梗图模板库 ==========

MEME_TEMPLATES = [
    {"id": "drake", "name": "德雷克嫌弃/喜欢", "lines": 2,
     "description": "上面是不喜欢的，下面是喜欢的", "keywords": ["对比", "选择", "偏好"]},
    {"id": "distracted", "name": "分心男友", "lines": 2,
     "description": "被新事物吸引，忽略了原来的", "keywords": ["诱惑", "新旧对比", "注意力"]},
    {"id": "change_my_mind", "name": "改变我的想法", "lines": 1,
     "description": "提出一个大胆观点等人反驳", "keywords": ["观点", "争议", "自信"]},
    {"id": "this_is_fine", "name": "一切正常（着火狗）", "lines": 1,
     "description": "明明很糟糕但假装没事", "keywords": ["自嘲", "危机", "淡定"]},
    {"id": "expanding_brain", "name": "脑洞扩展", "lines": 4,
     "description": "从普通到高级的递进", "keywords": ["升级", "进化", "层次"]},
    {"id": "two_buttons", "name": "两难选择", "lines": 2,
     "description": "两个选项都想要，难以抉择", "keywords": ["选择困难", "两难", "纠结"]},
]


# ========== State ==========

class State(TypedDict):
    company_info: str           # 用户输入的企业信息
    company_analysis: dict      # 结构化企业分析
    meme_concepts: list         # 梗图概念列表
    selected_templates: list    # 匹配的模板
    meme_texts: list            # 最终文案
    output: str                 # 最终输出


# ========== 节点1：信息采集 ==========

def collect_info_node(state: State):
    print()
    print('=' * 60)
    print('【1 - 信息采集】')
    print('=' * 60)
    print(f'>>> 企业信息: {state["company_info"][:100]}...')
    return {}


# ========== 节点2：企业分析 ==========

def analyze_company_node(state: State):
    print()
    print('=' * 60)
    print('【2 - 企业分析】（结构化输出）')
    print('=' * 60)

    system = ("分析企业信息，提取以下要素。返回JSON：\n"
              '{"tone": "品牌调性", "target_audience": "目标受众", '
              '"value_proposition": "核心卖点", '
              '"key_products": ["产品1", "产品2"], '
              '"brand_personality": "品牌个性"}')

    result = extract_json(call_llm(state["company_info"], system))
    if not result:
        result = {"tone": "专业", "target_audience": "用户", "value_proposition": "优质服务",
                  "key_products": ["核心产品"], "brand_personality": "可靠"}

    print(f'>>> 企业分析:')
    for k, v in result.items():
        print(f'    {k}: {v}')

    return {"company_analysis": result}


# ========== 节点3：概念生成 ==========

def generate_concepts_node(state: State):
    print()
    print('=' * 60)
    print('【3 - 梗图概念生成】')
    print('=' * 60)

    analysis = state["company_analysis"]
    system = ("你是创意营销专家。根据企业分析生成3个梗图创意概念。\n"
              "每个概念需要：消息（想传达的核心点）、情绪（搞笑/自嘲/励志等）、受众关联（为什么目标受众会喜欢）\n"
              '返回JSON：{"concepts": [{"message": "...", "emotion": "...", "audience_relevance": "..."}]}')

    prompt = json.dumps(analysis, ensure_ascii=False)
    result = extract_json(call_llm(prompt, system))
    concepts = result.get("concepts", [{"message": "默认概念", "emotion": "幽默", "audience_relevance": "通用"}])

    print(f'>>> 生成 {len(concepts)} 个概念:')
    for i, c in enumerate(concepts, 1):
        print(f'    {i}. {c.get("message", "")} ({c.get("emotion", "")})')

    return {"meme_concepts": concepts}


# ========== 节点4：模板匹配 ==========

def select_templates_node(state: State):
    print()
    print('=' * 60)
    print('【4 - 模板匹配】')
    print('=' * 60)

    templates_info = json.dumps(MEME_TEMPLATES, ensure_ascii=False)
    concepts_info = json.dumps(state["meme_concepts"], ensure_ascii=False)

    system = ("你是梗图专家。为每个概念匹配最合适的梗图模板。\n"
              f"可用模板：{templates_info}\n\n"
              '返回JSON：{"matches": [{"concept_index": 0, "template_id": "drake", "reason": "匹配原因"}]}')

    result = extract_json(call_llm(concepts_info, system))
    matches = result.get("matches", [])

    # 兜底
    if not matches:
        for i in range(len(state["meme_concepts"])):
            matches.append({"concept_index": i, "template_id": MEME_TEMPLATES[i % len(MEME_TEMPLATES)]["id"]})

    selected = []
    for m in matches:
        idx = m.get("concept_index", 0)
        tid = m.get("template_id", "drake")
        template = next((t for t in MEME_TEMPLATES if t["id"] == tid), MEME_TEMPLATES[0])
        concept = state["meme_concepts"][idx] if idx < len(state["meme_concepts"]) else state["meme_concepts"][0]
        selected.append({"template": template, "concept": concept})
        print(f'    概念 "{concept.get("message", "")[:20]}" → 模板 "{template["name"]}"')

    return {"selected_templates": selected}


# ========== 节点5：文案生成 ==========

def generate_text_node(state: State):
    print()
    print('=' * 60)
    print('【5 - 梗图文案生成】')
    print('=' * 60)

    meme_texts = []
    for i, item in enumerate(state["selected_templates"], 1):
        template = item["template"]
        concept = item["concept"]

        system = (f"你是梗图文案大师。为「{template['name']}」模板生成文案。\n"
                  f"模板说明：{template['description']}\n"
                  f"文字行数：{template['lines']}行\n"
                  f"概念：{concept.get('message', '')}\n"
                  f"情绪：{concept.get('emotion', '')}\n\n"
                  f"要求：简短有力、幽默、符合中文互联网风格。\n"
                  f'返回JSON：{{"lines": ["第一行文字", "第二行文字"]}}')

        result = extract_json(call_llm("生成梗图文案", system))
        lines = result.get("lines", [concept.get("message", "梗图文案")])

        meme_texts.append({
            "template_name": template["name"],
            "template_desc": template["description"],
            "lines": lines,
            "concept": concept,
        })

        print(f'    梗图{i} [{template["name"]}]:')
        for j, line in enumerate(lines):
            print(f'      行{j+1}: {line}')

    return {"meme_texts": meme_texts}


# ========== 节点6：输出展示 ==========

def output_node(state: State):
    print()
    print('=' * 60)
    print('【6 - 最终输出】')
    print('=' * 60)

    output_parts = []
    for i, meme in enumerate(state["meme_texts"], 1):
        part = (f"梗图 #{i}\n"
                f"  模板: {meme['template_name']}\n"
                f"  说明: {meme['template_desc']}\n"
                f"  文案:\n")
        for j, line in enumerate(meme["lines"]):
            part += f"    {'上' if j == 0 else '下'}文: {line}\n"
        part += f"  情绪: {meme['concept'].get('emotion', '')}\n"
        part += f"  受众: {meme['concept'].get('audience_relevance', '')}"
        output_parts.append(part)

    output = "\n\n".join(output_parts)
    print(output)

    return {"output": output}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("collect_info", collect_info_node)
    workflow.add_node("analyze_company", analyze_company_node)
    workflow.add_node("generate_concepts", generate_concepts_node)
    workflow.add_node("select_templates", select_templates_node)
    workflow.add_node("generate_text", generate_text_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("collect_info")
    workflow.add_edge("collect_info", "analyze_company")
    workflow.add_edge("analyze_company", "generate_concepts")
    workflow.add_edge("generate_concepts", "select_templates")
    workflow.add_edge("select_templates", "generate_text")
    workflow.add_edge("generate_text", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第20课 - 企业梗图生成器')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()

    examples = [
        "我们是一家做AI编程助手的公司，产品叫CodePilot，目标用户是程序员，主打提效50%",
        "我们是一家奶茶品牌，叫「茶里有料」，目标年轻人，主打真材实料、高颜值",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    info = input('请输入企业/产品信息（回车用示例1）: ').strip()
    if not info:
        info = examples[0]
        print(f'>>> 使用: {info}')

    print()
    print('#' * 60)
    print('#  开始生成梗图')
    print('#' * 60)

    result = app.invoke({
        "company_info": info,
        "company_analysis": {},
        "meme_concepts": [],
        "selected_templates": [],
        "meme_texts": [],
        "output": "",
    })

    print()
    print('#' * 60)
    print('#  梗图生成完成')
    print('#' * 60)
