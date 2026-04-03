"""
第37课：播客自动生成代理（透明调试版）

不使用任何框架，纯手写展示三层图嵌套 + 多轮访谈 + Map-Reduce 的内部机制。
"""

import os
import re
import sys
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3
MAX_TURNS = 2  # 减少轮数加快调试


def call_llm(messages: list, temperature: float = 0.7) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应（公益站可能过载）")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"无choices字段: {str(data)[:200]}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("LLM返回空内容")
            return content
        except Exception as e:
            print(f"    !! LLM失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES + 1:
                wait = (attempt + 1) * 5
                print(f"    等待{wait}秒后重试...")
                time.sleep(wait)
            else:
                raise


def main():
    topic = "人工智能如何改变教育的未来"

    print("=" * 70)
    print("  播客自动生成代理（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  主题: {topic}")

    # ==============================================================
    # 【规划子图】生成关键词 → 生成子话题
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【规划子图】这是嵌套在主图中的第一个子图")
    print(f"  原教程: plan_builder = StateGraph(Planning)")
    print(f"  节点: Keywords → Structure")
    print(f"{'=' * 70}")

    print(f"\n  --- 节点1: 生成关键词 ---")
    kw_prompt = f"请为播客主题'{topic}'生成5个关键词，用逗号分隔，只输出关键词。"
    print(f"  >>> Prompt: {kw_prompt}")
    keywords_str = call_llm([{"role": "user", "content": kw_prompt}], temperature=0.3)
    keywords = [k.strip() for k in keywords_str.split(",")]
    print(f"  >>> 关键词: {keywords}")

    print(f"\n  --- 节点2: 生成子话题 ---")
    st_prompt = f"请为播客主题'{topic}'生成3个子话题，关键词: {', '.join(keywords)}。每行一个，只输出子话题名称。"
    print(f"  >>> Prompt: {st_prompt}")
    subtopics_str = call_llm([{"role": "user", "content": st_prompt}], temperature=0.3)
    subtopics = [s.strip() for s in subtopics_str.strip().split("\n") if s.strip()][:3]
    print(f"  >>> 子话题: {subtopics}")

    print(f"\n  [规划子图完成] → 返回主图")

    # ==============================================================
    # 【Send API 并行】对每个子话题启动访谈子图
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【Send API 并行】对每个子话题启动独立访谈")
    print(f"  原教程: [Send('Create podcast', {{topic: subtopic}}) for subtopic in subtopics]")
    print(f"  本质: 对列表中的每个元素启动一个子图实例（类似 map）")
    print(f"  本课简化为顺序执行（原教程是并行的）")
    print(f"{'=' * 70}")

    sections = []

    for idx, subtopic in enumerate(subtopics, 1):
        # ==============================================================
        # 【访谈子图】多轮主持人-专家对话
        # ==============================================================
        print(f"\n{'=' * 70}")
        print(f"【访谈子图 {idx}/{len(subtopics)}】子话题: {subtopic}")
        print(f"  原教程: interview_builder = StateGraph(InterviewState)")
        print(f"  流程: 主持人提问 → [并行]Web+Wiki搜索 → 专家回答 → [够了?] → 循环/保存")
        print(f"{'=' * 70}")

        conversation = []

        for turn in range(MAX_TURNS):
            # 主持人提问
            print(f"\n  --- 第{turn+1}轮 主持人提问 ---")
            host_system = (
                f"你是播客主持人，正在采访'{subtopic}'领域的专家。"
                f"{'请介绍话题并提出问题。' if turn == 0 else '追问更深入的问题。'}"
                f"{'用\"非常感谢\"结束访谈。' if turn == MAX_TURNS - 1 else ''}"
                f"用中文，简洁。"
            )
            if conversation:
                last_answer = conversation[-1]["content"]
                host_msgs = [
                    {"role": "system", "content": host_system},
                    {"role": "user", "content": f"专家上轮回答：{last_answer}\n\n请继续提问。"}
                ]
            else:
                host_msgs = [{"role": "system", "content": host_system},
                           {"role": "user", "content": f"开始关于'{subtopic}'的访谈"}]

            question = call_llm(host_msgs, 0.5)
            print(f"  主持人: {question[:150]}...")
            conversation.append({"role": "assistant", "content": question})

            if "感谢" in question and turn > 0:
                break

            # 专家回答（原教程先搜索 Web+Wiki，再基于搜索结果回答）
            print(f"\n  --- 第{turn+1}轮 专家回答 ---")
            print(f"  (原教程在此处并行执行 Web搜索 + Wiki搜索，将结果注入 context)")
            print(f"  (本课简化: 专家直接基于知识回答)")

            expert_system = f"你是'{subtopic}'的专家，正在接受播客采访。提供具体例子和数据。用中文，300字以内。"
            expert_msgs = [
                {"role": "system", "content": expert_system},
                {"role": "user", "content": f"主持人问你：{question}"}
            ]
            answer = call_llm(expert_msgs, 0.5)
            print(f"  专家: {answer[:150]}...")
            conversation.append({"role": "user", "content": answer})

        # 整理为脚本片段（对应 write_section 节点）
        print(f"\n  --- 整理为脚本片段 ---")
        script_prompt = f"将以下访谈整理为约200字的播客脚本片段，标注主持人和专家。子话题: {subtopic}\n\n"
        for msg in conversation:
            role = "主持人" if msg["role"] == "assistant" else "专家"
            script_prompt += f"{role}: {msg['content'][:200]}\n\n"

        section = call_llm([{"role": "user", "content": script_prompt}])
        sections.append(section)
        print(f"  脚本片段: {section[:200]}...")

    # ==============================================================
    # 【主图汇聚】Map-Reduce: 并行写开场/正文/结语 → 合并
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【汇聚阶段】三路并行写作 → 合并")
    print(f"  原教程: Write introduction / Write report / Write conclusion 三个节点并行")
    print(f"  最后 Finalize podcast 节点合并")
    print(f"{'=' * 70}")

    all_sections = "\n\n---\n\n".join(sections)

    print(f"\n  --- 写开场白 ---")
    intro = call_llm([{"role": "user", "content": f"为播客'{topic}'写150字开场白。\n片段概要:\n{all_sections[:300]}...\n用中文。"}])
    print(f"  开场白: {intro[:100]}...")

    print(f"\n  --- 整合正文 ---")
    content = call_llm([{"role": "user", "content": f"整合以下播客片段为连贯正文:\n{all_sections}\n用中文。"}])
    print(f"  正文: {content[:100]}...")

    print(f"\n  --- 写结语 ---")
    conclusion = call_llm([{"role": "user", "content": f"为播客'{topic}'写100字结语，总结要点。用中文。"}])
    print(f"  结语: {conclusion[:100]}...")

    # 合并
    final = f"# 播客: {topic}\n\n{intro}\n\n{content}\n\n{conclusion}"

    # 保存
    filename = os.path.join(os.path.dirname(__file__), f"播客_{topic.replace(' ', '_')}_debug.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final)
    print(f"\n  已保存: {filename}")

    print(f"\n{'=' * 70}")
    print(f"【最终播客脚本】")
    print(f"{'=' * 70}")
    print(final)

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结：本课是目前架构最复杂的一课")
    print(f"{'=' * 70}")
    print(f"  1. 三层图嵌套:")
    print(f"     - 主图: 规划→并行访谈→汇总→输出")
    print(f"     - 规划子图: 关键词→子话题结构")
    print(f"     - 访谈子图: 提问→搜索→回答→循环")
    print(f"  2. Send API: 对每个子话题并行启动独立的访谈子图实例")
    print(f"     本质: list comprehension + 并发执行")
    print(f"  3. 多轮循环: 访谈子图中主持人-专家多轮对话，直到满意为止")
    print(f"     本质: while 循环 + 终止条件判断")
    print(f"  4. 三路并行: 开场/正文/结语并行生成后合并")
    print(f"     本质: 三个独立的 LLM 调用 + 字符串拼接")
    print(f"  5. 整体本质: 嵌套的 for 循环 + while 循环 + LLM 调用")
    print(f"     LangGraph 将这些组织为可视化的图，方便理解和调试")


if __name__ == "__main__":
    main()
