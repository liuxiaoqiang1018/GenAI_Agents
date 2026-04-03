"""
第37课：播客自动生成代理（简化版）

架构：规划子话题 → 逐个模拟访谈 → 整理脚本 → 生成完整播客
简化：不用子图嵌套和 Send API，用顺序执行模拟并行逻辑

核心模式：规划 → 多轮访谈循环 → Map-Reduce 汇总 → 播客脚本
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

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3
MAX_INTERVIEW_TURNS = 2  # 每个子话题的访谈轮数（减少调用次数，防止API过载）


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES + 2):  # 多给2次机会应对空响应
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            # 处理空响应（公益站偶尔返回200但body为空）
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
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                wait = (attempt + 1) * 5  # 5s, 10s, 15s, 20s 递增等待
                print(f"    等待{wait}秒后重试...")
                time.sleep(wait)
            else:
                raise


# ===== 阶段1：规划（1次LLM调用）=====
def plan_subtopics(topic: str) -> dict:
    """生成关键词和子话题（合并为1次调用）"""
    print("\n" + "=" * 60)
    print("【阶段1】规划 — 生成关键词和子话题")
    print("=" * 60)

    result = call_llm([
        {"role": "user", "content": (
            f"请为以下播客主题做规划:\n主题: {topic}\n\n"
            f"输出格式（严格遵守）:\n"
            f"关键词: 词1,词2,词3\n"
            f"子话题1: xxx\n"
            f"子话题2: xxx\n"
            f"只输出以上内容，不要其他。"
        )}
    ], temperature=0.3)

    # 解析
    keywords = []
    subtopics = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if "关键词" in line and ":" in line:
            keywords = [k.strip() for k in line.split(":", 1)[1].split(",")]
        elif "子话题" in line and ":" in line:
            subtopics.append(line.split(":", 1)[1].strip())

    if not subtopics:
        subtopics = ["AI在该领域的应用现状", "未来发展趋势与挑战"]

    print(f"  关键词: {keywords}")
    print(f"  子话题: {subtopics[:2]}")

    return {"keywords": keywords, "subtopics": subtopics[:2]}


# ===== 阶段2：模拟访谈（一次LLM调用直接生成脚本片段）=====
def conduct_interview(topic: str, subtopic: str) -> str:
    """对一个子话题直接生成播客脚本片段（1次LLM调用）"""
    print(f"\n{'=' * 60}")
    print(f"【访谈】子话题: {subtopic}")
    print(f"{'=' * 60}")

    prompt = (
        f"你是一位播客脚本编剧。请为以下子话题写一段播客对话脚本。\n\n"
        f"播客主题: {topic}\n"
        f"本段子话题: {subtopic}\n\n"
        f"要求:\n"
        f"1. 包含主持人和专家的2-3轮对话\n"
        f"2. 主持人提出有深度的问题，专家给出具体的例子和数据\n"
        f"3. 用'主持人:'和'专家:'标注发言\n"
        f"4. 控制在400字以内\n"
        f"5. 用中文，语气自然生动"
    )

    section_script = call_llm([{"role": "user", "content": prompt}], temperature=0.5)
    print(f"  脚本片段: {section_script[:200]}...")

    return section_script


# ===== 阶段3：汇总生成完整播客（1次LLM调用）=====
def generate_podcast(topic: str, sections: list) -> str:
    """将各片段汇总为完整播客脚本（1次LLM调用搞定）"""
    print(f"\n{'=' * 60}")
    print(f"【汇总】生成完整播客脚本")
    print(f"{'=' * 60}")

    all_sections = "\n\n---\n\n".join(sections)

    prompt = (
        f"你是播客制作人和脚本编辑。请根据以下访谈片段生成一份完整的播客脚本。\n\n"
        f"播客主题: {topic}\n\n"
        f"访谈片段:\n{all_sections}\n\n"
        f"要求:\n"
        f"1. 开场白（100字，吸引听众）\n"
        f"2. 正文（整合各段访谈，加过渡语句使其连贯）\n"
        f"3. 结语（100字，总结要点）\n"
        f"用中文，Markdown格式。"
    )

    final_script = call_llm([{"role": "user", "content": prompt}])
    final_script = f"# 播客: {topic}\n\n{final_script}"
    return final_script


# ===== 主函数 =====
def main():
    topic = "人工智能如何改变教育的未来"

    print("=" * 60)
    print("  播客自动生成代理")
    print("=" * 60)
    print(f"  主题: {topic}")

    # 阶段1：规划
    plan = plan_subtopics(topic)

    # 阶段2：逐个子话题进行访谈（原教程用 Send API 并行）
    sections = []
    for subtopic in plan["subtopics"]:
        section = conduct_interview(topic, subtopic)
        sections.append(section)

    # 阶段3：汇总
    final_script = generate_podcast(topic, sections)

    # 保存
    filename = os.path.join(os.path.dirname(__file__), f"播客_{topic.replace(' ', '_')}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_script)
    print(f"\n  播客脚本已保存: {filename}")

    print(f"\n{'=' * 60}")
    print(f"【最终播客脚本】")
    print(f"{'=' * 60}")
    print(final_script)


if __name__ == "__main__":
    main()
