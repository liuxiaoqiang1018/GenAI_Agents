"""
第44课：学术系统综述自动生成（简化版）

架构：规划 → 模拟论文数据 → 分析 → 并行写6章节 → 汇聚 → 评审修改循环 → 定稿
核心模式：6路并行写作 + 汇聚 + 评审-修改循环（最复杂的流水线）
"""

import os
import re
import sys
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3
MAX_REVISIONS = 2


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.3) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"无choices: {str(data)[:200]}")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    [LLM错误] 第{attempt+1}次: {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


# ===== 模拟论文分析数据 =====
MOCK_ANALYSES = """
论文1: "扩散模型在音乐生成中的应用综述"
- 扩散模型通过逐步去噪过程生成音乐，相比GAN更稳定
- 代表模型：DiffWave、Noise2Music、MusicLDM
- 关键发现：在音频质量和多样性方面优于传统方法

论文2: "基于文本到音乐的扩散模型"
- 用文本描述控制音乐生成（如"欢快的爵士乐"）
- 关键技术：CLAP编码器将文本映射到音频潜空间
- 挑战：长时间一致性和音乐结构控制仍是难题

论文3: "扩散模型与音乐信息检索"
- 扩散模型可用于音乐修复、风格迁移和伴奏生成
- 与传统MIR方法结合可提升检索准确性
- 数据集限制是当前研究的主要瓶颈
"""


# ===== 步骤函数 =====

def plan_outline(topic: str) -> str:
    """步骤1：规划综述大纲"""
    print("\n" + "=" * 60)
    print("【步骤1】规划综述大纲")
    print("=" * 60)

    prompt = (
        "你是一位学术研究者，正在规划一篇系统综述论文。\n"
        "系统综述通常包含：标题、摘要、引言、方法、结果、结论、参考文献。\n\n"
        f"研究主题: {topic}\n\n"
        "请为这个主题创建一个系统综述大纲，每个章节列出2-3个要点。用中文。"
    )

    outline = call_llm([{"role": "user", "content": prompt}])
    print(f"  大纲:\n{outline[:400]}...")
    return outline


def analyze_papers(outline: str) -> str:
    """步骤2：分析论文（使用模拟数据）"""
    print("\n" + "=" * 60)
    print("【步骤2】论文分析（模拟 Semantic Scholar + PDF 解析）")
    print("=" * 60)
    print(f"  原教程: Semantic Scholar API搜索 → 下载PDF → pymupdf4llm解析")
    print(f"  本课: 使用模拟论文分析数据")
    print(f"  论文分析:\n{MOCK_ANALYSES[:300]}...")
    return MOCK_ANALYSES


def write_section(section_name: str, outline: str, analyses: str) -> str:
    """写一个章节"""
    prompts = {
        "摘要": "写一个200字的结构化摘要，包含背景、方法、结果和结论。",
        "引言": "写一个引言部分，介绍研究背景、动机和本综述的目的。",
        "方法": "写方法部分，描述文献检索策略、筛选标准和分析方法。",
        "结果": "写结果部分，总结分析发现的主要结果和趋势。",
        "结论": "写结论部分，总结关键发现、局限性和未来研究方向。",
        "参考文献": "列出3-5个虚构但格式正确的参考文献条目。"
    }

    prompt = (
        f"你是学术论文写作专家。请撰写系统综述的【{section_name}】部分。\n\n"
        f"大纲:\n{outline[:500]}\n\n"
        f"论文分析:\n{analyses[:500]}\n\n"
        f"要求: {prompts.get(section_name, '撰写该部分。')}\n"
        f"用中文，学术风格。"
    )

    return call_llm([{"role": "user", "content": prompt}])


def write_all_sections(outline: str, analyses: str) -> dict:
    """步骤3：真正并行写6个章节"""
    print("\n" + "=" * 60)
    print("【步骤3】并行撰写6个章节")
    print("=" * 60)
    print("  使用线程池并行调用LLM，大幅缩短等待时间")

    section_names = ["摘要", "引言", "方法", "结果", "结论", "参考文献"]
    sections = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(write_section, name, outline, analyses): name
            for name in section_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                sections[name] = result
                print(f"  完成【{name}】（{len(result)}字）: {result[:100]}...")
            except Exception as e:
                print(f"  【{name}】写作失败: {e}")
                sections[name] = f"（{name}部分生成失败）"

    return sections


def aggregate(sections: dict) -> str:
    """步骤4：汇聚为完整论文"""
    print("\n" + "=" * 60)
    print("【步骤4】汇聚为完整论文")
    print("=" * 60)

    # 生成标题
    title = call_llm([
        {"role": "user", "content": f"根据以下摘要生成一个学术论文标题:\n{sections['摘要'][:200]}\n只输出标题。"}
    ])

    draft = f"# {title}\n\n"
    for name in ["摘要", "引言", "方法", "结果", "结论", "参考文献"]:
        draft += f"## {name}\n\n{sections[name]}\n\n"

    print(f"  论文总长: {len(draft)} 字")
    return draft


def critique_and_revise(draft: str, outline: str) -> str:
    """步骤5：评审-修改循环"""
    current_draft = draft

    for revision in range(MAX_REVISIONS):
        print(f"\n{'=' * 60}")
        print(f"【步骤5.{revision+1}】评审第 {revision+1} 轮")
        print("=" * 60)

        # 评审
        critique = call_llm([
            {"role": "user", "content": (
                f"你是学术论文评审专家。请评审以下系统综述论文草稿。\n\n"
                f"大纲:\n{outline[:300]}\n\n"
                f"论文草稿:\n{current_draft[:2000]}\n\n"
                f"请指出需要改进的地方。如果质量已经足够好，回复'ACCEPT'。用中文。"
            )}
        ])

        print(f"  评审意见: {critique[:200]}...")

        if "ACCEPT" in critique.upper():
            print(f"  评审通过！")
            break

        # 修改
        print(f"\n  修改中...")
        revised = call_llm([
            {"role": "user", "content": (
                f"根据评审意见修改论文。\n\n"
                f"评审意见:\n{critique}\n\n"
                f"当前草稿:\n{current_draft[:2000]}\n\n"
                f"请输出修改后的完整论文。用中文。"
            )}
        ])

        current_draft = revised
        print(f"  修改完成（{len(current_draft)}字）")

    return current_draft


# ===== 主函数 =====
def main():
    topic = "扩散模型在音乐生成中的应用"

    print("=" * 60)
    print("  学术系统综述自动生成")
    print("=" * 60)
    print(f"  主题: {topic}")

    # 步骤1: 规划
    outline = plan_outline(topic)

    # 步骤2: 论文分析
    analyses = analyze_papers(outline)

    # 步骤3: 并行写6章节
    sections = write_all_sections(outline, analyses)

    # 步骤4: 汇聚
    draft = aggregate(sections)

    # 步骤5: 评审-修改循环
    final = critique_and_revise(draft, outline)

    # 保存
    filepath = os.path.join(os.path.dirname(__file__), "系统综述论文.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final)
    print(f"\n  论文已保存: {filepath}")

    print(f"\n{'=' * 60}")
    print(f"【最终论文（前1000字）】")
    print(f"{'=' * 60}")
    print(final[:1000])


if __name__ == "__main__":
    main()
