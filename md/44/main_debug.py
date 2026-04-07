"""
第44课：学术系统综述自动生成（透明调试版）

展示20节点图中最核心的流程：规划→分析→写作→评审循环。
重点展示并行写作和评审循环的Prompt构建过程。
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

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


def call_llm(messages: list, temperature: float = 0.3) -> str:
    url = f"{API_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    for attempt in range(MAX_RETRIES + 2):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            if not resp.content or not resp.text.strip():
                raise ValueError("API返回空响应")
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError("无choices")
            content = choices[0]["message"]["content"]
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if not content:
                raise ValueError("空内容")
            return content
        except Exception as e:
            print(f"    !! LLM失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES + 1:
                time.sleep((attempt + 1) * 5)
            else:
                raise


MOCK_ANALYSES = """
论文1: 扩散模型通过逐步去噪生成音乐，代表模型DiffWave/MusicLDM，质量优于GAN。
论文2: 文本到音乐扩散模型用CLAP编码器映射文本到音频空间，长时一致性是挑战。
论文3: 扩散模型可用于音乐修复和风格迁移，数据集限制是瓶颈。
"""


def main():
    topic = "扩散模型在音乐生成中的应用"

    print("=" * 70)
    print("  学术系统综述自动生成（透明调试版 - 无框架）")
    print("=" * 70)
    print(f"  主题: {topic}")
    print(f"  原教程: 20个节点的 LangGraph 图，是所有课中最大的")

    # ==============================================================
    # 步骤1：规划
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤1】规划综述大纲")
    print(f"{'=' * 70}")

    plan_prompt = (
        "你是学术研究者，请为以下主题创建系统综述大纲。\n"
        "包含: 摘要、引言、方法、结果、结论、参考文献。\n"
        f"主题: {topic}\n每章2-3个要点。中文。"
    )
    print(f"  >>> Prompt: {plan_prompt[:100]}...")
    outline = call_llm([{"role": "user", "content": plan_prompt}])
    print(f"  >>> 大纲:\n{outline[:300]}...")

    # ==============================================================
    # 步骤2-3：搜索+下载+分析（模拟）
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤2-3】搜索+下载+分析论文（模拟）")
    print(f"{'=' * 70}")
    print(f"  原教程流程:")
    print(f"    researcher → LLM生成5个搜索查询")
    print(f"    search_articles → Semantic Scholar API")
    print(f"    article_decisions → LLM筛选哪些论文有用")
    print(f"    download_articles → 下载PDF")
    print(f"    paper_analyzer → pymupdf4llm转文本 + LLM提取关键信息")
    print(f"  本课简化: 使用模拟分析数据")
    print(f"  分析数据: {MOCK_ANALYSES[:200]}...")

    # ==============================================================
    # 步骤4：并行写6个章节
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤4】并行写6个章节（原教程6个节点同时执行）")
    print(f"{'=' * 70}")
    print(f"  原教程: paper_analyzer 同时连接6个write节点:")
    print(f"    paper_analyzer → write_abstract")
    print(f"    paper_analyzer → write_introduction")
    print(f"    paper_analyzer → write_methods")
    print(f"    paper_analyzer → write_results")
    print(f"    paper_analyzer → write_conclusion")
    print(f"    paper_analyzer → write_references")
    print(f"  6个节点都完成后 → aggregate_paper 汇聚")

    section_names = ["摘要", "引言", "方法"]  # 简化只写3个
    sections = {}

    def write_one_section(name):
        prompt = (
            f"撰写系统综述的【{name}】部分。\n"
            f"大纲:\n{outline[:300]}\n论文分析:\n{MOCK_ANALYSES}\n中文，学术风格，200字以内。"
        )
        print(f"  >>> 【{name}】Prompt: {prompt[:100]}...")
        result = call_llm([{"role": "user", "content": prompt}])
        print(f"  >>> 【{name}】完成: {result[:150]}...")
        return name, result

    print(f"  分批并行调用LLM（每批2个，避免API限流）")
    batch_size = 2
    for i in range(0, len(section_names), batch_size):
        batch = section_names[i:i + batch_size]
        print(f"\n  --- 第{i // batch_size + 1}批: {batch} ---")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(write_one_section, name) for name in batch]
            for future in as_completed(futures):
                try:
                    name, content = future.result()
                    sections[name] = content
                except Exception as e:
                    print(f"  !! 章节写作失败: {e}")
        if i + batch_size < len(section_names):
            time.sleep(3)

    # ==============================================================
    # 步骤5：汇聚
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤5】汇聚为完整论文")
    print(f"{'=' * 70}")

    draft = ""
    for name, content in sections.items():
        draft += f"## {name}\n\n{content}\n\n"
    print(f"  论文总长: {len(draft)} 字")

    # ==============================================================
    # 步骤6：评审-修改循环
    # ==============================================================
    print(f"\n{'=' * 70}")
    print(f"【步骤6】评审-修改循环（最多2轮）")
    print(f"{'=' * 70}")
    print(f"  原教程: critique_paper → exists_action 条件路由:")
    print(f"    - 'ACCEPT' 或达到上限 → final_draft")
    print(f"    - 需修改 → revise_paper → 回到 critique_paper")

    critique_prompt = (
        f"评审以下论文草稿，指出改进点。质量足够好则回复ACCEPT。中文。\n\n{draft[:1000]}"
    )
    print(f"\n  >>> 评审中...")
    critique = call_llm([{"role": "user", "content": critique_prompt}])
    print(f"  >>> 评审意见: {critique[:200]}...")

    if "ACCEPT" not in critique.upper():
        print(f"\n  >>> 修改中...")
        revised = call_llm([{"role": "user", "content": f"根据意见修改:\n{critique}\n\n草稿:\n{draft[:1000]}"}])
        draft = revised
        print(f"  >>> 修改完成: {draft[:200]}...")

    # 保存
    filepath = os.path.join(os.path.dirname(__file__), "系统综述_debug.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {topic} - 系统综述\n\n{draft}")
    print(f"\n  已保存: {filepath}")

    # 调试总结
    print(f"\n{'=' * 70}")
    print(f"  调试总结")
    print(f"{'=' * 70}")
    print(f"  1. 这是所有课中节点最多(20个)的图")
    print(f"  2. 核心架构:")
    print(f"     规划 → 搜索 → 决策 → 下载 → 分析")
    print(f"     → [6路并行写作] → 汇聚 → 评审 → 修改循环 → 定稿")
    print(f"  3. 6路并行: paper_analyzer 同时出6条边到6个write节点")
    print(f"     所有write节点完成后汇聚到 aggregate_paper")
    print(f"     LangGraph 自动等待所有并行节点完成再执行汇聚")
    print(f"  4. 评审循环: critique → 条件路由(accept/revise) → 最多2轮")
    print(f"  5. 真实学术工具链: Semantic Scholar API + PDF下载 + pymupdf4llm")
    print(f"  6. 本质: 多次LLM调用 + API调用 + 文件IO + 循环控制")


if __name__ == "__main__":
    main()
