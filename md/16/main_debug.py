"""
第16课 - GIF动画生成器内部机制（不使用 LangGraph）

目的：让你看清提示词链式传递的本质：
  1. 角色描述 = 一次LLM调用，只有用户输入
  2. 剧情生成 = 一次LLM调用，prompt包含 用户输入+角色描述
  3. 提示词生成 = 一次LLM调用，prompt包含 角色描述+剧情
  4. 图像生成 = N次API调用（每帧一次，本课模拟）
  5. 合成GIF = PIL拼帧（本课模拟）

对比 main.py（LangGraph 框架版），理解：
  - 5个节点 → 就是5个函数顺序调用
  - State累积 → 就是变量越传越多
  - 提示词链 → 就是后面的函数能读到前面函数的返回值
  - 一致性 → 就是把角色描述复制粘贴到每个prompt里
"""

import os
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
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not choices[0].get("message"):
                raise ValueError(f"API返回空响应: {data}")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                print(f'    ⚠ API异常，{wait}秒后重试({attempt+1}/{MAX_RETRIES}): {e}')
                time.sleep(wait)
            else:
                raise


# ================================================================
#  完整流程
# ================================================================

def generate_gif(query: str) -> dict:
    """
    GIF动画生成的完整流程。

    Java 类比：
        @Service
        public class GifGeneratorService {
            public GifResult generate(String query) {
                // 提示词链：每步都把前面的输出传进去
                String charDesc = step1_describeCharacter(query);
                String plot = step2_generatePlot(query, charDesc);       // ← 包含charDesc
                List<String> prompts = step3_createPrompts(charDesc, plot); // ← 包含charDesc+plot
                List<Image> images = step4_generateImages(prompts);
                byte[] gif = step5_assembleGif(images);
                return new GifResult(gif);
            }
        }

    关键：charDesc 从第1步一直传到第3步 → 保证每帧角色一致
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：角色描述（只有用户输入）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：角色描述】（LLM调用 #1）')
    print('=' * 60)
    print(f'>>> prompt 包含: 用户输入')

    character_desc = call_llm(
        f"根据'{query}'，创建详细的角色描述（外观、特征、颜色）。中文，150字以内。"
    )
    total_llm_calls += 1
    print(f'>>> 角色描述: {character_desc}')

    # ==========================================
    # 第2步：剧情生成（用户输入 + 角色描述）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：剧情生成】（LLM调用 #2）')
    print('=' * 60)
    print(f'>>> prompt 包含: 用户输入 + 第1步的角色描述')
    print(f'>>> ↑ 这就是提示词链：后面的步骤能看到前面的输出')

    plot = call_llm(
        f"用户描述：{query}\n"
        f"角色特征：{character_desc}\n\n"
        f"创建5帧GIF动画剧情，每帧一句话。中文。\n"
        f"格式：1. xxx\n2. xxx\n..."
    )
    total_llm_calls += 1
    print(f'>>> 剧情:\n{plot}')

    # ==========================================
    # 第3步：提示词生成（角色描述 + 剧情）
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：提示词生成】（LLM调用 #3）')
    print('=' * 60)
    print(f'>>> prompt 包含: 第1步角色描述 + 第2步剧情')
    print(f'>>> 一致性关键：每个提示词都要包含角色特征')

    response = call_llm(
        f"角色特征：{character_desc}\n"
        f"剧情：{plot}\n\n"
        f"为每帧生成图像提示词。每个提示词必须包含角色关键特征（保一致性）。\n"
        f"格式：1. 提示词\n2. 提示词\n..."
    )
    total_llm_calls += 1

    # 解析
    prompts = []
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            text = line.split('.', 1)[1].strip()
            if text:
                prompts.append(text)
    if not prompts:
        prompts = [response]

    print(f'>>> {len(prompts)} 个提示词:')
    for i, p in enumerate(prompts, 1):
        print(f'    帧{i}: {p[:60]}...')

    # ==========================================
    # 第4步：图像生成（模拟）
    # ==========================================
    print()
    print('=' * 60)
    print('【第4步：图像生成】（模拟 — 每个提示词一次API调用）')
    print('=' * 60)

    print('>>> 生产环境中的代码:')
    print('>>>   for prompt in prompts:')
    print('>>>       image = dall_e.generate(prompt)  # 每帧一次调用')
    print()

    scenes = []
    for i, prompt in enumerate(prompts, 1):
        scenes.append(prompt)  # 模拟：直接用提示词（生产中调图像API）
        print(f'    帧{i}: {prompt[:80]}...')

    # ==========================================
    # 第5步：合成GIF（模拟）
    # ==========================================
    print()
    print('=' * 60)
    print('【第5步：合成GIF】（模拟 — 用 PIL 拼帧）')
    print('=' * 60)

    print('>>> 生产环境中的代码:')
    print('>>>   images = [Image.open(url) for url in image_urls]')
    print('>>>   images[0].save("output.gif", format="GIF",')
    print('>>>       save_all=True, append_images=images[1:],')
    print('>>>       duration=1000, loop=0)')
    print()

    # 展示结果
    print('>>> 最终动画帧序列:')
    for i, scene in enumerate(scenes, 1):
        print(f'    帧{i}: {scene}')

    print()
    print(f'>>> 提示词链总结:')
    print(f'    第1步 prompt: [用户输入]')
    print(f'    第2步 prompt: [用户输入] + [角色描述]')
    print(f'    第3步 prompt: [角色描述] + [剧情]')
    print(f'    第4步 prompt: [每个图像提示词]（每帧含角色特征→一致性）')
    print(f'    总LLM调用: {total_llm_calls}次')

    return {
        "character_description": character_desc,
        "plot": plot,
        "image_prompts": prompts,
        "scene_descriptions": scenes,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第16课 - GIF动画生成器（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 提示词链 = 每步把前面的输出塞进下一步的prompt')
    print('  - 一致性 = 角色描述复制粘贴到每个图像提示词里')
    print('  - 5个节点 = 5个函数顺序调用，变量越传越多')
    print('  - 多模态 = 从文本到图像的跳转（第3→4步）')
    print()

    examples = [
        "一只戴礼帽的猫坐在书桌前用鹅毛笔写信",
        "一个机器人在厨房里学做中国菜",
        "一只小狗在雪地里追蝴蝶",
    ]

    print('示例:')
    for i, ex in enumerate(examples, 1):
        print(f'  {i}. {ex}')
    print()

    query = input('GIF场景描述（回车用示例1）: ').strip()
    if not query:
        query = examples[0]
        print(f'>>> 使用: {query}')

    result = generate_gif(query)

    print()
    print('#' * 60)
    print(f'#  GIF生成完成！LLM调用 {result["total_llm_calls"]} 次')
    print(f'#  帧数: {len(result["scene_descriptions"])}')
    print(f'#  接入 DALL-E API 即可生成真实图像')
    print('#' * 60)
