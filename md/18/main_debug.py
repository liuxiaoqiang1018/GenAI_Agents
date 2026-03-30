"""
第18课 - AI音乐作曲代理内部机制（不使用 LangGraph）

目的：让你看清逐步丰富模式的本质：
  1. 旋律 = call_llm(用户输入)
  2. 和声 = call_llm(旋律)              ← 1个维度
  3. 节奏 = call_llm(旋律, 和声)        ← 2个维度
  4. 作品 = call_llm(旋律, 和声, 节奏, 风格) ← 全部维度
  5. MIDI = 格式转换（不调LLM）

对比 main.py（LangGraph 框架版），理解：
  - 5个节点 → 5个函数顺序调用
  - 逐步丰富 → 后面的函数参数越来越多
  - 和第16课的区别 → 不是固定上下文传递，而是每步增加一个新维度
"""

import os
import sys
import time
import httpx
from dotenv import load_dotenv

# 复用 main.py 的 MIDI 生成函数
sys.path.insert(0, os.path.dirname(__file__) or '.')
try:
    from main import generate_midi_file, HAS_MUSIC21
except ImportError:
    HAS_MUSIC21 = False
    def generate_midi_file(user_input, output_path):
        return ""

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
                raise ValueError("API返回空响应")
            return choices[0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, ValueError, KeyError, TypeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ================================================================
#  完整流程
# ================================================================

def compose_music(user_input: str, style: str = "古典") -> dict:
    """
    AI作曲的完整流程。

    Java 类比：
        // 装饰器模式：每层增加一个维度
        Music result = new StyleAdapter(style,      // 第4层：适配风格
            new RhythmLayer(                        // 第3层：加节奏
                new HarmonyLayer(                   // 第2层：加和声
                    new MelodyGenerator(input)      // 第1层：生旋律
                )
            )
        ).compose();

        // 等价于：
        String melody = step1(input);                    // 1个变量
        String harmony = step2(melody);                  // 2个变量
        String rhythm = step3(melody, harmony);          // 3个变量
        String comp = step4(melody, harmony, rhythm, style); // 4个变量
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：旋律（只有用户输入）
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：旋律生成】prompt包含: [用户输入]')
    print('=' * 60)

    melody = call_llm(
        user_input,
        "你是作曲家。生成旋律，用音符名表示。用中文解释。"
    )
    total_llm_calls += 1
    print(f'>>> 旋律: {melody[:150]}...')

    # ==========================================
    # 第2步：和声（+旋律）
    # ==========================================
    print()
    print('=' * 60)
    print('【第2步：和声创作】prompt包含: [旋律] ← 增加1个维度')
    print('=' * 60)

    harmony = call_llm(
        f"旋律：{melody}",
        "你是和声专家。为旋律创作和声，用和弦名表示。用中文。"
    )
    total_llm_calls += 1
    print(f'>>> 和声: {harmony[:150]}...')

    # ==========================================
    # 第3步：节奏（+旋律+和声）
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：节奏分析】prompt包含: [旋律]+[和声] ← 增加到2个维度')
    print('=' * 60)

    rhythm = call_llm(
        f"旋律：{melody}\n和声：{harmony}",
        "你是节奏专家。建议节奏模式，说明拍号和力度。用中文。"
    )
    total_llm_calls += 1
    print(f'>>> 节奏: {rhythm[:150]}...')

    # ==========================================
    # 第4步：风格适配（+旋律+和声+节奏+风格）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第4步：{style}风格适配】prompt包含: [旋律]+[和声]+[节奏]+[风格] ← 全部维度')
    print('=' * 60)

    composition = call_llm(
        f"风格：{style}\n旋律：{melody}\n和声：{harmony}\n节奏：{rhythm}",
        f"你是{style}风格编曲大师。整合为完整作品，描述配器和演奏细节。用中文。"
    )
    total_llm_calls += 1
    print(f'>>> 作品: {composition[:200]}...')

    # ==========================================
    # 第5步：MIDI转换（真实生成）
    # ==========================================
    print()
    print('=' * 60)
    print('【第5步：MIDI转换】（不调LLM，用 music21 生成）')
    print('=' * 60)

    midi_path = ""
    if HAS_MUSIC21:
        output_path = os.path.join(os.path.dirname(__file__) or '.', "output_debug.mid")
        midi_path = generate_midi_file(user_input, output_path)
        if midi_path:
            print(f'>>> MIDI 文件已生成: {os.path.abspath(midi_path)}')
            print(f'>>> 双击 .mid 文件即可播放')
    else:
        print(f'>>> music21 未安装，跳过。安装: pip install music21')

    print()
    print(f'>>> 逐步丰富总结:')
    print(f'    第1步: call_llm(用户输入)                          → 旋律')
    print(f'    第2步: call_llm(旋律)                              → 和声')
    print(f'    第3步: call_llm(旋律, 和声)                        → 节奏')
    print(f'    第4步: call_llm(旋律, 和声, 节奏, 风格)            → 完整作品')
    print(f'    参数从1个增长到4个，像滚雪球一样越来越丰富')
    print(f'    总LLM调用: {total_llm_calls}次')

    return {
        "melody": melody,
        "harmony": harmony,
        "rhythm": rhythm,
        "composition": composition,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第18课 - AI音乐作曲代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 逐步丰富 = 每步的prompt参数越来越多')
    print('  - 5个节点 = 5个函数，后面的能读到前面所有输出')
    print('  - 和第16课区别：第16课传固定上下文，本课每步增加新维度')
    print('  - 4次LLM调用 + 1次格式转换')
    print()

    examples = [
        ("创作一首欢快的C大调钢琴曲", "古典"),
        ("一首忧伤的小调小提琴独奏", "浪漫主义"),
    ]

    print('示例:')
    for i, (d, s) in enumerate(examples, 1):
        print(f'  {i}. {d} ({s})')
    print()

    desc = input('音乐描述（回车用示例1）: ').strip()
    if not desc:
        desc, style = examples[0]
        print(f'>>> 使用: {desc} ({style})')
    else:
        style = input('风格（回车默认"古典"）: ').strip() or "古典"

    result = compose_music(desc, style)

    print()
    print('#' * 60)
    print(f'#  作曲完成！LLM调用 {result["total_llm_calls"]} 次')
    print('#' * 60)
