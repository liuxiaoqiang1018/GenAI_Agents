"""
第18课 - AI音乐作曲代理（LangGraph 框架版）

核心概念：
  - 逐步丰富：每步在前一步基础上增加一个维度（旋律→和声→节奏→风格）
  - 线性链式流水线：5个节点顺序执行
  - 多模态输出：文本描述→音乐符号→MIDI真实文件
  - music21 生成 MIDI，Windows 媒体播放器可直接播放
"""

import os
import time
import random
from typing import TypedDict

import httpx
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False
    print("⚠ 未安装 music21，MIDI生成将跳过。安装命令: pip install music21")

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MAX_RETRIES = 3


def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（带重试和空响应防护）"""
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


# ========== State ==========

class MusicState(TypedDict):
    user_input: str      # 用户描述
    style: str           # 音乐风格
    melody: str          # 旋律
    harmony: str         # 和声
    rhythm: str          # 节奏
    composition: str     # 完整作品
    midi_info: str       # MIDI信息（模拟）


# ========== 节点1：旋律生成 ==========

def melody_node(state: MusicState):
    print()
    print('=' * 60)
    print('【1 - 旋律生成】')
    print('=' * 60)
    print(f'>>> 输入: {state["user_input"]}')

    system = ("你是作曲家。根据用户描述生成旋律。\n"
              "用简谱或音符名表示（如 C4 D4 E4 F4 G4），说明调式和情感。用中文解释。")
    melody = call_llm(state["user_input"], system)

    print(f'>>> 旋律:\n{melody[:200]}')
    return {"melody": melody}


# ========== 节点2：和声创作 ==========

def harmony_node(state: MusicState):
    print()
    print('=' * 60)
    print('【2 - 和声创作】（基于旋律）')
    print('=' * 60)

    system = ("你是和声编配专家。为以下旋律创作和声。\n"
              "用和弦名表示（如 Cmaj Fmaj G7），说明和声进行逻辑。用中文。")
    prompt = f"旋律：{state['melody']}"
    harmony = call_llm(prompt, system)

    print(f'>>> 和声:\n{harmony[:200]}')
    return {"harmony": harmony}


# ========== 节点3：节奏分析 ==========

def rhythm_node(state: MusicState):
    print()
    print('=' * 60)
    print('【3 - 节奏分析】（基于旋律+和声）')
    print('=' * 60)

    system = ("你是节奏编排专家。根据旋律和和声建议节奏模式。\n"
              "说明拍号、节奏型和力度变化。用中文。")
    prompt = f"旋律：{state['melody']}\n和声：{state['harmony']}"
    rhythm = call_llm(prompt, system)

    print(f'>>> 节奏:\n{rhythm[:200]}')
    return {"rhythm": rhythm}


# ========== 节点4：风格适配 ==========

def style_node(state: MusicState):
    print()
    print('=' * 60)
    print('【4 - 风格适配】（基于旋律+和声+节奏+风格）')
    print('=' * 60)

    style = state.get("style", "古典")
    system = (f"你是{style}风格的编曲大师。\n"
              f"将以下音乐元素整合为一首完整的{style}风格作品。\n"
              f"描述配器、演奏技法、速度、力度等细节。用中文。")
    prompt = (f"风格：{style}\n"
              f"旋律：{state['melody']}\n"
              f"和声：{state['harmony']}\n"
              f"节奏：{state['rhythm']}")
    composition = call_llm(prompt, system)

    print(f'>>> 完整作品:\n{composition[:300]}')
    return {"composition": composition}


# ========== 音乐生成工具函数 ==========

# 音阶定义
SCALES = {
    'C大调': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'C小调': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
    'D大调': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'G大调': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'A小调': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
}

# 和弦定义
CHORDS = {
    'C': ['C4', 'E4', 'G4'],
    'Dm': ['D4', 'F4', 'A4'],
    'Em': ['E4', 'G4', 'B4'],
    'F': ['F4', 'A4', 'C5'],
    'G': ['G4', 'B4', 'D5'],
    'Am': ['A4', 'C5', 'E5'],
    'Cm': ['C4', 'Eb4', 'G4'],
    'Fm': ['F4', 'Ab4', 'C5'],
    'Gm': ['G4', 'Bb4', 'D5'],
}

# 常见和弦进行
PROGRESSIONS = {
    '大调': ['C', 'Am', 'F', 'G', 'C', 'F', 'G', 'C'],
    '小调': ['Am', 'Dm', 'Em', 'Am', 'Dm', 'Em', 'Am', 'Am'],
    '欢快': ['C', 'G', 'Am', 'F', 'C', 'G', 'F', 'C'],
    '忧伤': ['Am', 'F', 'C', 'G', 'Am', 'Dm', 'Em', 'Am'],
}


def detect_key_and_mood(user_input: str) -> tuple:
    """从用户输入猜测调式和情绪"""
    text = user_input.lower()
    if '小调' in text or '忧伤' in text or '悲' in text:
        scale_name = 'A小调'
        mood = '忧伤'
    elif 'c大调' in text or '欢快' in text or '快乐' in text:
        scale_name = 'C大调'
        mood = '欢快'
    elif 'd大调' in text:
        scale_name = 'D大调'
        mood = '大调'
    elif 'g大调' in text:
        scale_name = 'G大调'
        mood = '大调'
    else:
        scale_name = 'C大调'
        mood = '欢快'
    return scale_name, mood


def generate_midi_file(user_input: str, output_path: str = "output.mid") -> str:
    """用 music21 生成真实的 MIDI 文件"""
    if not HAS_MUSIC21:
        print(">>> music21 未安装，跳过MIDI生成")
        return ""

    scale_name, mood = detect_key_and_mood(user_input)
    scale_notes = SCALES[scale_name]
    progression = PROGRESSIONS.get(mood, PROGRESSIONS['欢快'])

    piece = music21.stream.Score()
    piece.insert(0, music21.tempo.MetronomeMark(number=100))

    # 旋律声部
    melody_part = music21.stream.Part()
    melody_part.insert(0, music21.instrument.Piano())
    durations = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5]

    for bar in range(len(progression)):
        bar_notes = []
        remaining = 2.0  # 每小节2拍（简化）
        while remaining > 0:
            dur = random.choice([0.25, 0.5, 0.5, 1.0])
            dur = min(dur, remaining)
            note_name = random.choice(scale_notes)
            octave = random.choice([4, 5]) if mood == '欢快' else random.choice([3, 4])
            n = music21.note.Note(f"{note_name}{octave}")
            n.quarterLength = dur
            n.volume.velocity = random.randint(60, 100)
            melody_part.append(n)
            remaining -= dur

    # 和声声部
    harmony_part = music21.stream.Part()
    harmony_part.insert(0, music21.instrument.Piano())

    for chord_name in progression:
        chord_notes = CHORDS.get(chord_name, CHORDS['C'])
        c = music21.chord.Chord(chord_notes)
        c.quarterLength = 2.0  # 每个和弦持续2拍
        c.volume.velocity = 50  # 和声稍轻
        harmony_part.append(c)

    piece.append(melody_part)
    piece.append(harmony_part)

    # 写入 MIDI 文件
    piece.write('midi', fp=output_path)
    return output_path


# ========== 节点5：MIDI转换（真实生成） ==========

def midi_node(state: MusicState):
    print()
    print('=' * 60)
    print('【5 - MIDI转换】')
    print('=' * 60)

    if not HAS_MUSIC21:
        print('>>> music21 未安装，跳过。安装: pip install music21')
        return {"midi_info": "未生成（需安装music21）"}

    # 生成 MIDI 文件
    output_path = os.path.join(os.path.dirname(__file__) or '.', "output.mid")
    midi_path = generate_midi_file(state["user_input"], output_path)

    if midi_path:
        abs_path = os.path.abspath(midi_path)
        print(f'>>> MIDI 文件已生成: {abs_path}')
        print(f'>>> 双击 .mid 文件即可用 Windows 媒体播放器播放')
        midi_info = f"MIDI文件: {abs_path}"
    else:
        midi_info = "MIDI生成失败"
        print(f'>>> {midi_info}')

    return {"midi_info": midi_info}


# ========== 构建图 ==========

def build_workflow():
    workflow = StateGraph(MusicState)

    workflow.add_node("melody", melody_node)
    workflow.add_node("harmony", harmony_node)
    workflow.add_node("rhythm", rhythm_node)
    workflow.add_node("style", style_node)
    workflow.add_node("midi", midi_node)

    workflow.set_entry_point("melody")
    workflow.add_edge("melody", "harmony")
    workflow.add_edge("harmony", "rhythm")
    workflow.add_edge("rhythm", "style")
    workflow.add_edge("style", "midi")
    workflow.add_edge("midi", END)

    return workflow.compile()


def print_graph(app):
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f'图可视化失败: {e}')
    print()


# ========== 运行 ==========

if __name__ == '__main__':
    print('第18课 - AI音乐作曲代理')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()

    app = build_workflow()
    print_graph(app)

    examples = [
        ("创作一首欢快的C大调钢琴曲", "古典"),
        ("一首忧伤的小调小提琴独奏", "浪漫主义"),
        ("节奏感强的电子舞曲", "电子"),
    ]

    print('示例:')
    for i, (desc, style) in enumerate(examples, 1):
        print(f'  {i}. {desc} ({style}风格)')
    print()

    desc = input('音乐描述（回车用示例1）: ').strip()
    if not desc:
        desc, style = examples[0]
        print(f'>>> 使用: {desc} ({style})')
    else:
        style = input('风格（回车默认"古典"）: ').strip() or "古典"

    print()
    print('#' * 60)
    print('#  开始作曲')
    print('#' * 60)

    result = app.invoke({
        "user_input": desc,
        "style": style,
        "melody": "",
        "harmony": "",
        "rhythm": "",
        "composition": "",
        "midi_info": "",
    })

    print()
    print('#' * 60)
    print('#  作曲完成')
    print('#' * 60)
