"""
第17课 - TTS诗歌朗读代理内部机制（不使用 LangGraph）

目的：让你看清菱形结构（分支+汇聚）的本质：
  1. 分类 = 一次LLM调用
  2. 风格改写 = if-elif 选择不同的 system prompt
  3. TTS = 所有分支都调同一个函数（汇聚点）
  4. 整个系统 = 1次分类 + 1次改写 + 1次TTS = 最多3次调用

对比 main.py（LangGraph 框架版），理解：
  - 4条 add_edge(process_xxx, "tts") → 就是所有 if 分支都调 tts()
  - 菱形结构 → 就是 if-elif + 统一的后置处理
"""

import os
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

def process_text(text: str) -> dict:
    """
    TTS诗歌朗读代理的完整流程。

    Java 类比：
        @Service
        public class TtsService {
            // 策略模式 + 统一后置处理
            Map<String, TextProcessor> processors = Map.of(
                "诗歌", new PoetryProcessor(),
                "新闻", new NewsProcessor(),
                "笑话", new JokeProcessor()
            );

            public AudioResult process(String text) {
                // 1. 分类
                String type = classifier.classify(text);
                // 2. 分支处理（策略模式选一个）
                String processed = processors
                    .getOrDefault(type, new GeneralProcessor())
                    .rewrite(text);
                // 3. 汇聚：不管哪个分支，都走TTS
                return ttsService.synthesize(processed, voiceMap.get(type));
            }
        }
    """

    total_llm_calls = 0

    # ==========================================
    # 第1步：内容分类
    # ==========================================
    print()
    print('=' * 60)
    print('【第1步：内容分类】（LLM调用 #1）')
    print('=' * 60)

    system = ("分类文本为：一般、诗歌、新闻、笑话。只回复类型名。")
    content_type = call_llm(text, system)
    total_llm_calls += 1

    for t in ["诗歌", "新闻", "笑话", "一般"]:
        if t in content_type:
            content_type = t
            break
    else:
        content_type = "一般"

    print(f'>>> 分类结果: {content_type}')

    # ==========================================
    # 第2步：风格改写（菱形分支部分）
    # ==========================================
    print()
    print('=' * 60)
    print(f'【第2步：风格改写 — {content_type}】')
    print('=' * 60)

    # 这就是菱形结构的分支部分（if-elif）
    # 每个分支只是 system prompt 不同
    style_prompts = {
        "诗歌": "把以下文字改写为一首优美的中文诗歌，注意韵律和意境。",
        "新闻": "把以下文字改写为正式的中文新闻播报稿，语气严肃专业。",
        "笑话": "把以下文字改写为一个幽默搞笑的中文笑话或段子。",
    }
    voice_types = {
        "诗歌": "柔和（诗歌朗诵）",
        "新闻": "沉稳（新闻播音）",
        "笑话": "活泼（相声口吻）",
        "一般": "标准",
    }

    print(f'    分支逻辑（就是 if-elif）:')
    print(f'    if content_type == "{content_type}":')

    if content_type in style_prompts:
        print(f'        → 用 "{content_type}" 风格改写')
        processed_text = call_llm(text, style_prompts[content_type])
        total_llm_calls += 1
        print(f'>>> 改写结果:\n{processed_text}')
    else:
        print(f'        → 一般文本，不改写')
        processed_text = text

    voice_type = voice_types.get(content_type, "标准")

    # ==========================================
    # 第3步：语音合成（菱形汇聚点）
    # ==========================================
    print()
    print('=' * 60)
    print('【第3步：语音合成 — 汇聚点】')
    print('=' * 60)
    print(f'>>> 不管前面走的哪个分支，都到这里')
    print(f'>>> 语音风格: {voice_type}')
    print(f'>>> 待朗读: {processed_text[:80]}...')
    print()
    print(f'>>> 生产环境:')
    print(f'>>>   voice_map = {{"一般":"alloy", "诗歌":"nova", "新闻":"onyx", "笑话":"shimmer"}}')
    print(f'>>>   audio = openai.audio.speech.create(voice=voice_map[type], input=text)')
    print()
    print(f'>>> LLM调用: {total_llm_calls}次（分类1次 + 改写0或1次）')

    return {
        "content_type": content_type,
        "processed_text": processed_text,
        "voice_type": voice_type,
        "total_llm_calls": total_llm_calls,
    }


# ================================================================
#  运行
# ================================================================

if __name__ == '__main__':
    print('第17课 - TTS诗歌朗读代理（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {API_BASE_URL}')
    print()
    print('核心发现：')
    print('  - 菱形结构 = if-elif分支 + 统一后置处理')
    print('  - 风格改写 = 同一文本，不同system prompt')
    print('  - TTS汇聚 = 不管走哪个分支，最后都调tts()')
    print('  - 整个系统 = 最多2次LLM调用（分类+改写）')
    print()

    examples = [
        ("春天来了，花儿开了，鸟儿在枝头歌唱", "诗歌"),
        ("今日A股三大指数集体高开，沪指涨0.5%", "新闻"),
        ("程序员最怕什么？产品经理说：需求变了", "笑话"),
        ("今天天气不错，我打算出去走走", "一般"),
    ]

    print('--- 示例测试 ---')
    for text, expected in examples:
        print()
        print('#' * 60)
        print(f'#  输入: {text}')
        print(f'#  预期: {expected}')
        print('#' * 60)
        process_text(text)

    # 交互模式
    print('\n输入文本，输入 /quit 退出\n')
    while True:
        user_input = input('输入文本: ').strip()
        if user_input == '/quit':
            print('再见！')
            break
        if not user_input:
            continue
        process_text(user_input)
