# 第17课 - TTS诗歌朗读代理（内容分类 + 风格改写 + 语音合成）

## 环境准备

```bash
pip install langgraph langchain-core httpx python-dotenv
```

`main_debug.py` 只需要 `httpx` 和 `python-dotenv`。

注：原始 notebook 使用 OpenAI TTS API 生成真实语音。本课模拟语音合成步骤，聚焦于分类→风格改写的模式。

## 概述

本课构建一个**TTS诗歌朗读代理**，根据输入内容自动：

1. **内容分类** → 判断是 一般文本 / 诗歌 / 新闻 / 笑话
2. **风格改写** → 4选1路由，每种类型用不同风格改写
3. **语音合成** → 所有路径汇聚到TTS节点，不同类型用不同语音

这是一个典型的**分类→分支处理→汇聚**模式（菱形结构）。

## 架构

```
用户输入
   │
   ▼
┌──────────┐
│ 内容分类  │
└──────────┘
   │
   ├── 一般 → 【原样输出】──┐
   ├── 诗歌 → 【诗意改写】──┤
   ├── 新闻 → 【播音改写】──┤
   └── 笑话 → 【幽默改写】──┤
                             │
                             ▼
                      ┌──────────┐
                      │ 语音合成  │  ← 所有分支汇聚
                      └──────────┘
```

## 核心概念

### 1. 分支+汇聚（Diamond Pattern）

之前第9课的4个分支各自 → END。本课不同：4个分支**汇聚到同一个节点**（语音合成），形成菱形结构：

```python
# 4条分支
workflow.add_conditional_edges("classify", lambda x: x["content_type"], {
    "一般": "process_general",
    "诗歌": "process_poem",
    "新闻": "process_news",
    "笑话": "process_joke",
})
# 汇聚到同一个节点
workflow.add_edge("process_general", "text_to_speech")
workflow.add_edge("process_poem", "text_to_speech")
workflow.add_edge("process_news", "text_to_speech")
workflow.add_edge("process_joke", "text_to_speech")
```

**Java 类比**：
```java
// 像策略模式 + 后置处理器
String processed = switch (contentType) {
    case "诗歌" -> poetryProcessor.rewrite(text);
    case "新闻" -> newsProcessor.rewrite(text);
    case "笑话" -> jokeProcessor.rewrite(text);
    default -> text;
};
// 所有分支汇聚到同一个后置处理
audioService.synthesize(processed, voiceMap.get(contentType));
```

### 2. 风格改写（Style Transfer）

同一段文字，不同的 system prompt 产生完全不同的风格：

```python
# 诗歌风格
system = "把以下文字改写为一首优美的诗歌"
# 新闻风格
system = "把以下文字改写为正式的新闻播报稿"
# 笑话风格
system = "把以下文字改写为一个幽默的笑话"
```

### 3. 与第9课的对比

| 维度 | 第9课（客服路由） | **第17课（TTS代理）** |
|------|-----------------|---------------------|
| 分支后 | 各自 → END | **汇聚到TTS节点** |
| 图形状 | 扇形 | **菱形（分支+汇聚）** |
| 分支内容 | 不同领域回答 | **同一文本不同风格** |
| 输出类型 | 纯文本 | **多模态（文本+语音）** |

## 关键洞察

1. **菱形结构 = 分支+汇聚**：4条分支处理完后汇聚到统一的后置处理节点，这比"各自→END"更有结构
2. **风格改写的核心是 system prompt**：同一个用户输入，换不同的 system prompt 就得到不同风格
3. **语音类型映射**：诗歌用柔和的声音、新闻用沉稳的声音、笑话用活泼的声音——类型决定声音

## 参考资源

- [OpenAI TTS API](https://platform.openai.com/docs/guides/text-to-speech)
- [原始教程笔记本](../all_agents_tutorials/tts_poem_generator_agent_langgraph.ipynb)
