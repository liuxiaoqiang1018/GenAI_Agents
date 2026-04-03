# API 配置

## 管理方式

所有课程的 `.env` 以 `md/01/.env` 为唯一源头，其他课程目录从这里复制。

创建新课程时执行：
```bash
cp md/01/.env md/新编号/.env
```

## .env 格式

```env
OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=https://xxx.xxx/v1
MODEL_NAME=gpt-4o-mini
```

## 当前使用的模型配置

> 注意：以下信息可能随时变化，以 `md/01/.env` 中的实际内容为准

- **模型**：gpt-5.4（或 gpt-4o-mini）
- **API 地址**：公益站（OpenAI 兼容接口）
- **调用方式**：全部通过 OpenAI 兼容接口

## 备用模型

| 模型 | API 地址 |
|---|---|
| MiniMax-M2.7 | https://api.minimaxi.com/v1 |
| Kimi-k2.5 | https://api.moonshot.cn/v1 |
| 其他公益站 | 随时可能新增或失效 |

## 注意事项

- API_BASE_URL **必须**包含 `/v1` 后缀
- 公益站 API 不稳定，可能随时 503，需要切换备用
- 全部通过 OpenAI 兼容接口调用（httpx POST 或 ChatOpenAI）
- 切换模型后记得同步到其他课程目录
