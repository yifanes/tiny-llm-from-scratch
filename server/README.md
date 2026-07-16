# Server

这里实现 OpenAI API 风格服务，支持普通 JSON 响应和 SSE 流式响应。

```http
POST /v1/chat/completions
```

启动服务：

```bash
uv run --group server uvicorn server.app:app --reload
```

请求字段包括 `messages`、`temperature`、`max_tokens` 和 `stream`。默认从
`exports/` 加载 NumPy 模型；测试或其他调用方可以通过 `create_app(backend)`
注入实现 `ChatBackend` 协议的后端。
