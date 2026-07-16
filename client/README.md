# Client

先启动本地服务，然后运行流式命令行聊天：

```bash
uv run python -m client.chat
```

输入 `/quit` 退出。服务地址、temperature 和生成长度均可配置：

```bash
uv run python -m client.chat --base-url http://127.0.0.1:8000 --temperature 0.8 --max-tokens 64
```

浏览器客户端是 `client/web.html`。用静态文件服务器打开它：

```bash
uv run python -m http.server 8080 --directory client
```

访问 `http://127.0.0.1:8080/web.html`，页面会调用 `http://127.0.0.1:8000`。
