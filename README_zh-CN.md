# Fast LLMs API

**中文** | [English](./README.md)

Fast LLMs API 是一个面向本地大语言模型的轻量级 OpenAI 兼容服务。项目通过 Gradio 提供图形化启动与调试界面，通过 FastAPI 暴露统一的 HTTP API，可将本机 GGUF、Transformers 或 Ollama 模型接入支持 OpenAI-compatible endpoint 的客户端。

项目当前同时包含模型启动、鉴权、Reasoning 过滤、Tool Calling 兼容、API 调试、DeepSeek Harness 配置与实时性能监控等功能。

## 功能概览

### OpenAI 兼容 API

服务提供以下主要接口：

```text
GET  /v1/models
POST /v1/chat/completions
GET  /health
GET  /metrics
```

`/v1/chat/completions` 支持普通与流式响应，并兼容常用的 OpenAI Chat Completions 请求字段。流式请求使用 SSE 返回数据。

当前 API 可处理或转发的能力包括：

- `messages`
- `stream`
- `stream_options`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `typical_p`
- `repeat_penalty`
- `stop`
- `seed`
- `presence_penalty`
- `frequency_penalty`
- `tools`
- `tool_choice`
- `response_format`
- `max_tokens`
- `max_completion_tokens`

具体参数是否生效仍取决于所使用的模型与推理后端。

## 推理后端

### GGUF

使用 `llama-cpp-python` 加载本地 `.gguf` 模型。

当前启动配置包括：

- CUDA GPU offload
- 自定义 Context Window
- 自定义 Max Output Tokens
- 自定义 API Model ID
- 常用采样参数透传
- OpenAI 风格 Tool Calling 请求
- SSE 流式输出
- llama.cpp 性能计数采集

### Transformers

支持直接填写包含 `config.json` 的 Transformers 模型目录，并使用模型自身的 tokenizer 与 chat template。

### Ollama

支持将 Ollama 模型作为后端，通过 Fast LLMs API 对外提供统一的 OpenAI-compatible endpoint。

## 图形化启动器

运行：

```bash
python launcher.py
```

启动器提供中英文界面，并可完成：

- 扫描本地 GGUF 模型目录
- 选择 GGUF / Ollama / Transformers 模型
- 设置 API Model ID
- 设置 Context Window
- 设置 Max Output Tokens
- 设置服务端口
- 查看 NVIDIA GPU、显存使用与显存估算
- 创建或读取模型专属 API Key
- 启动与停止推理服务
- 显示并复制 Base URL 与 Chat Completions URL
- 配置 Reasoning Filter
- 打开独立性能监控终端

默认 OpenAI-compatible Base URL 形式为：

```text
http://127.0.0.1:<port>/v1
```

Chat Completions URL 为：

```text
http://127.0.0.1:<port>/v1/chat/completions
```

## 模型专属 API Key

API Key 按模型持久化保存。

同一个模型再次被选择时，启动器会读取此前创建的 Key，而不是每次启动重新生成。不同模型可以拥有不同的 API Key。

Key 的本地映射由启动器维护在：

```text
model_api_keys.json
```

API 请求使用 Bearer Authentication：

```http
Authorization: Bearer <API_KEY>
```

## Reasoning Filter

不同模型可能输出 `<think>`、Harmony channel 或其他 reasoning 内容。Fast LLMs API 提供三种过滤模式：

### `strict`

优先保证内部 reasoning 不泄漏。

适用于可能出现以下形式的模型：

```text
reasoning...
</think>
final answer
```

或 gpt-oss / Harmony 风格的：

```text
analysis
→ final
```

如果模型没有明确的 reasoning / final 边界，该模式可能需要缓存更多内容后再输出。

### `tagged`

只在检测到明确 reasoning 标记时进入过滤流程。

更适合正常情况下直接输出正文、同时又可能偶尔生成 `<think>` 的模型，可获得更自然的流式体验。

### `off`

关闭 Reasoning Filter，后端生成的文本直接交给客户端。

适合调试模型原始输出。

## 实时性能监控

项目包含独立的：

```text
monitor.py
```

也可以直接在 Launcher 中点击：

```text
Open Performance Monitor
```

启动实时终端仪表盘。

监控器使用原地刷新，不会在终端中不断追加整块界面。按 `Ctrl+C` 退出后会恢复原来的终端画面。

当前监控内容包括：

- Prompt Tokens
- Prompt Eval Tokens
- Completion Tokens
- Total Tokens
- Context Window 使用量与占用率
- Prompt Processing Speed
- Backend Decode Speed
- Raw TTFT
- Visible TTFT
- 请求总耗时
- KV Prefix Reuse 估算
- llama.cpp graph reuse count
- GPU 利用率
- GPU 温度
- GPU 功耗
- 显存占用
- 系统内存占用
- 最近请求历史
- 请求类型、消息数量与工具数量

手动运行示例：

```bash
python monitor.py --url "http://127.0.0.1:7021" --key "<API_KEY>" --interval 0.5
```

也可以通过环境变量提供配置：

```powershell
$env:FAST_LLMS_BASE_URL="http://127.0.0.1:7021"
$env:FAST_LLMS_API_KEY="<API_KEY>"
python monitor.py
```

监控器中的 KV Prefix Reuse 是基于 llama.cpp 上下文与性能计数得到的估算值；`graph reuse count` 是独立的 compute graph 指标，两者并不是同一个概念。

## API 调试

Launcher 内置 `API Debug` 页面，可直接测试当前模型服务。

包括：

- Chat Completions 请求
- 原始 JSON 响应
- `/v1/models`
- Tool Calling

Tool Calling 测试用于确认当前模型与后端是否能够返回结构化的 `tool_calls`。API 能接受工具定义并不意味着所有模型都具备可靠的工具调用能力。

## DeepSeek Harness

Launcher 内置 DeepSeek Harness 配置页面，可生成本地 Provider 配置，并检查：

- OpenAI Completions 连接
- SSE
- `stream_options.include_usage`
- `tool_calls`
- `finish_reason`
- `max_completion_tokens`

界面可以输出 Web 配置参数以及可选的 `settings.yaml` 片段。

## 客户端接入

任何支持自定义 OpenAI-compatible endpoint 的客户端都可以使用 Fast LLMs API。

以 Jan 为例：

```text
Provider Type:
OpenAI-compatible

Base URL:
http://127.0.0.1:7021/v1

API Key:
<当前模型对应的 API Key>
```

模型列表由：

```text
GET /v1/models
```

提供。

模型对话通过：

```text
POST /v1/chat/completions
```

完成。

如果客户端需要 MCP、Tool Calling 或 Agent 能力，还需要所使用的模型本身能够正确理解并输出相应结构。

## 环境要求

推荐：

- Python 3.10 - 3.12
- Windows / Linux
- NVIDIA GPU
- 可用的 CUDA 环境
- 对 GGUF 推理安装支持 CUDA 的 `llama-cpp-python`

项目的具体 Python 依赖以：

```text
requirements.txt
```

为准。

## 安装

```bash
git clone https://github.com/zerocheducat/Fast-LLMs-API.git
cd Fast-LLMs-API
pip install -r requirements.txt
```

启动：

```bash
python launcher.py
```

## 基本使用流程

1. 启动 `launcher.py`。
2. 选择模型。
3. 设置 Context Window 与 Max Output Tokens。
4. 创建或读取该模型的 API Key。
5. 选择合适的 Reasoning Filter。
6. 点击 `Start Service`。
7. 复制 OpenAI-compatible Base URL。
8. 在客户端中添加自定义 OpenAI-compatible Provider。
9. 如需观察性能，打开 Performance Monitor。

## API 示例

查询模型：

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  http://127.0.0.1:7021/v1/models
```

非流式对话：

```bash
curl http://127.0.0.1:7021/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_ID>",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "stream": false
  }'
```

## 项目结构

核心文件：

```text
Fast-LLMs-API/
├── launcher.py
├── core_api.py
├── monitor.py
├── requirements.txt
├── launcher_config.json
├── model_api_keys.json
└── service_access.log
```

其中：

- `launcher.py`：Gradio 图形化控制台。
- `core_api.py`：FastAPI 推理服务与 OpenAI-compatible API。
- `monitor.py`：实时性能终端仪表盘。
- `launcher_config.json`：Launcher 本地配置。
- `model_api_keys.json`：模型与固定 API Key 的本地映射。
- `service_access.log`：本地服务事件记录。

## 适用场景

Fast LLMs API 适合用于：

- 将本地 GGUF 模型接入桌面对话客户端
- 为多个本地 AI 客户端提供统一模型入口
- 本地 Agent / Tool Calling 实验
- MCP 客户端的本地模型后端
- 模型 Prompt、采样参数与 Reasoning 输出调试
- 本地推理性能、Token 与上下文使用情况观察
- 不同模型与量化版本的快速切换和测试
