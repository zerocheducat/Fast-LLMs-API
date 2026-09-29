# Fast LLMs API

[中文](./README_zh-CN.md) | **English**

Fast LLMs API is a lightweight OpenAI-compatible service for local large language models. It provides a Gradio-based launcher and debugging console, while FastAPI exposes a unified HTTP API for local GGUF, Transformers, and Ollama models.

The project currently includes model launching, authentication, reasoning filtering, tool-calling compatibility, API debugging, DeepSeek Harness setup, and live inference monitoring.

## Features

### OpenAI-Compatible API

The service exposes these primary endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
GET  /health
GET  /metrics
```

`/v1/chat/completions` supports both non-streaming and SSE streaming responses and accepts common OpenAI Chat Completions fields.

Currently handled or forwarded fields include:

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

Whether a parameter is effective still depends on the selected model and inference backend.

## Inference Backends

### GGUF

Local `.gguf` models are loaded through `llama-cpp-python`.

The current GGUF path supports:

- CUDA GPU offload
- Configurable Context Window
- Configurable Max Output Tokens
- Custom API Model ID
- Common sampling parameters
- OpenAI-style tool definitions
- SSE streaming
- llama.cpp performance counters

### Transformers

A local Transformers model directory containing `config.json` can be used directly with the model's tokenizer and chat template.

### Ollama

Ollama models can be used as the inference backend while Fast LLMs API exposes a unified OpenAI-compatible endpoint to clients.

## Graphical Launcher

Run:

```bash
python launcher.py
```

The launcher provides both English and Simplified Chinese UI and supports:

- Scanning local GGUF model folders
- Selecting GGUF / Ollama / Transformers models
- Setting the API Model ID
- Configuring Context Window
- Configuring Max Output Tokens
- Configuring the service port
- NVIDIA GPU and VRAM status
- VRAM estimation before model launch
- Creating or loading a persistent per-model API Key
- Starting and stopping the inference service
- Displaying and copying Base URL and Chat Completions URL
- Selecting a Reasoning Filter mode
- Opening the standalone performance monitor

The OpenAI-compatible Base URL has the form:

```text
http://127.0.0.1:<port>/v1
```

The Chat Completions endpoint is:

```text
http://127.0.0.1:<port>/v1/chat/completions
```

## Per-Model API Keys

API keys are persisted per model.

When the same model is selected again, the launcher loads its existing key instead of generating a new key on every startup. Different models can therefore use different API keys.

The local mapping is maintained in:

```text
model_api_keys.json
```

API requests use Bearer authentication:

```http
Authorization: Bearer <API_KEY>
```

## Reasoning Filter

Different models may emit `<think>` blocks, Harmony channels, or other internal reasoning text. Fast LLMs API provides three filtering modes.

### `strict`

Prioritizes preventing internal reasoning from leaking to the client.

It is intended for models that may emit:

```text
reasoning...
</think>
final answer
```

or gpt-oss / Harmony-style:

```text
analysis
→ final
```

If the model does not expose a clear reasoning/final boundary, this mode may buffer more content before displaying the answer.

### `tagged`

Filters reasoning only when an explicit reasoning marker is detected.

This is useful for models that normally stream final text directly but may occasionally emit `<think>` blocks.

### `off`

Disables reasoning filtering and forwards the model output to the client as-is.

This mode is useful for inspecting raw model behavior.

## Live Performance Monitor

The project includes:

```text
monitor.py
```

It can also be launched directly from the Launcher through:

```text
Open Performance Monitor
```

The terminal dashboard redraws in place instead of appending a new full screen on every refresh. `Ctrl+C` exits the monitor and restores the original terminal screen.

Current metrics include:

- Prompt Tokens
- Prompt Eval Tokens
- Completion Tokens
- Total Tokens
- Context Window usage
- Prompt Processing Speed
- Backend Decode Speed
- Raw TTFT
- Visible TTFT
- Total request time
- Estimated KV Prefix Reuse
- llama.cpp graph reuse count
- GPU utilization
- GPU temperature
- GPU power
- VRAM usage
- System memory usage
- Recent request history
- Request type, message count, and tool count

Manual usage:

```bash
python monitor.py --url "http://127.0.0.1:7021" --key "<API_KEY>" --interval 0.5
```

Environment variables can also be used:

```powershell
$env:FAST_LLMS_BASE_URL="http://127.0.0.1:7021"
$env:FAST_LLMS_API_KEY="<API_KEY>"
python monitor.py
```

KV Prefix Reuse is an estimate derived from llama.cpp context/performance counters. `graph reuse count` is a separate compute-graph metric and is not a KV-cache hit counter.

## API Debugging

The Launcher includes an `API Debug` page for testing the running service.

Available checks include:

- Chat Completions
- Raw JSON response
- `/v1/models`
- Tool Calling

The Tool Calling probe checks whether the selected model/backend returns structured `tool_calls`. An API accepting tool definitions does not imply that every model will perform tool calling reliably.

## DeepSeek Harness

The Launcher includes a DeepSeek Harness setup page that can generate local provider configuration and test:

- OpenAI Completions connectivity
- SSE
- `stream_options.include_usage`
- `tool_calls`
- `finish_reason`
- `max_completion_tokens`

The page can output both web configuration values and an optional `settings.yaml` snippet.

## Client Integration

Any client that supports a custom OpenAI-compatible endpoint can connect to Fast LLMs API.

For example, in Jan:

```text
Provider Type:
OpenAI-compatible

Base URL:
http://127.0.0.1:7021/v1

API Key:
<API key assigned to the selected model>
```

The model list is exposed through:

```text
GET /v1/models
```

Chat requests are served through:

```text
POST /v1/chat/completions
```

For MCP, Tool Calling, or agent workflows, the selected model must also be capable of understanding and producing the required structured output.

## Requirements

Recommended environment:

- Python 3.10 - 3.12
- Windows or Linux
- NVIDIA GPU
- Working CUDA environment
- CUDA-enabled `llama-cpp-python` for accelerated GGUF inference

Python package requirements are defined by:

```text
requirements.txt
```

## Installation

```bash
git clone https://github.com/zerocheducat/Fast-LLMs-API.git
cd Fast-LLMs-API
pip install -r requirements.txt
```

Start the launcher:

```bash
python launcher.py
```

## Basic Workflow

1. Start `launcher.py`.
2. Select a model.
3. Configure Context Window and Max Output Tokens.
4. Create or load the model-specific API Key.
5. Select the appropriate Reasoning Filter.
6. Click `Start Service`.
7. Copy the OpenAI-compatible Base URL.
8. Add it as a custom OpenAI-compatible provider in the client.
9. Open the Performance Monitor when runtime metrics are needed.

## API Examples

List models:

```bash
curl -H "Authorization: Bearer <API_KEY>" \
  http://127.0.0.1:7021/v1/models
```

Non-streaming chat completion:

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

## Project Layout

Core files:

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

- `launcher.py`: Gradio graphical launcher and control panel.
- `core_api.py`: FastAPI inference service and OpenAI-compatible API.
- `monitor.py`: live terminal performance dashboard.
- `launcher_config.json`: local Launcher configuration.
- `model_api_keys.json`: local model-to-key mapping.
- `service_access.log`: local service event log.

## Use Cases

Fast LLMs API is suitable for:

- Connecting local GGUF models to desktop chat clients
- Providing one local model endpoint to multiple AI clients
- Local agent and Tool Calling experiments
- Local model backends for MCP-enabled clients
- Prompt, sampling, and reasoning-output debugging
- Monitoring local inference performance, token usage, and context usage
- Quickly switching between different local models and quantizations
