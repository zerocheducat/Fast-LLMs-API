# Fast LLMs API — Local OpenAI-compatible Service Console

[**中文**](./README_zh-CN.md) | **English**

A lightweight local LLM deployment and management console. It ships a Gradio-based graphical launcher (`launcher.py`), a FastAPI-based OpenAI-compatible inference backend (`core_api.py`), and a terminal performance monitor (`monitor.py`). The goal is to make local model deployment, secure API exposure, and runtime observation simple and reproducible.

## Core Advantages

- **Extremely Lightweight, Out-of-the-Box**: Pure and minimal architecture with a graphical launcher. Launch a local OpenAI-compatible service with a single click.
- **Truly Zero Cost**: No cloud token billing. Your only investment is the local hardware.
- **Zero Latency & Absolute Privacy**: All data stays on your machine. No network round-trip, no data leakage.
- **Observable by Design**: Built-in `/metrics` endpoint plus a live terminal dashboard (GPU, VRAM, tokens, throughput, KV prefix reuse, TTFT).

## Core Features

- **Multi-Engine Backend**
  - GGUF via `llama-cpp-python` (CUDA accelerated with `n_gpu_layers=-1`)
  - Transformers with native `apply_chat_template` / `parse_response` support
  - Ollama OpenAI-compatible proxy mode
- **Graphical Launcher (Gradio)**
  - Visual model selection (Ollama list + GGUF folder scanning + custom Transformers directory)
  - Configurable Context Window, Max Output Tokens, Service Port
  - Bilingual UI (English / 简体中文)
- **Intelligent Hardware Monitoring**
  - Detects NVIDIA GPU via `nvidia-smi`
  - Estimates VRAM usage from model file size before loading (GGUF)
  - Live refresh from the launcher
- **Security Authentication**
  - Bearer token (`Authorization: Bearer <key>`) on every endpoint
  - Per-model persistent API key stored in `model_api_keys.json`
  - SHA-256 fingerprint written to `service_access.log` on key creation
- **Reasoning / Thinking Filter**
  - Handles `<think>...</think>` and gpt-oss / Harmony channel markers
  - Three modes:
    - `strict`: safest — buffers untagged output until a final marker or end of stream
    - `tagged`: only filters when an explicit reasoning tag is detected
    - `off`: pass raw model output
- **OpenAI-Compatible API Surface**
  - `POST /v1/chat/completions` (streaming SSE and non-streaming)
  - `GET /v1/models`
  - `GET /health`
  - `GET /metrics` (runtime + history + hardware snapshot)
  - Supports `max_completion_tokens`, `max_tokens`, `tools`, `tool_choice`, `stop`, `seed`, `response_format`, `stream_options.include_usage`, etc.
- **Built-in API Debug Panel**
  - Chat Completions test
  - `/v1/models` test
  - Tool Calling test
  - DeepSeek Harness compatibility probe (`max_completion_tokens` + SSE + usage + tool_calls)
- **DeepSeek Harness Integration**
  - Generates provider configuration and a `settings.yaml` snippet
  - Uses `apiKeyEnv` reference instead of plaintext keys
- **Terminal Performance Monitor**
  - No third-party dependency
  - Live GPU / VRAM / temperature / power
  - Token budget, prompt eval / generation tok/s, KV prefix reuse, TTFT (raw & visible), wall time
  - Recent request history sparkline

## Requirements

- Python 3.10.0 – 3.12.0 (the default development environment uses UTF-8 encoding)
- Recommended: NVIDIA GPU with CUDA for hardware acceleration
- Optional backends:
  - `llama-cpp-python` for GGUF
  - `transformers` + `torch` for Transformers mode
  - Ollama running locally for proxy mode

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository
