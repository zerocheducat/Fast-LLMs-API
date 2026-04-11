# Local LLM Service Console

[**中文**](./README_zh-CN.md) | **English**

This is a lightweight local Large Language Model (LLM) service deployment and management console. The project provides a Gradio-based graphical launcher and a high-performance inference backend based on FastAPI, dedicated to simplifying the configuration and invocation process of local AI models.

## Core Advantages

- **Extremely Lightweight, Out-of-the-Box**: Say goodbye to bloated and complex dependencies and configurations. Adopting a pure and lightweight architecture design with an intuitive graphical interface, even beginners can launch their exclusive large model service with a single click.
- **Truly Zero Cost**: Completely break free from the anxiety of expensive Token billing from cloud APIs. Enjoy unlimited, smooth conversations—your only investment is the local hardware providing the computational support.
- **Zero Latency & Absolute Privacy**: Data flows entirely within your local device. Not only do you enjoy lightning-fast responses that bypass network bottlenecks, but your core privacy is also fundamentally safeguarded.

## Core Features
- **Multi-Engine Support**: The backend seamlessly integrates with GGUF (via `llama-cpp-python`), Transformers, and Ollama proxy modes.
- **Graphical Launcher**: Visually select model paths, customize Context Size, and set service ports.
- **Intelligent Hardware Monitoring**: Automatically detects NVIDIA GPU status and accurately estimates VRAM usage based on model size before loading, effectively preventing OOM (Out of Memory) risks.
- **Security Authentication**: Built-in dynamic API Key generation and binding to ensure secure invocation of local API endpoints.
- **Built-in API Debugging Panel**: The console integrates a chat testing component to send real-time requests and view underlying JSON responses.

## Requirements
- Python 3.10.0 to 3.12.0 is recommended (The default development environment uses UTF-8 encoding).
- An NVIDIA GPU is highly recommended to enable CUDA hardware acceleration.

## Installation & Usage

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YourUsername/YourRepository.git](https://github.com/YourUsername/YourRepository.git)
   cd YourRepository
   ```
2. Install the required dependencies:

   ```Bash
   pip install -r requirements.txt
   ```
3. Launch the service console:
Double-click web_api.bat to run, or execute directly in the command line:

   ```Bash
   python launcher.py
   ```
4. Configure the model in the automatically opened browser window and click "Start Service" to obtain the OpenAI-compatible API address.