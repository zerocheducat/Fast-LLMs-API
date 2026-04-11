# Local LLM Service Console

[**中文**](./README_zh-CN.md) | **English**

这是一个轻量级的本地大语言模型（LLM）服务部署与管理控制台。项目提供了一个基于 Gradio 的图形化启动界面和基于 FastAPI 的高性能推理后端，致力于简化本地 AI 模型的配置与调用流程。

## 核心特性
- **多引擎支持**：底层无缝兼容 GGUF (基于 `llama-cpp-python`)、Transformers 以及 Ollama 代理模式。
- **图形化启动器**：可视化选择模型路径、自定义上下文大小 (Context Size) 及服务端口。
- **智能硬件监测**：自动检测 NVIDIA 显卡状态，并在加载前根据模型体积精准预估显存占用，有效防止 OOM 风险。
- **安全鉴权机制**：内置 API Key 动态生成与绑定功能，保障本地 API 接口的安全调用。
- **内置 API 调试面板**：控制台中集成了对话测试组件，可实时发送请求并查看底层 JSON 响应。

## 环境要求
- 推荐 Python 3.10.0 ~ 3.12.0 版本 (默认开发环境使用 UTF-8 编码格式)。
- 建议配备 NVIDIA 显卡以启用 CUDA 硬件加速。

## 安装与运行

1. 克隆本项目到本地：
   ```Bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository

2. 安装所需依赖库：
    ```Bash
    pip install -r requirements.txt
    ```

3. 启动服务控制台：
双击运行 web_api.bat，或在命令行中直接执行：
    ```Bash
    python launcher.py
    ```
4. 在自动打开的浏览器窗口中配置模型并点击"启动服务"，即可获取 OpenAI 兼容格式的 API 地址。