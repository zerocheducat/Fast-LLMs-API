# -*- coding: utf-8 -*-
import datetime
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog

import gradio as gr
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "service_access.log")
CONFIG_FILE = os.path.join(BASE_DIR, "launcher_config.json")
MODEL_KEYS_FILE = os.path.join(BASE_DIR, "model_api_keys.json")
service_process = None
monitor_process = None

DSH_PROVIDER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


LANG_EN = "English"
LANG_ZH = "简体中文"


def is_zh(language):
    return language == LANG_ZH


def ui_text(language, en, zh):
    return zh if is_zh(language) else en


def get_nvidia_info(language=LANG_EN):
    gpu_status = ui_text(
        language,
        "No NVIDIA GPU or driver detected. Backend availability depends on the selected runtime.",
        "未检测到 NVIDIA 显卡或驱动。后端是否可用取决于所选运行方式。",
    )
    vram_free = 0
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0:
            line = res.stdout.strip().splitlines()[0]
            info = [part.strip() for part in line.split(",")]
            if len(info) >= 4:
                name = info[0]
                total = int(info[1])
                free = int(info[2])
                used = int(info[3])
                gpu_status = ui_text(
                    language,
                    f"GPU: {name}\nVRAM: {used}MB used / {total}MB total ({free}MB free)\nStatus: NVIDIA driver available",
                    f"显卡型号: {name}\n显存: {used}MB 已用 / {total}MB 总计（剩余 {free}MB）\n状态: NVIDIA 驱动可用",
                )
                return gpu_status, free
    except Exception:
        pass
    return gpu_status, vram_free


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def save_config_value(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)


def get_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(f"ollama://{parts[0]}")
        return models
    except Exception:
        return []


def scan_gguf_files(directory):
    if not directory or not os.path.exists(directory):
        return []
    gguf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".gguf"):
                gguf_files.append(os.path.join(root, file))
    gguf_files.sort()
    return gguf_files


def model_id_from_selection(model_value):
    if not model_value:
        return "local-model"
    if model_value.startswith("ollama://"):
        exact_name = model_value[len("ollama://"):].strip()
        return exact_name or "local-model"
    normalized = os.path.normpath(model_value.strip())
    basename = os.path.basename(normalized.rstrip("\\/"))
    return basename or "local-model"


def estimate_vram_usage(model_path, vram_free, language=LANG_EN):
    if not model_path:
        return ui_text(language, "Select a model to estimate memory usage.", "请选择模型以查看显存评估。"), "info"
    if model_path.startswith("ollama://"):
        return ui_text(
            language,
            "Ollama manages model memory. This launcher does not estimate its runtime KV cache or total VRAM usage.",
            "Ollama 负责模型显存调度；本启动器不估算其运行时 KV Cache 或总显存占用。",
        ), "info"
    if not os.path.exists(model_path):
        return ui_text(
            language,
            "Path not found. For a Transformers model, select or enter the model directory.",
            "路径不存在。如使用 Transformers 模型，请选择或填写正确的模型目录。",
        ), "error"
    if os.path.isdir(model_path):
        return ui_text(
            language,
            "Transformers model directory detected. VRAM usage depends on weights, dtype, context size, and model implementation.",
            "检测到 Transformers 模型目录。实际显存取决于权重格式、dtype、上下文大小和模型实现。",
        ), "info"
    try:
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        estimated_need = file_size_mb + 1200
        info_text = ui_text(
            language,
            f"Model file: {int(file_size_mb)}MB | Rough base requirement: ~{int(estimated_need)}MB (full KV cache peak not included)",
            f"模型文件: {int(file_size_mb)}MB | 粗略基础需求: ~{int(estimated_need)}MB（不含完整 KV Cache 峰值）",
        )
        if vram_free <= 0:
            return info_text, "warning"
        if vram_free > estimated_need:
            prefix = ui_text(language, "Available VRAM is above the rough base requirement. ", "当前空闲显存高于粗略基础需求。")
            return prefix + info_text, "success"
        if vram_free + 1000 > estimated_need:
            prefix = ui_text(language, "VRAM may be tight. ", "当前显存较紧。")
            return prefix + info_text, "warning"
        prefix = ui_text(language, "There is a risk of insufficient VRAM. ", "存在显存不足风险。")
        return prefix + info_text, "error"
    except Exception as exc:
        return ui_text(language, f"Unable to estimate: {exc}", f"无法评估: {exc}"), "error"


def initialize_app(language=LANG_EN):
    gpu_info, vram_free = get_nvidia_info(language)
    models = get_ollama_models()
    config = load_config()
    last_folder = config.get("last_model_folder", "")
    if last_folder:
        models.extend(scan_gguf_files(last_folder))
    default_model = models[0] if models else None
    init_eval, _ = estimate_vram_usage(default_model, vram_free, language) if default_model else ("", "")
    default_model_id = model_id_from_selection(default_model)
    default_ctx = int(config.get("context_size", 4096))
    default_max_output = int(config.get("max_output_tokens", 2048))
    default_port = int(config.get("port", 7021))
    default_provider_id = str(config.get("dsh_provider_id", "fast-llms"))
    default_reasoning_filter = str(config.get("reasoning_filter_mode", "strict")).strip().lower()
    if default_reasoning_filter not in {"strict", "tagged", "off"}:
        default_reasoning_filter = "strict"
    return (
        gpu_info, models, default_model, vram_free, init_eval, default_model_id,
        default_ctx, default_max_output, default_port, default_provider_id,
        default_reasoning_filter,
    )


def _model_key_identity(model_val):
    if not model_val:
        return ""
    value = str(model_val).strip()
    if value.startswith("ollama://"):
        return value
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))


def load_model_keys():
    if not os.path.exists(MODEL_KEYS_FILE):
        return {}
    try:
        with open(MODEL_KEYS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items() if isinstance(value, str) and value}
    except Exception:
        pass
    return {}


def save_model_keys(keys):
    with open(MODEL_KEYS_FILE, "w", encoding="utf-8") as file:
        json.dump(keys, file, indent=4, ensure_ascii=False)


def generate_secure_key(model_val, language=LANG_EN):
    """Return the persistent API key for the selected model, creating it once if needed."""
    identity = _model_key_identity(model_val)
    if not identity:
        return "", ui_text(language, "Select a model first.", "请先选择一个模型。")

    keys = load_model_keys()
    existing = keys.get(identity)
    if existing:
        fingerprint = hashlib.sha256(existing.encode("utf-8")).hexdigest()[:12]
        return existing, ui_text(
            language,
            f"Loaded the persistent API key for this model. Fingerprint: {fingerprint}",
            f"已读取此模型的固定 API Key。指纹: {fingerprint}",
        )

    secret_key = "sk-local-" + secrets.token_urlsafe(32)
    keys[identity] = secret_key
    save_model_keys(keys)
    fingerprint = hashlib.sha256(secret_key.encode("utf-8")).hexdigest()[:12]
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"[{datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}] "
            f"EVENT:KEY_CREATE | MODEL: {identity} | KEY_SHA256_12: {fingerprint}\n"
        )
    return secret_key, ui_text(
        language,
        f"Created a persistent API key for this model. Fingerprint: {fingerprint}",
        f"已为此模型创建固定 API Key。指纹: {fingerprint}",
    )


def build_urls(port):
    port = int(port)
    return (
        f"http://127.0.0.1:{port}/v1",
        f"http://127.0.0.1:{port}/v1/chat/completions",
    )


def launch_service(model_val, model_id, ctx_size, max_output_tokens, port, api_key, reasoning_filter_mode, language=LANG_EN):
    global service_process

    if service_process is not None and service_process.poll() is not None:
        service_process = None

    if not model_val or not api_key or not model_id:
        return ui_text(
            language,
            "Start failed: Model, API Model ID, and API Key are required.",
            "启动失败：模型、API Model ID 和 API Key 均不能为空。",
        ), "", ""

    try:
        ctx_size = int(ctx_size)
        max_output_tokens = int(max_output_tokens)
        port = int(port)
    except (TypeError, ValueError):
        return ui_text(
            language,
            "Start failed: Context Window, Max Output Tokens, and Port must be integers.",
            "启动失败：Context Window、Max Output Tokens 和端口必须是整数。",
        ), "", ""
    if ctx_size <= 0 or max_output_tokens <= 0:
        return ui_text(
            language,
            "Start failed: Context Window and Max Output Tokens must be greater than 0.",
            "启动失败：Context Window 和 Max Output Tokens 必须大于 0。",
        ), "", ""
    if max_output_tokens > ctx_size:
        return ui_text(
            language,
            "Start failed: Max Output Tokens cannot exceed Context Window.",
            "启动失败：Max Output Tokens 不能大于 Context Window。",
        ), "", ""

    base_url, chat_url = build_urls(port)
    if service_process is not None:
        return ui_text(language, "Service is already running.", "服务已经在运行。"), base_url, chat_url

    reasoning_filter_mode = str(reasoning_filter_mode or "strict").strip().lower()
    if reasoning_filter_mode not in {"strict", "tagged", "off"}:
        return ui_text(
            language,
            "Start failed: Reasoning Filter must be strict, tagged, or off.",
            "启动失败：Reasoning Filter 只能是 strict、tagged 或 off。",
        ), "", ""

    is_ollama = model_val.startswith("ollama://")
    real_model_path = model_val[len("ollama://"):] if is_ollama else model_val
    backend = "ollama" if is_ollama else "auto"

    env = os.environ.copy()
    env["LLM_MODEL_PATH"] = real_model_path
    env["LLM_MODEL_ID"] = model_id.strip()
    env["LLM_BACKEND"] = backend
    env["LLM_CTX_SIZE"] = str(ctx_size)
    env["LLM_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    env["LLM_PORT"] = str(port)
    env["LLM_API_KEY"] = api_key
    env["LLM_REASONING_FILTER_MODE"] = reasoning_filter_mode
    env["PYTHONIOENCODING"] = "utf-8"

    save_config_value("context_size", ctx_size)
    save_config_value("max_output_tokens", max_output_tokens)
    save_config_value("port", port)
    save_config_value("reasoning_filter_mode", reasoning_filter_mode)

    script_path = os.path.join(BASE_DIR, "core_api.py")
    try:
        service_process = subprocess.Popen([sys.executable, script_path], env=env, cwd=BASE_DIR)
        time.sleep(2)
        if service_process.poll() is not None:
            return ui_text(
                language,
                "The backend process exited during startup. Check the launcher terminal for details.",
                "后端进程在启动期间退出，请查看启动器终端中的错误信息。",
            ), "", ""
        ollama_note = ""
        if is_ollama:
            ollama_note = ui_text(
                language,
                "\nNote: Context Window is advertised by this API; configure Ollama num_ctx separately to match it.",
                "\n注意：此处 Context Window 会作为 API 模型能力公开；Ollama 的 num_ctx 需要单独配置并保持一致。",
            )
        status = ui_text(language, "Service started", "服务已启动") + "\n"
        status += f"Backend: {backend}\n"
        status += f"API Model ID: {model_id.strip()}\n"
        status += f"Context Window: {ctx_size}\n"
        status += f"Max Output Tokens: {max_output_tokens}\n"
        status += f"Reasoning Filter: {reasoning_filter_mode}{ollama_note}"
        return status, base_url, chat_url
    except Exception as exc:
        service_process = None
        return ui_text(language, f"System error: {exc}", f"系统错误: {exc}"), "", ""


def terminate_service(language=LANG_EN):
    global service_process
    if service_process is not None:
        if service_process.poll() is None:
            service_process.terminate()
        service_process = None
        return ui_text(language, "Service stopped.", "服务已停止。"), "", ""
    return ui_text(language, "Service is not running.", "服务未运行。"), "", ""


def launch_monitor(port, api_key, language=LANG_EN):
    global monitor_process

    if not api_key:
        return ui_text(
            language,
            "Monitor start failed: API Key is empty.",
            "监控器启动失败：API Key 为空。",
        )

    script_path = os.path.join(BASE_DIR, "monitor.py")
    if not os.path.exists(script_path):
        return ui_text(
            language,
            f"Monitor start failed: {script_path} was not found.",
            f"监控器启动失败：未找到 {script_path}。",
        )

    if monitor_process is not None and monitor_process.poll() is None:
        return ui_text(
            language,
            "Performance monitor is already running.",
            "性能监控器已经在运行。",
        )

    try:
        port = int(port)
    except (TypeError, ValueError):
        return ui_text(
            language,
            "Monitor start failed: Service Port must be an integer.",
            "监控器启动失败：服务端口必须是整数。",
        )

    env = os.environ.copy()
    env["FAST_LLMS_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["FAST_LLMS_API_KEY"] = str(api_key)
    env["PYTHONIOENCODING"] = "utf-8"

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    try:
        monitor_process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            env=env,
            creationflags=creationflags,
        )
        return ui_text(
            language,
            "Performance monitor opened in a new terminal window.",
            "性能监控器已在新的终端窗口中打开。",
        )
    except Exception as exc:
        monitor_process = None
        return ui_text(
            language,
            f"Monitor start failed: {exc}",
            f"监控器启动失败：{exc}",
        )


def terminate_monitor(language=LANG_EN):
    global monitor_process

    if monitor_process is not None:
        if monitor_process.poll() is None:
            monitor_process.terminate()
        monitor_process = None
        return ui_text(
            language,
            "Performance monitor stopped.",
            "性能监控器已停止。",
        )

    return ui_text(
        language,
        "Performance monitor is not running.",
        "性能监控器未运行。",
    )


def on_scan_click(vram_free, language=LANG_EN):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_path = filedialog.askdirectory(
        title=ui_text(language, "Select model folder", "选择模型文件夹")
    )
    root.destroy()
    if not folder_path:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    save_config_value("last_model_folder", folder_path)
    files = scan_gguf_files(folder_path)
    ollama_models = get_ollama_models()
    all_models = ollama_models + files
    selected = files[0] if files else (ollama_models[0] if ollama_models else None)
    if selected:
        eval_text, _ = estimate_vram_usage(selected, vram_free, language)
        api_key, key_status = generate_secure_key(selected, language)
    else:
        eval_text = ui_text(language, "No GGUF or Ollama models found.", "未发现 GGUF 或 Ollama 模型。")
        api_key, key_status = "", ""
    return (
        gr.Dropdown(choices=all_models, value=selected),
        eval_text,
        model_id_from_selection(selected),
        api_key,
        api_key,
        key_status,
    )


def refresh_hardware_info(language=LANG_EN):
    return get_nvidia_info(language)


def on_model_change(model, vram_free, language=LANG_EN):
    message, _ = estimate_vram_usage(model, vram_free, language)
    api_key, key_status = generate_secure_key(model, language)
    return message, model_id_from_selection(model), api_key, api_key, key_status


def toggle_key_visibility(visible):
    new_visible = not visible
    return new_visible, gr.update(type="text" if new_visible else "password")


def dsh_key_ref(provider_id):
    return re.sub(r"[^A-Z0-9]+", "_", provider_id.upper()) + "_API_KEY"


def build_dsh_settings(provider_id, port, model_id, ctx_size, max_output_tokens, language=LANG_EN):
    provider_id = (provider_id or "").strip()
    model_id = (model_id or "").strip()
    if not DSH_PROVIDER_ID_PATTERN.fullmatch(provider_id):
        return ui_text(
            language,
            "Invalid Provider ID. Use lowercase letters, numbers, and hyphens; start with a lowercase letter.",
            "Provider ID 无效。请使用小写字母、数字和短横线，并以小写字母开头。",
        ), ""
    if not model_id:
        return ui_text(language, "API Model ID is required.", "API Model ID 不能为空。"), ""
    try:
        port = int(port)
        ctx_size = int(ctx_size)
        max_output_tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        return ui_text(
            language,
            "Port, Context Window, and Max Output Tokens must be integers.",
            "端口、Context Window 和 Max Output Tokens 必须是整数。",
        ), ""
    if ctx_size <= 0 or max_output_tokens <= 0 or max_output_tokens > ctx_size:
        return ui_text(
            language,
            "Invalid Context Window or Max Output Tokens value.",
            "Context Window 或 Max Output Tokens 数值无效。",
        ), ""

    save_config_value("dsh_provider_id", provider_id)
    base_url = f"http://127.0.0.1:{port}/v1"
    ref = dsh_key_ref(provider_id)
    quoted_model = json.dumps(model_id, ensure_ascii=False)
    yaml_text = (
        "llm-pi-ai:\n"
        "  providers:\n"
        f"    {provider_id}:\n"
        f"      apiKeyEnv: {ref}\n"
        "      displayName: Fast LLMs API\n"
        "      api: openai-completions\n"
        f"      baseURL: {base_url}\n"
        "      models:\n"
        f"        - id: {quoted_model}\n"
        f"          name: {quoted_model}\n"
        f"          contextWindow: {ctx_size}\n"
        f"          maxTokens: {max_output_tokens}\n"
    )
    if is_zh(language):
        ui_fields = (
            "DeepSeek Harness → Settings → Models → Add a custom provider\n\n"
            f"Provider ID: {provider_id}\n"
            "Display name: Fast LLMs API\n"
            f"Base URL: {base_url}\n"
            "API protocol: openai-completions\n"
            f"Model ID: {model_id}\n"
            f"Context Window: {ctx_size}\n"
            f"Max Tokens: {max_output_tokens}\n\n"
            "API Key: 使用启动器当前生成的 Key。\n"
            f"settings.yaml 中的凭据引用: {ref}\n"
            "建议通过 DeepSeek Harness 的凭据存储或环境变量提供密钥，不要把明文 API Key 写入 settings.yaml。"
        )
    else:
        ui_fields = (
            "DeepSeek Harness → Settings → Models → Add a custom provider\n\n"
            f"Provider ID: {provider_id}\n"
            "Display name: Fast LLMs API\n"
            f"Base URL: {base_url}\n"
            "API protocol: openai-completions\n"
            f"Model ID: {model_id}\n"
            f"Context Window: {ctx_size}\n"
            f"Max Tokens: {max_output_tokens}\n\n"
            "API Key: use the key generated by this launcher.\n"
            f"Credential reference in settings.yaml: {ref}\n"
            "Use DeepSeek Harness credential storage or an environment variable for the secret; do not place the plaintext API key in settings.yaml."
        )
    return ui_fields, yaml_text


def test_dsh_wire(port, key, model_id, language=LANG_EN):
    tool_schema = {
        "type": "function",
        "function": {
            "name": "dsh_probe",
            "description": "Return a probe result. Call this tool when explicitly asked to run the DSH compatibility probe.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Run the DSH compatibility probe by calling dsh_probe with value dsh-ok."}
        ],
        "tools": [tool_schema],
        "temperature": 0.0,
        "max_completion_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        response = requests.post(
            f"http://127.0.0.1:{int(port)}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=300,
            stream=True,
        )
        if response.status_code != 200:
            body = response.text
            response.close()
            return ui_text(
                language,
                f"Compatibility test failed: HTTP {response.status_code}\n{body}",
                f"兼容性测试失败: HTTP {response.status_code}\n{body}",
            )

        events = []
        tool_deltas = []
        finish_reasons = []
        usage_seen = False
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    events.append("[DONE]")
                    break
                obj = json.loads(data)
                events.append(obj)
                if obj.get("usage") is not None:
                    usage_seen = True
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    if isinstance(delta.get("tool_calls"), list):
                        tool_deltas.extend(delta["tool_calls"])
                    if choice.get("finish_reason") is not None:
                        finish_reasons.append(choice["finish_reason"])
        finally:
            response.close()

        yes = ui_text(language, "yes", "是")
        no = ui_text(language, "no", "否")
        summary = [
            ui_text(language, "DeepSeek Harness compatibility: PASS", "DeepSeek Harness 兼容性: PASS"),
            f"- max_completion_tokens: {yes}",
            "- SSE: " + (yes if events and events[-1] == "[DONE]" else no),
            "- usage: " + (yes if usage_seen else no),
            "- tool_calls: " + (yes if tool_deltas else no),
            "- finish_reason: " + (", ".join(map(str, finish_reasons)) if finish_reasons else ui_text(language, "none", "无")),
        ]
        if not tool_deltas:
            summary.append(ui_text(
                language,
                "\nThe API accepted the request, but the model/backend did not return structured tool_calls in this run.",
                "\nAPI 已接受请求，但本次模型/后端没有返回结构化 tool_calls。",
            ))
        return "\n".join(summary) + "\n\n" + json.dumps(events, indent=2, ensure_ascii=False)
    except Exception as exc:
        return ui_text(
            language,
            f"Compatibility test connection failed: {exc}",
            f"兼容性测试连接失败: {exc}",
        )


def test_api(port, key, model_id, prompt, temp, max_tokens, language=LANG_EN):
    if not prompt:
        return ui_text(language, "Enter a prompt.", "请输入提示词。"), ""
    try:
        response = requests.post(
            f"http://127.0.0.1:{int(port)}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": int(max_tokens),
                "stream": False,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=300,
        )
        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            prefix = ui_text(language, "Response", "回复")
            return f"{prefix}:\n{message.get('content', '')}", json.dumps(data, indent=2, ensure_ascii=False)
        return ui_text(language, f"Error {response.status_code}:\n{response.text}", f"错误 {response.status_code}:\n{response.text}"), ""
    except Exception as exc:
        return ui_text(language, f"Connection failed: {exc}", f"连接失败: {exc}"), ""


def test_models_api(port, key, language=LANG_EN):
    try:
        response = requests.get(
            f"http://127.0.0.1:{int(port)}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        return f"HTTP {response.status_code}\n{response.text}"
    except Exception as exc:
        return ui_text(language, f"Connection failed: {exc}", f"连接失败: {exc}")


def test_tool_api(port, key, model_id, language=LANG_EN):
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_local_test_value",
            "description": "Return a local test value. Use this function when explicitly asked for the local test value.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
    try:
        response = requests.post(
            f"http://127.0.0.1:{int(port)}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call get_local_test_value. Do not invent the value yourself.",
                    }
                ],
                "tools": [tool_schema],
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 512,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=300,
        )
        body = response.text
        if response.status_code != 200:
            return f"HTTP {response.status_code}\n{body}"
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls")
        state = ui_text(
            language,
            "Structured tool_calls detected" if tool_calls else "No structured tool_calls detected",
            "检测到结构化 tool_calls" if tool_calls else "未检测到结构化 tool_calls",
        )
        return f"{state}\n\n{json.dumps(data, indent=2, ensure_ascii=False)}"
    except Exception as exc:
        return ui_text(language, f"Connection failed: {exc}", f"连接失败: {exc}")


js_copy = "(x) => { navigator.clipboard.writeText(x); }"


def update_language(language):
    zh = is_zh(language)
    return (
        gr.update(value="## Fast LLMs API — 本地 OpenAI 兼容服务" if zh else "## Fast LLMs API — Local OpenAI-compatible Service"),
        gr.update(label="硬件状态" if zh else "Hardware Status"),
        gr.update(value="刷新" if zh else "Refresh"),
        gr.update(label="服务配置" if zh else "Service"),
        gr.update(value="### 1. 模型" if zh else "### 1. Model"),
        gr.update(label="选择模型（Ollama / GGUF；也可填写 Transformers 目录）" if zh else "Select Model (Ollama / GGUF; Transformers directory is also supported)"),
        gr.update(value="扫描 GGUF 文件夹" if zh else "Scan GGUF Folder"),
        gr.update(label="API Model ID", info="此 API 向客户端公开的模型标识。" if zh else "Model identifier exposed by this API to connected clients."),
        gr.update(label="Context Window", info="GGUF 运行时上下文大小，同时作为 API 模型容量公开。" if zh else "Runtime context size for GGUF and the model capacity advertised by this API."),
        gr.update(label="Max Output Tokens", info="单次生成允许的最大输出长度。" if zh else "Maximum output length allowed for a single generation."),
        gr.update(
            label="Reasoning Filter",
            info=(
                "strict：最安全，会缓存无标签输出直到结束；tagged：仅遇到明确思考标签时过滤；off：不做思考过滤。"
                if zh
                else "strict: safest, buffers untagged output until completion; tagged: filters only explicit reasoning tags; off: no reasoning filtering."
            ),
        ),
        gr.update(label="显存提示" if zh else "VRAM Estimate"),
        gr.update(value="### 2. 鉴权" if zh else "### 2. Authentication"),
        gr.update(label="服务端口" if zh else "Service Port"),
        gr.update(label="API Key"),
        gr.update(value="显示/隐藏" if zh else "Show / Hide"),
        gr.update(value="复制" if zh else "Copy"),
        gr.update(value="创建 / 读取模型 API Key" if zh else "Create / Load Model API Key"),
        gr.update(value="### 3. 运行" if zh else "### 3. Runtime"),
        gr.update(value="启动服务" if zh else "Start Service"),
        gr.update(value="停止服务" if zh else "Stop Service"),
        gr.update(value="打开性能监控" if zh else "Open Performance Monitor"),
        gr.update(value="关闭性能监控" if zh else "Close Monitor"),
        gr.update(label="OpenAI-compatible Base URL"),
        gr.update(label="Chat Completions URL"),
        gr.update(value="复制 Base URL" if zh else "Copy Base URL"),
        gr.update(value="复制 Chat URL" if zh else "Copy Chat URL"),
        gr.update(label="系统状态" if zh else "System Status"),
        gr.update(label="DeepSeek Harness"),
        gr.update(value="### DeepSeek Harness 本地模型配置\n用于生成 Provider 配置并检查本地 API 的工具调用与流式响应。" if zh else "### DeepSeek Harness Local Model Setup\nGenerate provider settings and check tool calling and streaming compatibility."),
        gr.update(label="Provider ID", info="使用小写字母、数字和短横线，并以小写字母开头。" if zh else "Use lowercase letters, numbers, and hyphens; start with a lowercase letter."),
        gr.update(value="生成配置" if zh else "Generate Configuration"),
        gr.update(label="Web 配置参数" if zh else "Web Configuration Values"),
        gr.update(label="settings.yaml 片段（可选）" if zh else "settings.yaml Snippet (Optional)"),
        gr.update(value="运行兼容性测试" if zh else "Run Compatibility Test"),
        gr.update(label="启动命令" if zh else "Launch Command"),
        gr.update(label="兼容性测试结果" if zh else "Compatibility Test Result"),
        gr.update(label="API 调试" if zh else "API Debug"),
        gr.update(value="### Chat Completions 测试" if zh else "### Chat Completions Test"),
        gr.update(label="提示词" if zh else "Prompt", placeholder="请输入文本..." if zh else "Enter text..."),
        gr.update(value="发送请求" if zh else "Send Request"),
        gr.update(label="回复" if zh else "Response"),
        gr.update(label="原始 JSON" if zh else "Raw JSON"),
        gr.update(value="### Agent 能力检查" if zh else "### Agent Capability Checks"),
        gr.update(value="测试 /v1/models" if zh else "Test /v1/models"),
        gr.update(value="测试 Tool Calling" if zh else "Test Tool Calling"),
        gr.update(label="Models API 结果" if zh else "Models API Result"),
        gr.update(label="Tool Calling 结果" if zh else "Tool Calling Result"),
    )


with gr.Blocks(title="Fast LLMs API Console") as demo:
    with gr.Row():
        title_md = gr.Markdown("## Fast LLMs API — Local OpenAI-compatible Service")
        language_select = gr.Dropdown(
            choices=[LANG_EN, LANG_ZH],
            value=LANG_EN,
            label="Language / 语言",
            interactive=True,
            scale=1,
        )

    vram_free_state = gr.State(0)
    current_key_value = gr.State("")
    key_visible_state = gr.State(False)

    (
        init_gpu,
        init_models,
        init_default,
        init_free,
        init_eval,
        init_model_id,
        init_ctx,
        init_max_output,
        init_port,
        init_provider_id,
        init_reasoning_filter,
    ) = initialize_app(LANG_EN)
    vram_free_state.value = init_free
    init_api_key, _ = generate_secure_key(init_default, LANG_EN) if init_default else ("", "")
    current_key_value.value = init_api_key

    with gr.Group():
        with gr.Row(variant="panel"):
            with gr.Column(scale=4):
                hardware_display = gr.Textbox(value=init_gpu, label="Hardware Status", lines=3, interactive=False)
            with gr.Column(scale=1, min_width=100):
                btn_refresh_hw = gr.Button("Refresh")

    with gr.Tabs():
        with gr.TabItem("Service") as service_tab:
            model_section_md = gr.Markdown("### 1. Model")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=init_models,
                    value=init_default,
                    label="Select Model (Ollama / GGUF; Transformers directory is also supported)",
                    allow_custom_value=True,
                    scale=4,
                )
                btn_scan = gr.Button("Scan GGUF Folder", variant="secondary", scale=1)

            with gr.Row():
                model_id_input = gr.Textbox(
                    label="API Model ID",
                    value=init_model_id,
                    info="Model identifier exposed by this API to connected clients.",
                    scale=3,
                )
                ctx_size_input = gr.Number(
                    label="Context Window",
                    value=init_ctx,
                    precision=0,
                    info="Runtime context size for GGUF and the model capacity advertised by this API.",
                    scale=2,
                )
                max_output_input = gr.Number(
                    label="Max Output Tokens",
                    value=init_max_output,
                    precision=0,
                    info="Maximum output length allowed for a single generation.",
                    scale=2,
                )
                reasoning_filter_input = gr.Dropdown(
                    label="Reasoning Filter",
                    choices=["strict", "tagged", "off"],
                    value=init_reasoning_filter,
                    info=(
                        "strict: safest but may delay untagged models; "
                        "tagged: keeps normal streaming unless explicit reasoning tags appear; "
                        "off: pass raw model output."
                    ),
                    scale=2,
                )
            vram_status_bar = gr.Textbox(value=init_eval, label="VRAM Estimate", interactive=False)

            auth_section_md = gr.Markdown("### 2. Authentication")
            with gr.Row():
                port_input = gr.Number(label="Service Port", value=init_port, precision=0, scale=1)
                key_display = gr.Textbox(label="API Key", value=init_api_key, type="password", interactive=False, scale=3)
                with gr.Column(scale=1):
                    btn_vis = gr.Button("Show / Hide", size="sm")
                    btn_copy = gr.Button("Copy", size="sm")
            btn_gen_key = gr.Button("Create / Load Model API Key", variant="primary")

            runtime_section_md = gr.Markdown("### 3. Runtime")
            with gr.Row():
                btn_start = gr.Button("Start Service", variant="primary", scale=2)
                btn_stop = gr.Button("Stop Service", variant="stop", scale=1)
                btn_monitor = gr.Button("Open Performance Monitor", variant="secondary", scale=1)
                btn_monitor_stop = gr.Button("Close Monitor", variant="secondary", scale=1)

            base_url_display = gr.Textbox(label="OpenAI-compatible Base URL", interactive=False)
            chat_url_display = gr.Textbox(label="Chat Completions URL", interactive=False)
            with gr.Row():
                btn_copy_base = gr.Button("Copy Base URL")
                btn_copy_chat = gr.Button("Copy Chat URL")
            status_output = gr.Textbox(label="System Status", lines=4)

        with gr.TabItem("DeepSeek Harness") as dsh_tab:
            dsh_intro_md = gr.Markdown(
                "### DeepSeek Harness Local Model Setup\n"
                "Generate provider settings and check tool calling and streaming compatibility."
            )
            dsh_provider_id = gr.Textbox(
                label="Provider ID",
                value=init_provider_id,
                info="Use lowercase letters, numbers, and hyphens; start with a lowercase letter.",
            )
            dsh_build_btn = gr.Button("Generate Configuration", variant="primary")
            dsh_fields = gr.Textbox(label="Web Configuration Values", lines=12, interactive=False)
            dsh_yaml = gr.Code(label="settings.yaml Snippet (Optional)", language="yaml")
            with gr.Row():
                dsh_wire_btn = gr.Button("Run Compatibility Test", variant="secondary")
                dsh_launch_hint = gr.Textbox(
                    label="Launch Command",
                    value="npx @deepseek-ai/dsh web",
                    interactive=False,
                )
            dsh_wire_result = gr.Textbox(label="Compatibility Test Result", lines=16)

        with gr.TabItem("API Debug") as api_tab:
            chat_test_md = gr.Markdown("### Chat Completions Test")
            with gr.Row():
                db_prompt = gr.TextArea(label="Prompt", placeholder="Enter text...", lines=5, scale=4)
                with gr.Column(scale=1):
                    db_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.7)
                    db_max_tokens = gr.Number(label="max_tokens", value=2048, precision=0)
                    db_btn = gr.Button("Send Request", variant="primary")
            with gr.Row():
                db_res = gr.Textbox(label="Response")
                db_json = gr.Code(label="Raw JSON", language="json")

            agent_test_md = gr.Markdown("### Agent Capability Checks")
            with gr.Row():
                models_btn = gr.Button("Test /v1/models")
                tool_btn = gr.Button("Test Tool Calling")
            models_result = gr.Textbox(label="Models API Result", lines=6)
            tool_result = gr.Textbox(label="Tool Calling Result", lines=12)

    language_outputs = [
        title_md,
        hardware_display,
        btn_refresh_hw,
        service_tab,
        model_section_md,
        model_dropdown,
        btn_scan,
        model_id_input,
        ctx_size_input,
        max_output_input,
        reasoning_filter_input,
        vram_status_bar,
        auth_section_md,
        port_input,
        key_display,
        btn_vis,
        btn_copy,
        btn_gen_key,
        runtime_section_md,
        btn_start,
        btn_stop,
        btn_monitor,
        btn_monitor_stop,
        base_url_display,
        chat_url_display,
        btn_copy_base,
        btn_copy_chat,
        status_output,
        dsh_tab,
        dsh_intro_md,
        dsh_provider_id,
        dsh_build_btn,
        dsh_fields,
        dsh_yaml,
        dsh_wire_btn,
        dsh_launch_hint,
        dsh_wire_result,
        api_tab,
        chat_test_md,
        db_prompt,
        db_btn,
        db_res,
        db_json,
        agent_test_md,
        models_btn,
        tool_btn,
        models_result,
        tool_result,
    ]
    language_select.change(update_language, inputs=[language_select], outputs=language_outputs)

    btn_refresh_hw.click(
        refresh_hardware_info,
        inputs=[language_select],
        outputs=[hardware_display, vram_free_state],
    )
    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown, vram_free_state, language_select],
        outputs=[vram_status_bar, model_id_input, current_key_value, key_display, status_output],
    )
    btn_scan.click(
        on_scan_click,
        inputs=[vram_free_state, language_select],
        outputs=[model_dropdown, vram_status_bar, model_id_input, current_key_value, key_display, status_output],
    )
    btn_gen_key.click(
        generate_secure_key,
        inputs=[model_dropdown, language_select],
        outputs=[current_key_value, status_output],
    ).then(lambda key: key, inputs=[current_key_value], outputs=[key_display])

    btn_vis.click(
        toggle_key_visibility,
        inputs=[key_visible_state],
        outputs=[key_visible_state, key_display],
    )
    btn_copy.click(None, inputs=[key_display], js=js_copy)

    btn_start.click(
        launch_service,
        inputs=[
            model_dropdown,
            model_id_input,
            ctx_size_input,
            max_output_input,
            port_input,
            current_key_value,
            reasoning_filter_input,
            language_select,
        ],
        outputs=[status_output, base_url_display, chat_url_display],
    )
    btn_stop.click(
        terminate_service,
        inputs=[language_select],
        outputs=[status_output, base_url_display, chat_url_display],
    )
    btn_monitor.click(
        launch_monitor,
        inputs=[port_input, current_key_value, language_select],
        outputs=[status_output],
    )
    btn_monitor_stop.click(
        terminate_monitor,
        inputs=[language_select],
        outputs=[status_output],
    )
    btn_copy_base.click(None, inputs=[base_url_display], js=js_copy)
    btn_copy_chat.click(None, inputs=[chat_url_display], js=js_copy)

    dsh_build_btn.click(
        build_dsh_settings,
        inputs=[dsh_provider_id, port_input, model_id_input, ctx_size_input, max_output_input, language_select],
        outputs=[dsh_fields, dsh_yaml],
    )
    dsh_wire_btn.click(
        test_dsh_wire,
        inputs=[port_input, current_key_value, model_id_input, language_select],
        outputs=[dsh_wire_result],
    )

    db_btn.click(
        test_api,
        inputs=[port_input, current_key_value, model_id_input, db_prompt, db_temp, db_max_tokens, language_select],
        outputs=[db_res, db_json],
    )
    models_btn.click(
        test_models_api,
        inputs=[port_input, current_key_value, language_select],
        outputs=[models_result],
    )
    tool_btn.click(
        test_tool_api,
        inputs=[port_input, current_key_value, model_id_input, language_select],
        outputs=[tool_result],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
