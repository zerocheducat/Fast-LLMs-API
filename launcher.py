# -*- coding: utf-8 -*-
import gradio as gr
import subprocess
import os
import sys
import time
import hashlib
import datetime
import json
import requests
import tkinter as tk
from tkinter import filedialog
import shutil

service_process = None
LOG_FILE = "service_access.log"
CONFIG_FILE = "launcher_config.json"


def get_nvidia_info():
    gpu_status = "No NVIDIA GPU or driver detected (using CPU)"
    vram_free = 0
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            info = res.stdout.strip().split(', ')
            name = info[0]
            total = int(info[1])
            free = int(info[2])
            used = int(info[3])
            gpu_status = (
                f"GPU Model: {name}\n"
                f"VRAM Details: {used}MB Used / {total}MB Total (Free: {free}MB)\n"
                f"Status: Driver OK, CUDA Ready"
            )
            return gpu_status, free
    except Exception:
        pass
    return gpu_status, 0


def estimate_vram_usage(model_path, vram_free):
    if not model_path:
        return "Please select a model to view evaluation", "info"
    if model_path.startswith("ollama://"):
        return "Ollama models automatically manage VRAM (usually optimized)", "success"
    if not os.path.exists(model_path):
        return "File not found", "error"
    try:
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        estimated_need = file_size_mb + 1200
        info_text = f"Model Size: {int(file_size_mb)}MB | Est. VRAM Need: ~{int(estimated_need)}MB"
        if vram_free <= 0:
            return f"{info_text} (No GPU detected, using System RAM)", "warning"
        if vram_free > estimated_need:
            return f"VRAM Sufficient. {info_text}", "success"
        elif vram_free + 1000 > estimated_need:
            return f"VRAM Tight. {info_text} (May use shared memory, slight speed drop)", "warning"
        else:
            return f"VRAM Insufficient Risk! {info_text} (High OOM probability or extremely slow)", "error"
    except Exception as e:
        return f"Evaluation failed: {str(e)}", "error"


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_config(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0: return []
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts: models.append(f"ollama://{parts[0]}")
        return models
    except:
        return []


def scan_gguf_files(directory):
    if not directory or not os.path.exists(directory): return []
    gguf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.gguf'):
                gguf_files.append(os.path.join(root, file))
    return gguf_files


def initialize_app():
    gpu_info, vram_free = get_nvidia_info()
    models = get_ollama_models()
    config = load_config()
    last_folder = config.get("last_model_folder", "")
    if last_folder:
        models.extend(scan_gguf_files(last_folder))
    default_val = models[0] if models else None
    init_eval, _ = estimate_vram_usage(default_val, vram_free) if default_val else ("", "")
    return gpu_info, models, default_val, vram_free, init_eval


def generate_secure_key(model_val):
    clean_name = model_val.replace("ollama://", "") if model_val else ""
    if not clean_name: return "", "Error: Please select a model first"
    salt = "local_deployment_secure_salt_v1"
    raw_str = f"{clean_name}{datetime.date.today()}{salt}"
    secret_key = "sk-" + hashlib.sha256(raw_str.encode()).hexdigest()[:24]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] EVENT:KEY_GEN | MODEL: {clean_name} | KEY: {secret_key}\n")
    return secret_key, f"Key generated and bound: {clean_name}"


def launch_service(model_val, ctx_size, port, api_key):
    global service_process
    if not api_key:
        return "Startup failed: No valid key detected, please generate one first", ""
    real_model_path = model_val.replace("ollama://", "") if model_val else ""
    if not real_model_path:
        return "Startup failed: No model selected", ""
    if service_process is not None:
        return "Service is already running (Stop it first to change key/model)", f"http://127.0.0.1:{int(port)}/v1/chat/completions"

    env = os.environ.copy()
    env["LLM_MODEL_PATH"] = real_model_path
    env["LLM_CTX_SIZE"] = str(int(ctx_size))
    env["LLM_PORT"] = str(int(port))
    env["LLM_API_KEY"] = api_key
    env["PYTHONIOENCODING"] = "utf-8"

    script_path = os.path.join(os.path.dirname(__file__), "core_api.py")
    try:
        service_process = subprocess.Popen([sys.executable, script_path], env=env)
        time.sleep(3)
        if service_process.poll() is not None:
            return "Error: Service startup failed, check console output.", ""
        return f"Service started\nModel: {os.path.basename(real_model_path)}", f"http://127.0.0.1:{int(port)}/v1/chat/completions"
    except Exception as e:
        return f"System error: {str(e)}", ""


def terminate_service():
    global service_process
    if service_process:
        service_process.terminate()
        service_process = None
        return "Service stopped", ""
    return "Service is not running", ""


def on_scan_click():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Model Folder")
    root.destroy()
    if not folder_path: return gr.update(), gr.update()
    save_config("last_model_folder", folder_path)
    files = scan_gguf_files(folder_path)
    ollama_mods = get_ollama_models()
    all_models = ollama_mods + files
    return gr.Dropdown(choices=all_models, value=files[0] if files else None,
                       label=f"Scanned Path: {folder_path}"), folder_path


def refresh_hardware_info():
    info, free = get_nvidia_info()
    return info, free


def on_model_change(model, vram_free):
    msg, _ = estimate_vram_usage(model, vram_free)
    return msg


def toggle_key_visibility(is_visible):
    new_state = not is_visible
    return new_state, gr.update(type="text" if new_state else "password")


js_copy = "(x) => { navigator.clipboard.writeText(x); alert('Copied to clipboard'); }"

with gr.Blocks(title="Local Model Service Console") as demo:
    gr.Markdown("## Local Large Language Models Service Console")
    vram_free_state = gr.State(0)
    current_key_value = gr.State("")
    key_visible_state = gr.State(False)
    init_gpu, init_models, init_def, init_free, init_eval = initialize_app()
    vram_free_state.value = init_free

    with gr.Group():
        with gr.Row(variant="panel"):
            with gr.Column(scale=4):
                hardware_display = gr.Textbox(value=init_gpu, label="Hardware Monitor", lines=3, interactive=False)
            with gr.Column(scale=1, min_width=100):
                btn_refresh_hw = gr.Button("Refresh Status")

    with gr.Tabs():
        with gr.TabItem("Service Configuration"):
            gr.Markdown("### 1. Model Loading")
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=init_models, value=init_def, label="Select Model (Supports Ollama / GGUF)",
                                             allow_custom_value=True, scale=4)
                btn_scan = gr.Button("Scan New Folder", variant="secondary", scale=1)

            ctx_size_input = gr.Slider(label="Model Context Size", minimum=512, maximum=32768, step=512,
                                       value=4096)
            vram_status_bar = gr.Textbox(value=init_eval, label="Estimated VRAM Usage", interactive=False)

            gr.Markdown("### 2. Security & Authentication")
            with gr.Row():
                port_input = gr.Number(label="Service Port", value=7021, precision=0, scale=1)
                key_display = gr.Textbox(label="API Key (Hidden)", type="password", interactive=False, scale=3)
                with gr.Column(scale=1):
                    with gr.Row():
                        btn_vis = gr.Button("Show/Hide", size="sm")
                        btn_copy = gr.Button("Copy", size="sm")
            btn_gen_key = gr.Button("Generate & Bind Key", variant="primary")

            gr.Markdown("### 3. Execution Control")
            with gr.Row():
                btn_start = gr.Button("Start Service", variant="primary", scale=2)
                btn_stop = gr.Button("Stop Service", variant="stop", scale=1)
            with gr.Row():
                endpoint_display = gr.Textbox(label="API Endpoint", interactive=False, scale=4)
                btn_copy_url = gr.Button("Copy Address", scale=1)
            status_output = gr.Textbox(label="System Logs", lines=2)

        with gr.TabItem("API Debugging"):
            def test_api(port, key, prompt, temp, max_tokens):
                if not prompt: return "Please enter test prompt", ""
                try:
                    res = requests.post(
                        f"http://127.0.0.1:{int(port)}/v1/chat/completions",
                        json={
                            "model": "debug",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temp,
                            "max_tokens": max_tokens,
                            "stream": False
                        },
                        headers={"Authorization": f"Bearer {key}"}, timeout=300
                    )
                    if res.status_code == 200:
                        return f"Response:\n{res.json()['choices'][0]['message']['content']}", json.dumps(res.json(),
                                                                                                      indent=2,
                                                                                                      ensure_ascii=False)
                    return f"Error {res.status_code}:\n{res.text}", ""
                except Exception as e:
                    return f"Connection failed: {e}", ""


            with gr.Row():
                db_prompt = gr.TextArea(label="Prompt", placeholder="Enter text...", lines=5, scale=4)
                with gr.Column(scale=1):
                    db_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.7)
                    db_max_tokens = gr.Number(label="Max Output Tokens", value=2048, precision=0)
                    db_btn = gr.Button("Send Request", variant="primary")
            with gr.Row():
                db_res = gr.Textbox(label="Response Content")
                db_json = gr.Code(label="JSON", language="json")

            db_btn.click(test_api, inputs=[port_input, current_key_value, db_prompt, db_temp, db_max_tokens],
                         outputs=[db_res, db_json])

    btn_refresh_hw.click(refresh_hardware_info, outputs=[hardware_display, vram_free_state])
    model_dropdown.change(on_model_change, inputs=[model_dropdown, vram_free_state], outputs=[vram_status_bar])
    btn_scan.click(on_scan_click, outputs=[model_dropdown, vram_status_bar])
    btn_gen_key.click(generate_secure_key, inputs=[model_dropdown], outputs=[current_key_value, status_output]).then(
        lambda k: k, inputs=[current_key_value], outputs=[key_display])

    btn_vis.click(toggle_key_visibility, inputs=[key_visible_state], outputs=[key_visible_state, key_display])

    btn_copy.click(None, inputs=[key_display], js=js_copy)
    btn_start.click(launch_service, inputs=[model_dropdown, ctx_size_input, port_input, current_key_value],
                    outputs=[status_output, endpoint_display])
    btn_stop.click(terminate_service, outputs=[status_output, endpoint_display])
    btn_copy_url.click(None, inputs=[endpoint_display], js=js_copy)

if __name__ == "__main__":
    demo.launch(inbrowser=True)