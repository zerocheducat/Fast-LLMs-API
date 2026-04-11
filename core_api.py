# -*- coding: utf-8 -*-
import os
import uvicorn
import requests
import torch
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional

# build by https://github.com/zerocheducat
try:
    from llama_cpp import Llama
    import llama_cpp
    import ctypes

    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

MODEL_PATH = os.getenv("LLM_MODEL_PATH")
API_KEY = os.getenv("LLM_API_KEY")
PORT = int(os.getenv("LLM_PORT", 8000))
OLLAMA_HOST = os.getenv("LLM_OLLAMA_HOST", "http://localhost:11434")

# Register low-level callback to safely silence C++ engine output
if HAS_LLAMA_CPP:
    # Use ctypes to wrap the Python function as a C callback pointer to resolve TypeError
    @llama_cpp.llama_log_callback
    def _mute_log_callback(level, message, user_data):
        pass


    llama_cpp.llama_log_set(_mute_log_callback, ctypes.c_void_p(None))


class ModelEngine:
    def __init__(self):
        self.mode = None
        self.engine = None
        self.tokenizer = None

    def load_model(self, source: str):
        if not source:
            return

        source = os.path.abspath(os.path.normpath(source.strip()))
        model_name = os.path.basename(source)
        gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print("=" * 60)
        print(f"[System] Initializing local LLM service...")
        print(f"[System] Hardware environment: {gpu_info}")
        print(f"[System] Mounting model: {model_name}")
        print("=" * 60)

        if os.path.isfile(source) and source.lower().endswith(".gguf"):
            if not HAS_LLAMA_CPP:
                raise ImportError("Missing dependency: llama-cpp-python")
            self.mode = "gguf"

            self.engine = Llama(
                model_path=source,
                n_gpu_layers=-1,
                n_ctx=4096,
                n_batch=1024,
                flash_attn=False,
                verbose=False
            )
            print(f"[System] GGUF engine started successfully (GPU acceleration enabled)")

        elif os.path.isdir(source) and os.path.exists(os.path.join(source, "config.json")):
            self.mode = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            self.engine = AutoModelForCausalLM.from_pretrained(
                source, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            print(f"[System] Transformers engine started successfully")

        else:
            self.mode = "ollama"
            print(f"[System] Ollama proxy mode ready")

    def generate_response(self, messages, temperature=0.7, max_tokens=2048):
        cleaned_messages = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')

            if isinstance(content, list):
                content = "".join([i.get("text", "") for i in content if i.get("type") == "text"])

            cleaned_messages.append({"role": role, "content": str(content)})

        if self.mode == "gguf":
            return self.engine.create_chat_completion(
                messages=cleaned_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        elif self.mode == "transformers":
            prompt = ""
            for m in cleaned_messages:
                prompt += f"{m['role']}: {m['content']}\n"
            prompt += "assistant: "

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.engine.device)
            outputs = self.engine.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_resp = resp.replace(prompt, "").strip()
            return {"choices": [{"message": {"role": "assistant", "content": final_resp}}]}

        elif self.mode == "ollama":
            try:
                res = requests.post(f"{OLLAMA_HOST}/api/chat", json={
                    "model": MODEL_PATH,
                    "messages": cleaned_messages,
                    "stream": False
                })
                data = res.json()
                return {"choices": [{"message": data.get("message", {})}]}
            except Exception as e:
                raise HTTPException(500, f"Ollama Connection Error: {str(e)}")


engine_instance = ModelEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH:
        engine_instance.load_model(MODEL_PATH)
    yield
    print("[System] Service is shutting down, cleaning up VRAM and memory resources...")
    if engine_instance.engine:
        del engine_instance.engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Local LLM Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class ChatRequest(BaseModel):
    messages: List[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


def verify_token(token: str = Depends(oauth2_scheme)):
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return token


@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatRequest, token: str = Depends(verify_token)):
    try:
        return engine_instance.generate_response(req.messages, req.temperature, req.max_tokens)
    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)