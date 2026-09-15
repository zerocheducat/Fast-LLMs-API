# -*- coding: utf-8 -*-
import copy
import gc
import inspect
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAS_LLAMA_CPP = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False


MODEL_PATH = (os.getenv("LLM_MODEL_PATH") or "").strip()
MODEL_ID = (os.getenv("LLM_MODEL_ID") or "local-model").strip() or "local-model"
API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
PORT = int(os.getenv("LLM_PORT", "8000"))
OLLAMA_HOST = (os.getenv("LLM_OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", "4096"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
BACKEND_HINT = (os.getenv("LLM_BACKEND") or "auto").strip().lower()
REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
SERVER_CREATED = int(time.time())

if CTX_SIZE <= 0:
    raise RuntimeError("LLM_CTX_SIZE must be a positive integer")
if MAX_OUTPUT_TOKENS <= 0:
    raise RuntimeError("LLM_MAX_OUTPUT_TOKENS must be a positive integer")
if MAX_OUTPUT_TOKENS > CTX_SIZE:
    raise RuntimeError("LLM_MAX_OUTPUT_TOKENS cannot exceed LLM_CTX_SIZE")


# ---------------------------------------------------------------------------
# 思考内容过滤
#
# 不同模型的思考标记不同，这里统一处理：
#   - Qwen 系列:        <think>...</think>，或只出现 </think> 闭合标记
#   - gpt-oss 系列:     <|channel|>analysis<|message|>...<|end|> 后跟 final 内容
#   - 其它带 channel 的: <|channel|>thought<|message|>...<|end|>
#
# 需要支持新格式时，只需往下面的列表里加正则/标记即可，其它逻辑不用改。
# ---------------------------------------------------------------------------

_THINK_BLOCK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>thought<\|message\|>.*?<\|end\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>commentary<\|message\|>.*?<\|end\|>", re.DOTALL | re.IGNORECASE),
]

_FINAL_MARKERS = [
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
]

_END_MARKERS = ["</think>", "<|end|>"]

# 流式过滤时，最多暂存这么多字符。超过则判断为“没有思考内容”，直接透传。
_STREAM_MAX_DISCARD = 200


def strip_thinking(text: Optional[str]) -> Optional[str]:
    """移除模型输出中的思考/分析内容，只保留最终回答。"""
    if not text:
        return text

    # 1. 移除完整的思考块
    for pattern in _THINK_BLOCK_PATTERNS:
        text = pattern.sub("", text)

    # 2. 处理只有闭合标记的情况（如 Qwen 的 </think>）
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 3. 处理 gpt-oss 的 final 标记：只保留 final 之后的内容
    for marker in _FINAL_MARKERS:
        if marker in text:
            text = text.split(marker)[-1]

    # 4. 清理残留的特殊标记
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()


class ThinkingStreamFilter:
    """流式输出时逐块过滤思考内容。

    设计：在遇到结束标记（</think> 或 <|end|>）之前，先暂存文本；
    一旦遇到结束标记，就从标记之后开始输出。若暂存超过
    _STREAM_MAX_DISCARD 仍未见结束标记，则判断该模型没有思考内容，
    把缓冲直接透传。
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.started = False

    def feed(self, text: str) -> str:
        if self.started:
            return text

        self.buffer += text

        for marker in _END_MARKERS:
            idx = self.buffer.find(marker)
            if idx == -1:
                continue
            after = self.buffer[idx + len(marker):]
            for fm in _FINAL_MARKERS:
                fidx = after.find(fm)
                if fidx != -1:
                    after = after[fidx + len(fm):]
                    break
            self.buffer = ""
            self.started = True
            return after

        if len(self.buffer) > _STREAM_MAX_DISCARD:
            self.started = True
            result = self.buffer
            self.buffer = ""
            return result

        return ""

    def flush(self) -> str:
        if self.started:
            return ""
        result = strip_thinking(self.buffer)
        self.buffer = ""
        self.started = True
        return result


def _filter_response_message(response: Any) -> Any:
    """对非流式响应中的 message.content 做思考内容过滤。"""
    if not isinstance(response, dict):
        return response
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = strip_thinking(message["content"])
    return response


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_model_field(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["model"] = MODEL_ID
    return payload


def _ensure_openai_response_shape(response: Dict[str, Any]) -> Dict[str, Any]:
    response = copy.deepcopy(response)
    response.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    response.setdefault("object", "chat.completion")
    response.setdefault("created", int(time.time()))
    response["model"] = MODEL_ID
    response.setdefault("choices", [])
    response.setdefault(
        "usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    return response


def _normalize_stream_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    chunk = copy.deepcopy(chunk)
    chunk.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    chunk.setdefault("object", "chat.completion.chunk")
    chunk.setdefault("created", int(time.time()))
    chunk["model"] = MODEL_ID
    chunk.setdefault("choices", [])
    return chunk


def _tool_choice_is_required(tool_choice: Any) -> bool:
    if tool_choice == "required":
        return True
    return isinstance(tool_choice, dict)


def _apply_stop_text(text: Optional[str], stop: Any) -> Tuple[Optional[str], bool]:
    if text is None or not stop:
        return text, False

    stops: List[str]
    if isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, list):
        stops = [item for item in stop if isinstance(item, str) and item]
    else:
        return text, False

    positions = [text.find(item) for item in stops]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return text, False

    cut = min(positions)
    return text[:cut], True


class ModelEngine:
    def __init__(self) -> None:
        self.mode: Optional[str] = None
        self.engine: Any = None
        self.tokenizer: Any = None
        self.source: str = ""
        self.ollama_model: str = ""

    def load_model(self, source: str) -> None:
        if not source:
            raise RuntimeError("LLM_MODEL_PATH is empty. Start the service through launcher.py or set the environment variable.")

        self.source = source.strip()

        if BACKEND_HINT == "ollama":
            self.mode = "ollama"
            self.ollama_model = self.source
            print(f"[System] Mode: Ollama OpenAI Proxy -> {self.ollama_model}")
            return

        normalized_path = os.path.abspath(os.path.normpath(self.source))
        print(f"[System] Initializing model source: {normalized_path}")

        if os.path.isfile(normalized_path) and normalized_path.lower().endswith(".gguf"):
            if not HAS_LLAMA_CPP:
                raise ImportError("Missing dependency: llama-cpp-python")
            self.mode = "gguf"
            print(f"[System] Mode: GGUF | context={CTX_SIZE} | GPU layers=-1")
            self.engine = Llama(
                model_path=normalized_path,
                n_gpu_layers=-1,
                n_ctx=CTX_SIZE,
                main_gpu=0,
                verbose=False,
            )
            return

        if os.path.isdir(normalized_path) and os.path.exists(os.path.join(normalized_path, "config.json")):
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                raise ImportError("Transformers backend requires both transformers and torch")
            self.mode = "transformers"
            print("[System] Mode: Transformers (CUDA + native chat template)")
            self.tokenizer = AutoTokenizer.from_pretrained(normalized_path, trust_remote_code=True)
            self.engine = AutoModelForCausalLM.from_pretrained(
                normalized_path,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            parser_state = "available" if callable(getattr(self.tokenizer, "parse_response", None)) else "unavailable"
            print(f"[System] Transformers response parser: {parser_state}")
            return

        if BACKEND_HINT not in ("auto", ""):
            raise RuntimeError(f"Unsupported LLM_BACKEND value: {BACKEND_HINT}")

        self.mode = "ollama"
        self.ollama_model = self.source
        print(f"[System] Mode: Ollama OpenAI Proxy -> {self.ollama_model}")

    def _check_llama_cpp_call_support(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        signature = inspect.signature(self.engine.create_chat_completion)
        params = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return kwargs

        unsupported = [key for key in kwargs if key not in params]
        if not unsupported:
            return kwargs

        critical = [key for key in unsupported if key in {"tools", "tool_choice"} and kwargs.get(key) is not None]
        if critical:
            raise RuntimeError(
                "Installed llama-cpp-python does not expose required tool-calling parameters: "
                + ", ".join(critical)
                + ". Upgrade llama-cpp-python before using an agent harness."
            )

        for key in unsupported:
            kwargs.pop(key, None)
        return kwargs

    def _generate_gguf(self, payload: Dict[str, Any]) -> Tuple[Any, bool]:
        kwargs: Dict[str, Any] = {
            "messages": payload["messages"],
            "temperature": payload.get("temperature", 0.7),
            "max_tokens": payload.get("max_tokens", MAX_OUTPUT_TOKENS),
            "stream": bool(payload.get("stream", False)),
            "model": MODEL_ID,
        }

        optional_keys = (
            "tools",
            "tool_choice",
            "top_p",
            "stop",
            "seed",
            "response_format",
            "presence_penalty",
            "frequency_penalty",
        )
        for key in optional_keys:
            if key in payload and payload[key] is not None:
                kwargs[key] = payload[key]

        kwargs = self._check_llama_cpp_call_support(kwargs)
        result = self.engine.create_chat_completion(**kwargs)

        if payload.get("stream"):
            def chunk_iter() -> Iterator[Dict[str, Any]]:
                for chunk in result:
                    if isinstance(chunk, dict):
                        yield _normalize_stream_chunk(chunk)
            return chunk_iter(), True

        if not isinstance(result, dict):
            raise RuntimeError("llama-cpp-python returned a non-dictionary chat completion response")
        return _ensure_openai_response_shape(result), False

    def _openai_messages_to_transformers(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = copy.deepcopy(messages)
        tool_id_to_name: Dict[str, str] = {}

        for message in converted:
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                call_id = tool_call.get("id")
                name = function.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    tool_id_to_name[call_id] = name

                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Tool call arguments for {call_id or name or '<unknown>'} are not valid JSON and cannot be converted for Transformers chat templates"
                        ) from exc

        for message in converted:
            if message.get("role") != "tool":
                continue
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and "name" not in message:
                mapped_name = tool_id_to_name.get(tool_call_id)
                if mapped_name:
                    message["name"] = mapped_name

        return converted

    def _transformers_message_to_openai(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        message = copy.deepcopy(parsed)
        message["role"] = "assistant"
        message.setdefault("content", None)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
            message["tool_calls"] = tool_calls

        if isinstance(tool_calls, list):
            normalized_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                item = copy.deepcopy(tool_call)
                item.setdefault("id", f"call_{uuid.uuid4().hex}")
                item.setdefault("type", "function")
                function = item.get("function")
                if isinstance(function, dict):
                    arguments = function.get("arguments", {})
                    if not isinstance(arguments, str):
                        function["arguments"] = _json_dumps(arguments)
                normalized_calls.append(item)
            message["tool_calls"] = normalized_calls

        return message

    def _generate_transformers(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        messages = self._openai_messages_to_transformers(payload["messages"])
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")

        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            template_kwargs["tools"] = tools
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice

        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {key: value.to(self.engine.device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[1])

        seed = payload.get("seed")
        if isinstance(seed, int):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        max_tokens = int(payload.get("max_tokens", MAX_OUTPUT_TOKENS))
        temperature = payload.get("temperature", 0.7)
        do_sample = temperature is not None and float(temperature) > 0.0

        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            if payload.get("top_p") is not None:
                generation_kwargs["top_p"] = float(payload["top_p"])

        outputs = self.engine.generate(**generation_kwargs)
        generated_tokens = outputs[0][input_length:]
        completion_tokens = int(generated_tokens.shape[0])

        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        plain_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        parsed_message: Optional[Dict[str, Any]] = None
        parse_response = getattr(self.tokenizer, "parse_response", None)
        if callable(parse_response):
            try:
                parse_kwargs: Dict[str, Any] = {"prefix": inputs["input_ids"][0]}
                if tools is not None:
                    parse_kwargs["tools"] = tools
                parsed = parse_response(raw_text, **parse_kwargs)
                if isinstance(parsed, dict):
                    parsed_message = self._transformers_message_to_openai(parsed)
            except Exception as exc:
                if tools is not None:
                    print(f"[Warning] Transformers response parser could not parse structured output: {exc}")

        if parsed_message is None:
            if tools is not None and _tool_choice_is_required(tool_choice):
                raise RuntimeError(
                    "This Transformers tokenizer cannot return a structured tool call for the current model/output. "
                    "Use a model with a Transformers response_template/parse_response implementation, or run the model through GGUF/llama-cpp-python or Ollama."
                )
            parsed_message = {"role": "assistant", "content": plain_text}

        stopped = False
        if isinstance(parsed_message.get("content"), str):
            parsed_message["content"], stopped = _apply_stop_text(parsed_message["content"], payload.get("stop"))

        has_tool_calls = bool(parsed_message.get("tool_calls"))
        if has_tool_calls:
            finish_reason = "tool_calls"
        elif completion_tokens >= max_tokens and not stopped:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": parsed_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
        }
        return response, False

    def _ollama_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_choice = payload.get("tool_choice")
        if _tool_choice_is_required(tool_choice):
            raise RuntimeError(
                "The current Ollama OpenAI-compatibility documentation does not define required/named tool_choice handling. "
                "Use tool_choice='auto' or omit tool_choice for this backend."
            )

        allowed = (
            "messages",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "max_tokens",
            "reasoning_effort",
            "reasoning",
        )
        forwarded: Dict[str, Any] = {"model": self.ollama_model}
        for key in allowed:
            if key in payload and payload[key] is not None:
                forwarded[key] = payload[key]

        if tool_choice != "none" and payload.get("tools") is not None:
            forwarded["tools"] = payload["tools"]

        return forwarded

    def _generate_ollama(self, payload: Dict[str, Any]) -> Tuple[Any, bool]:
        url = f"{OLLAMA_HOST}/v1/chat/completions"
        forwarded = self._ollama_payload(payload)
        stream = bool(payload.get("stream", False))

        try:
            response = requests.post(
                url,
                json=forwarded,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama connection error: {exc}") from exc

        if response.status_code >= 400:
            body = response.text
            response.close()
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {body}")

        if stream:
            def chunk_iter() -> Iterator[Dict[str, Any]]:
                try:
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if isinstance(chunk, dict):
                            yield _normalize_stream_chunk(chunk)
                finally:
                    response.close()
            return chunk_iter(), True

        try:
            data = response.json()
        finally:
            response.close()
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned a non-dictionary OpenAI chat completion response")
        return _ensure_openai_response_shape(data), False

    def generate_response(self, payload: Dict[str, Any]) -> Tuple[Any, bool]:
        if self.mode == "gguf":
            return self._generate_gguf(payload)
        if self.mode == "transformers":
            return self._generate_transformers(payload)
        if self.mode == "ollama":
            return self._generate_ollama(payload)
        raise RuntimeError("No model backend is loaded")


engine_instance = ModelEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_instance.load_model(MODEL_PATH)
    yield
    print("[System] Shutting down, releasing resources...")
    if engine_instance.engine is not None:
        del engine_instance.engine
        engine_instance.engine = None
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Fast LLMs API", version="2.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured")
    if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials


def _request_summary(payload: Any) -> str:
    if not isinstance(payload, dict):
        return f"body_type={type(payload).__name__}"

    messages = payload.get("messages")
    tools = payload.get("tools")
    return (
        f"model={payload.get('model')!r} | "
        f"messages={len(messages) if isinstance(messages, list) else 'invalid'} | "
        f"tools={len(tools) if isinstance(tools, list) else (0 if tools is None else 'invalid')} | "
        f"stream={bool(payload.get('stream', False))} | "
        f"max_tokens={payload.get('max_tokens')!r} | "
        f"max_completion_tokens={payload.get('max_completion_tokens')!r}"
    )


def _validate_chat_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")
    if not all(isinstance(message, dict) for message in messages):
        raise HTTPException(status_code=400, detail="Every messages item must be a JSON object")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model must be a string")

    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="tools must be an array when provided")

    legacy_limit = payload.get("max_tokens")
    completion_limit = payload.get("max_completion_tokens")
    requested_limit = completion_limit if completion_limit is not None else legacy_limit
    if requested_limit is None:
        requested_limit = MAX_OUTPUT_TOKENS
    if not isinstance(requested_limit, int) or isinstance(requested_limit, bool) or requested_limit <= 0:
        raise HTTPException(
            status_code=400,
            detail="max_completion_tokens/max_tokens must be a positive integer",
        )

    # Treat client-side token limits as requests, not routing constraints. A local
    # single-model gateway can safely cap a larger client request to the model's
    # configured output limit instead of rejecting the whole agent turn.
    max_tokens = min(requested_limit, MAX_OUTPUT_TOKENS, max(1, CTX_SIZE - 1))
    if requested_limit != max_tokens:
        print(
            f"[Request] Output limit capped: requested={requested_limit} -> effective={max_tokens} "
            f"(configured_max={MAX_OUTPUT_TOKENS}, context={CTX_SIZE})"
        )
    if legacy_limit is not None and completion_limit is not None and legacy_limit != completion_limit:
        print(
            f"[Request] Both token-limit fields received; using max_completion_tokens={completion_limit} "
            f"and ignoring max_tokens={legacy_limit}"
        )

    normalized = copy.deepcopy(payload)
    # This service exposes one loaded model. The incoming model value is treated
    # as a client-side alias; responses consistently advertise MODEL_ID.
    normalized["model"] = MODEL_ID
    normalized["max_tokens"] = max_tokens
    normalized.pop("max_completion_tokens", None)
    normalized["stream"] = bool(payload.get("stream", False))
    if "temperature" not in normalized or normalized["temperature"] is None:
        normalized["temperature"] = 0.7
    return normalized


def _sse_from_chunks(chunks: Iterable[Dict[str, Any]]) -> Iterator[str]:
    thinker = ThinkingStreamFilter()
    for chunk in chunks:
        chunk = _normalize_stream_chunk(chunk)
        choices = chunk.get("choices", [])
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                if isinstance(delta.get("content"), str):
                    filtered = thinker.feed(delta["content"])
                    if filtered:
                        delta["content"] = filtered
                    else:
                        delta.pop("content", None)
        yield f"data: {_json_dumps(chunk)}\n\n"

    remaining = thinker.flush()
    if remaining:
        tail_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": remaining},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {_json_dumps(tail_chunk)}\n\n"

    yield "data: [DONE]\n\n"


def _sse_from_complete_response(response: Dict[str, Any], stream_options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
    response = _ensure_openai_response_shape(response)
    choice = response.get("choices", [{}])[0] if response.get("choices") else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    finish_reason = choice.get("finish_reason", "stop") if isinstance(choice, dict) else "stop"
    chunk_id = response["id"]
    created = response["created"]

    yield f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content:
        content = strip_thinking(content)
        if content:
            yield f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list) and tool_calls:
        deltas = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            deltas.append(
                {
                    "index": index,
                    "id": tool_call.get("id", f"call_{uuid.uuid4().hex}"),
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments", ""),
                    },
                }
            )
        if deltas:
            yield f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'tool_calls': deltas}, 'finish_reason': None}]})}\n\n"

    yield f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"

    if isinstance(stream_options, dict) and stream_options.get("include_usage"):
        yield f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': response.get('usage', {})})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "name": MODEL_ID,
                "object": "model",
                "created": SERVER_CREATED,
                "owned_by": "fast-llms-api",
                "context_window": CTX_SIZE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        ],
    }


@app.get("/health")
async def health(token: str = Depends(verify_token)):
    return {
        "status": "ok",
        "backend": engine_instance.mode,
        "model": MODEL_ID,
        "context_size": CTX_SIZE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request, token: str = Depends(verify_token)):
    raw_payload: Any = None
    try:
        raw_payload = await request.json()
        print(f"[Request] Chat Completions | {_request_summary(raw_payload)}")
        payload = _validate_chat_payload(raw_payload)
        result, native_stream = engine_instance.generate_response(payload)

        if payload.get("stream"):
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if native_stream:
                return StreamingResponse(
                    _sse_from_chunks(result),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return StreamingResponse(
                _sse_from_complete_response(result, payload.get("stream_options")),
                media_type="text/event-stream",
                headers=headers,
            )

        # 非流式响应统一过滤思考内容
        return _filter_response_message(result)

    except HTTPException as exc:
        print(f"[RequestRejected] status={exc.status_code} detail={exc.detail} | {_request_summary(raw_payload)}")
        raise
    except json.JSONDecodeError:
        print("[RequestRejected] status=400 detail=Request body is not valid JSON")
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")
    except Exception as exc:
        print(f"[Error] Generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)