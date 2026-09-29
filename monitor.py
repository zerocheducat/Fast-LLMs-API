# -*- coding: utf-8 -*-
"""
Fast LLMs API - terminal performance monitor

No third-party dependency is required.

The launcher passes FAST_LLMS_BASE_URL and FAST_LLMS_API_KEY automatically.
Manual use in PowerShell:

    $env:FAST_LLMS_BASE_URL="http://127.0.0.1:7021"
    $env:FAST_LLMS_API_KEY="your-key"
    python monitor.py
"""

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = os.getenv(
    "FAST_LLMS_BASE_URL",
    "http://127.0.0.1:7021",
).rstrip("/")

DEFAULT_API_KEY = os.getenv(
    "FAST_LLMS_API_KEY",
    "",
).strip()

DEFAULT_INTERVAL = 0.5


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def enable_vt_mode() -> None:
    if os.name != "nt":
        return

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def set_console_title(title: str) -> None:
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleTitleW(title)
            return
        except Exception:
            pass

    sys.stdout.write(f"\033]0;{title}\007")


class TerminalScreen:
    """Render the dashboard in place without growing terminal scrollback."""

    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        enable_vt_mode()

        # Alternate screen buffer:
        # - keeps the normal terminal history untouched
        # - prevents every refresh from adding a full dashboard to scrollback
        # - hides the cursor while the live monitor is running
        sys.stdout.write("\033[?1049h")
        sys.stdout.write("\033[?25l")
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        self.active = True
        return self

    def draw(self, frame: str) -> None:
        # Move to the top-left and overwrite the previous frame in place.
        # Erase everything below the new frame so shrinking/resizing does not
        # leave stale characters behind.
        sys.stdout.write("\033[H")
        sys.stdout.write(frame)
        sys.stdout.write("\033[J")
        sys.stdout.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.active:
            return

        # Restore the user's original terminal buffer and cursor.
        sys.stdout.write(C.RESET)
        sys.stdout.write("\033[?25h")
        sys.stdout.write("\033[?1049l")
        sys.stdout.flush()

        self.active = False


def visible_width(text: str) -> int:
    # The monitor only aligns ASCII labels/model IDs. ANSI escapes are not
    # passed here, so len() is adequate for this dashboard.
    return len(text)


def fit(text: Any, width: int) -> str:
    value = str(text)
    if width <= 1:
        return value[:width]
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return str(value)

    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000.0:.2f} s"


def fmt_bytes_mb(value: Any) -> str:
    if value is None:
        return "-"
    try:
        mb = float(value)
    except (TypeError, ValueError):
        return str(value)

    if mb >= 1024:
        return f"{mb / 1024.0:.2f} GB"
    return f"{mb:.0f} MB"


def fmt_duration(seconds: Any) -> str:
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "-"

    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def pct_color(value: float) -> str:
    if value >= 90:
        return C.RED
    if value >= 75:
        return C.YELLOW
    return C.GREEN


def bar(value: Any, maximum: Any, width: int = 24) -> str:
    try:
        current = max(0.0, float(value))
        total = max(0.000001, float(maximum))
        ratio = min(1.0, current / total)
    except (TypeError, ValueError):
        return "░" * width

    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def sparkline(values: List[float], width: int = 24) -> str:
    blocks = "▁▂▃▄▅▆▇█"

    clean = []
    for value in values[-width:]:
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            pass

    if not clean:
        return "·" * min(width, 12)

    lo = min(clean)
    hi = max(clean)

    if hi <= lo:
        return blocks[4] * len(clean)

    chars = []
    for value in clean:
        ratio = (value - lo) / (hi - lo)
        index = min(len(blocks) - 1, int(round(ratio * (len(blocks) - 1))))
        chars.append(blocks[index])
    return "".join(chars)


def get_system_memory() -> Dict[str, float]:
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                total_mb = state.ullTotalPhys / (1024 * 1024)
                available_mb = state.ullAvailPhys / (1024 * 1024)
                return {
                    "total_mb": round(total_mb, 1),
                    "used_mb": round(total_mb - available_mb, 1),
                    "load_pct": float(state.dwMemoryLoad),
                }
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        try:
            values: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as file:
                for line in file:
                    key, rest = line.split(":", 1)
                    values[key] = int(rest.strip().split()[0])
            total_kb = values["MemTotal"]
            available_kb = values["MemAvailable"]
            used_kb = total_kb - available_kb
            return {
                "total_mb": round(total_kb / 1024.0, 1),
                "used_mb": round(used_kb / 1024.0, 1),
                "load_pct": round(used_kb * 100.0 / total_kb, 1),
            }
        except Exception:
            pass

    return {}


def query_nvidia() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]

    kwargs: Dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 3,
    }

    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        result = subprocess.run(command, **kwargs)
        if result.returncode != 0:
            return {}

        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            return {}

        return {
            "name": parts[0],
            "util_pct": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_total_mb": float(parts[3]),
            "temp_c": float(parts[4]),
            "power_w": float(parts[5]),
            "power_limit_w": float(parts[6]),
        }
    except Exception:
        return {}


class MetricsClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/metrics",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def row(label: str, value: str, label_width: int = 20) -> str:
    return f"{C.GRAY}{label:<{label_width}}{C.RESET}{value}"


def render(
    data: Dict[str, Any],
    gpu: Dict[str, Any],
    system_memory: Dict[str, Any],
    last_error: Optional[str] = None,
    interval: float = DEFAULT_INTERVAL,
) -> str:
    terminal_width = shutil.get_terminal_size((110, 34)).columns
    width = max(88, min(132, terminal_width))
    inner = width - 2

    server = data.get("server") if isinstance(data.get("server"), dict) else {}
    active = data.get("active_request") if isinstance(data.get("active_request"), dict) else None
    last = data.get("last_request") if isinstance(data.get("last_request"), dict) else None
    history = data.get("history") if isinstance(data.get("history"), list) else []

    current = active or last or {}
    state = "RUNNING" if active else ("IDLE" if last else "WAITING")
    state_color = C.GREEN if active else (C.CYAN if last else C.YELLOW)

    lines: List[str] = []

    title = " FAST LLMs API • Live Performance Monitor "
    left = max(1, (inner - len(title)) // 2)
    right = max(1, inner - len(title) - left)
    lines.append(C.CYAN + "╭" + "─" * left + C.BOLD + title + C.RESET + C.CYAN + "─" * right + "╮" + C.RESET)

    model = fit(server.get("model", current.get("model", "-")), max(28, inner - 46))
    header = (
        f" {state_color}● {state:<7}{C.RESET}"
        f"  {C.BOLD}{model}{C.RESET}"
        f"  backend={server.get('backend', current.get('backend', '-'))}"
        f"  uptime={fmt_duration(server.get('uptime_s'))}"
    )
    lines.append("│" + fit(header, inner + 40) + " " * max(0, inner - visible_width(re.sub(r'\x1b\[[0-9;]*m', '', header))) + "│")

    filter_mode = server.get("reasoning_filter", current.get("reasoning_filter", "-"))
    meta = (
        f" requests={data.get('requests_total', 0)}"
        f"  current={current.get('kind', '-')}"
        f"  messages={current.get('messages', '-')}"
        f"  tools={current.get('tools', '-')}"
        f"  reasoning-filter={filter_mode}"
        f"  pid={server.get('pid', '-')}"
    )
    lines.append("│" + fit(meta, inner) + " " * max(0, inner - len(fit(meta, inner))) + "│")
    lines.append("├" + "─" * inner + "┤")

    # Token/context section
    ctx = current.get("context_tokens")
    ctx_max = current.get("context_window", server.get("context_window"))
    ctx_pct = current.get("context_usage_pct")
    try:
        ctx_pct_f = float(ctx_pct or 0.0)
    except (TypeError, ValueError):
        ctx_pct_f = 0.0

    lines.append(f"│ {C.BOLD}{C.MAGENTA}TOKEN BUDGET{C.RESET}" + " " * max(0, inner - 13) + "│")
    context_bar = bar(ctx or 0, ctx_max or 1, 30)
    context_value = (
        f"{fmt_num(ctx)} / {fmt_num(ctx_max)}"
        f"  {pct_color(ctx_pct_f)}{fmt_num(ctx_pct_f)}%{C.RESET}"
    )
    context_plain = f"Context  [{context_bar}]  {fmt_num(ctx)} / {fmt_num(ctx_max)}  {fmt_num(ctx_pct_f)}%"
    context_colored = f" Context  [{C.BLUE}{context_bar}{C.RESET}]  {context_value}"
    lines.append("│" + context_colored + " " * max(0, inner - len(context_plain)) + "│")

    tokens_plain = (
        f" Prompt {fmt_num(current.get('prompt_tokens'))}"
        f"   Eval {fmt_num(current.get('prompt_eval_tokens'))}"
        f"   Reused {fmt_num(current.get('reused_prompt_tokens'))}"
        f"   Completion {fmt_num(current.get('completion_tokens'))}"
        f"   Total {fmt_num(current.get('total_tokens'))}"
    )
    lines.append("│" + fit(tokens_plain, inner) + " " * max(0, inner - len(fit(tokens_plain, inner))) + "│")
    lines.append("├" + "─" * inner + "┤")

    # Performance section
    pp = current.get("prompt_eval_tps")
    tg = current.get("generation_tps")
    raw_ttft = current.get("backend_ttft_ms")
    visible_ttft = current.get("visible_ttft_ms")
    elapsed = current.get("elapsed_ms", current.get("wall_time_ms"))

    lines.append(f"│ {C.BOLD}{C.GREEN}PERFORMANCE{C.RESET}" + " " * max(0, inner - 13) + "│")
    perf_line = (
        f" Prompt processing  {C.CYAN}{fmt_num(pp, 2)} tok/s{C.RESET}"
        f"      Generation  {C.GREEN}{fmt_num(tg, 2)} tok/s{C.RESET}"
    )
    perf_plain = re.sub(r"\x1b\[[0-9;]*m", "", perf_line)
    lines.append("│" + perf_line + " " * max(0, inner - len(perf_plain)) + "│")

    latency_plain = (
        f" Raw TTFT {fmt_ms(raw_ttft)}"
        f"   Visible TTFT {fmt_ms(visible_ttft)}"
        f"   Elapsed {fmt_ms(elapsed)}"
    )
    lines.append("│" + fit(latency_plain, inner) + " " * max(0, inner - len(fit(latency_plain, inner))) + "│")

    speeds = [
        item.get("generation_tps")
        for item in history
        if isinstance(item, dict) and item.get("generation_tps") is not None
    ]
    graph = sparkline(speeds, 36)
    graph_line = f" Generation history  {C.GREEN}{graph}{C.RESET}"
    graph_plain = re.sub(r"\x1b\[[0-9;]*m", "", graph_line)
    lines.append("│" + graph_line + " " * max(0, inner - len(graph_plain)) + "│")
    lines.append("├" + "─" * inner + "┤")

    # Cache section
    reuse_pct = current.get("kv_prefix_reuse_pct")
    try:
        reuse_f = float(reuse_pct or 0.0)
    except (TypeError, ValueError):
        reuse_f = 0.0

    lines.append(f"│ {C.BOLD}{C.YELLOW}CACHE / PREFIX REUSE{C.RESET}" + " " * max(0, inner - 22) + "│")
    reuse_bar = bar(reuse_f, 100.0, 30)
    reuse_plain = (
        f" KV prefix reuse  [{reuse_bar}]  {fmt_num(reuse_f)}%"
        f"   graph reuse={fmt_num(current.get('graph_reuse_count'))}"
    )
    reuse_colored = (
        f" KV prefix reuse  [{C.YELLOW}{reuse_bar}{C.RESET}]  "
        f"{fmt_num(reuse_f)}%   graph reuse={fmt_num(current.get('graph_reuse_count'))}"
    )
    lines.append("│" + reuse_colored + " " * max(0, inner - len(reuse_plain)) + "│")
    note = " Reuse is derived from llama.cpp context/perf counters; graph reuse is a separate compute-graph metric."
    lines.append("│" + C.DIM + fit(note, inner) + C.RESET + " " * max(0, inner - len(fit(note, inner))) + "│")
    lines.append("├" + "─" * inner + "┤")

    # Hardware section
    lines.append(f"│ {C.BOLD}{C.BLUE}HARDWARE{C.RESET}" + " " * max(0, inner - 10) + "│")

    if gpu:
        vram_used = gpu.get("vram_used_mb", 0)
        vram_total = gpu.get("vram_total_mb", 1)
        util = gpu.get("util_pct", 0)
        temp = gpu.get("temp_c")
        power = gpu.get("power_w")
        power_limit = gpu.get("power_limit_w")

        gpu_line = (
            f" GPU {fit(gpu.get('name', '-'), 28):<28}"
            f"  util {fmt_num(util)}%"
            f"  temp {fmt_num(temp)} C"
            f"  power {fmt_num(power)} / {fmt_num(power_limit)} W"
        )
        lines.append("│" + fit(gpu_line, inner) + " " * max(0, inner - len(fit(gpu_line, inner))) + "│")

        vram_line_plain = (
            f" VRAM [{bar(vram_used, vram_total, 30)}]  "
            f"{fmt_bytes_mb(vram_used)} / {fmt_bytes_mb(vram_total)}"
        )
        vram_line = (
            f" VRAM [{C.BLUE}{bar(vram_used, vram_total, 30)}{C.RESET}]  "
            f"{fmt_bytes_mb(vram_used)} / {fmt_bytes_mb(vram_total)}"
        )
        lines.append("│" + vram_line + " " * max(0, inner - len(vram_line_plain)) + "│")
    else:
        gpu_line = " GPU metrics unavailable (nvidia-smi not found or query failed)"
        lines.append("│" + C.DIM + fit(gpu_line, inner) + C.RESET + " " * max(0, inner - len(fit(gpu_line, inner))) + "│")

    process_ws = server.get("working_set_mb", current.get("working_set_mb"))
    process_private = server.get("private_mb", current.get("private_mb"))
    process_line = (
        f" API process RAM  working-set {fmt_bytes_mb(process_ws)}"
        f"   private {fmt_bytes_mb(process_private)}"
    )
    lines.append("│" + fit(process_line, inner) + " " * max(0, inner - len(fit(process_line, inner))) + "│")

    if system_memory:
        ram_used = system_memory.get("used_mb", 0)
        ram_total = system_memory.get("total_mb", 1)
        ram_pct = system_memory.get("load_pct", 0)
        ram_plain = (
            f" System RAM [{bar(ram_used, ram_total, 30)}]  "
            f"{fmt_bytes_mb(ram_used)} / {fmt_bytes_mb(ram_total)}  {fmt_num(ram_pct)}%"
        )
        ram_colored = (
            f" System RAM [{C.MAGENTA}{bar(ram_used, ram_total, 30)}{C.RESET}]  "
            f"{fmt_bytes_mb(ram_used)} / {fmt_bytes_mb(ram_total)}  {fmt_num(ram_pct)}%"
        )
        lines.append("│" + ram_colored + " " * max(0, inner - len(ram_plain)) + "│")

    lines.append("├" + "─" * inner + "┤")

    # Recent requests
    lines.append(f"│ {C.BOLD}RECENT REQUESTS{C.RESET}" + " " * max(0, inner - 16) + "│")
    recent = [item for item in history[-5:] if isinstance(item, dict)]
    if not recent:
        empty = " No completed requests yet."
        lines.append("│" + empty + " " * max(0, inner - len(empty)) + "│")
    else:
        header = " kind    msg  prompt  completion   PP tok/s   TG tok/s   reuse%   wall"
        lines.append("│" + C.GRAY + fit(header, inner) + C.RESET + " " * max(0, inner - len(fit(header, inner))) + "│")
        for item in recent:
            line = (
                f" {str(item.get('kind', '-')):<7}"
                f" {str(item.get('messages', '-')):>3}"
                f" {fmt_num(item.get('prompt_tokens')):>7}"
                f" {fmt_num(item.get('completion_tokens')):>11}"
                f" {fmt_num(item.get('prompt_eval_tps'), 1):>10}"
                f" {fmt_num(item.get('generation_tps'), 1):>10}"
                f" {fmt_num(item.get('kv_prefix_reuse_pct'), 1):>8}"
                f" {fmt_ms(item.get('wall_time_ms')):>8}"
            )
            lines.append("│" + fit(line, inner) + " " * max(0, inner - len(fit(line, inner))) + "│")

    if last_error:
        lines.append("├" + "─" * inner + "┤")
        err = f" {C.RED}Connection: {fit(last_error, max(10, inner - 14))}{C.RESET}"
        err_plain = re.sub(r"\x1b\[[0-9;]*m", "", err)
        lines.append("│" + err + " " * max(0, inner - len(err_plain)) + "│")

    lines.append("╰" + "─" * inner + "╯")
    footer = (
        f"{C.DIM}refresh {interval:g}s • Ctrl+C exit • "
        f"API {server.get('backend', '-')} • "
        f"token source: {current.get('token_source', '-')}{C.RESET}"
    )
    lines.append(footer)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast LLMs API terminal monitor")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="API base URL, without /v1")
    parser.add_argument("--key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Refresh interval in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = str(args.key or "").strip()
    base_url = str(args.url or DEFAULT_BASE_URL).rstrip("/")
    interval = max(0.2, float(args.interval))

    if not api_key:
        print("FAST_LLMS_API_KEY is empty.")
        print("Open the monitor from launcher.py, or set the environment variable first.")
        return 2

    enable_vt_mode()
    set_console_title("Fast LLMs API - Performance Monitor")

    client = MetricsClient(base_url, api_key)
    gpu_cache: Dict[str, Any] = {}
    gpu_last_update = 0.0
    latest_data: Dict[str, Any] = {}
    last_error: Optional[str] = None

    try:
        with TerminalScreen() as screen:
            while True:
                now = time.monotonic()

                try:
                    latest_data = client.fetch()
                    last_error = None
                except urllib.error.HTTPError as exc:
                    last_error = f"HTTP {exc.code} {exc.reason}"
                except urllib.error.URLError as exc:
                    last_error = str(exc.reason)
                except Exception as exc:
                    last_error = str(exc)

                if now - gpu_last_update >= 1.0:
                    gpu_cache = query_nvidia()
                    gpu_last_update = now

                memory = get_system_memory()

                frame = render(
                    latest_data,
                    gpu_cache,
                    memory,
                    last_error=last_error,
                    interval=interval,
                )

                screen.draw(frame)
                time.sleep(interval)

    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
