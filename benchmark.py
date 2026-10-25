#!/usr/bin/env python3
"""Benchmark discovered local llama-server model profiles.

The local deployment runs one model per llama-server process. This benchmark
therefore switches the canonical user service before each model run unless
--no-switch is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:11434").rstrip("/")
HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(
    os.environ.get(
        "BENCHMARK_RESULTS_DIR",
        Path.home() / ".hermes" / "benchmarks" / "llama-server",
    )
)
REGISTRY_PATH = Path(os.environ.get("LLAMA_BROKER_REGISTRY", HERE / "generated/models.json"))


def load_models() -> dict[str, dict[str, str]]:
    try:
        records = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("models", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(record["slug"]): {
            "alias": str(record.get("alias", record["slug"])),
            "label": str(record.get("name", record["slug"])),
        }
        for record in records
    }


MODELS: dict[str, dict[str, str]] = {}

TESTS = [
    {
        "name": "safe_divide",
        "prompt": (
            "Write a Python function named safe_divide(a: float, b: float) -> "
            "float | None. Return None on division by zero. Output only code."
        ),
        "max_tokens": 180,
        "temperature": 0.2,
    },
    {
        "name": "tool_json",
        "prompt": (
            'Return JSON for tool_calls: get_weather(city="Boston") and '
            'get_stock(symbol="AAPL"). Use shape {"tool_calls":[{"name":"...",'
            '"arguments":{}}]}.'
        ),
        "max_tokens": 180,
        "temperature": 0.0,
        "json_object": True,
    },
    {
        "name": "bug_fix",
        "prompt": (
            "Fix this Python function and explain the change in two sentences:\n"
            "def divide(a, b):\n"
            "    return a / b\n"
        ),
        "max_tokens": 220,
        "temperature": 0.2,
    },
    {
        "name": "argument_extraction",
        "prompt": (
            "Extract function calls from these requests and return JSON only: "
            "1. Get the current price of Apple stock. "
            "2. Check if it is raining in Seattle. "
            "3. Search for Python machine learning libraries."
        ),
        "max_tokens": 220,
        "temperature": 0.0,
        "json_object": True,
    },
]


@dataclass
class Result:
    model_key: str
    model_alias: str
    test_name: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    completion_tokens_per_second: float | None
    success: bool
    json_valid: bool | None
    error: str | None
    response_preview: str


def http_json(method: str, path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def loaded_model() -> str | None:
    try:
        data = http_json("GET", "/v1/models", timeout=10)
    except Exception:
        return None
    entries = data.get("data") or []
    if not entries:
        return None
    return entries[0].get("id")


def switch_model(model_key: str) -> None:
    subprocess.run(
        [str(HERE / "manage-services.sh"), "switch", model_key],
        check=True,
    )


def run_chat(model_alias: str, test: dict) -> tuple[str, float, dict]:
    payload = {
        "model": model_alias,
        "messages": [{"role": "user", "content": test["prompt"]}],
        "max_tokens": test["max_tokens"],
        "temperature": test["temperature"],
        "stream": False,
    }
    if test.get("json_object"):
        payload["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    data = http_json("POST", "/v1/chat/completions", payload, timeout=240)
    elapsed = time.perf_counter() - start
    content = data["choices"][0]["message"]["content"]
    return content, elapsed, data.get("usage", {})


def benchmark_model(model_key: str, no_switch: bool) -> list[Result]:
    alias = MODELS[model_key]["alias"]
    if not no_switch:
        switch_model(model_key)

    active = loaded_model()
    if active != alias:
        raise RuntimeError(f"expected loaded model {alias}, got {active}")

    results: list[Result] = []
    for test in TESTS:
        json_valid = None
        error = None
        success = True
        content = ""
        elapsed = 0.0
        usage: dict = {}
        try:
            content, elapsed, usage = run_chat(alias, test)
            if test.get("json_object"):
                try:
                    json.loads(content)
                    json_valid = True
                except json.JSONDecodeError as exc:
                    json_valid = False
                    success = False
                    error = f"invalid json: {exc}"
        except urllib.error.HTTPError as exc:
            # Preserve the upstream/bridge body; status alone cannot establish
            # whether a failure was a length stop, validator rejection, or
            # transport error.
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:
                detail = "<unreadable HTTP error body>"
            success = False
            error = f"HTTP {exc.code} {exc.reason}: {detail}"
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError) as exc:
            success = False
            error = repr(exc)

        completion_tokens = usage.get("completion_tokens")
        tps = None
        if completion_tokens is not None and elapsed > 0:
            tps = completion_tokens / elapsed

        results.append(
            Result(
                model_key=model_key,
                model_alias=alias,
                test_name=test["name"],
                elapsed_seconds=elapsed,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                completion_tokens_per_second=tps,
                success=success,
                json_valid=json_valid,
                error=error,
                response_preview=content[:500],
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated discovered model slugs. Default: all local models.",
    )
    parser.add_argument(
        "--no-switch",
        action="store_true",
        help="Do not switch the systemd service; only test the currently loaded model.",
    )
    return parser.parse_args()


def main() -> int:
    global MODELS
    subprocess.run([str(HERE / "manage-services.sh"), "sync"], check=True)
    MODELS = load_models()
    if not MODELS:
        raise SystemExit("no GGUF models found; place models under ~/.local/share/models")
    args = parse_args()
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()] or list(MODELS)
    unknown = [key for key in model_keys if key not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model key(s): {', '.join(unknown)}")

    all_results: list[Result] = []
    for model_key in model_keys:
        print(f"\n=== {MODELS[model_key]['label']} ({MODELS[model_key]['alias']}) ===")
        results = benchmark_model(model_key, args.no_switch)
        all_results.extend(results)
        for result in results:
            status = "ok" if result.success else "fail"
            tps = "n/a" if result.completion_tokens_per_second is None else f"{result.completion_tokens_per_second:.2f}"
            print(
                f"{result.test_name}: {status}, "
                f"{result.elapsed_seconds:.2f}s, "
                f"{result.completion_tokens} completion tokens, {tps} tok/s"
            )
            if result.error:
                print(f"  error: {result.error}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "results": [asdict(result) for result in all_results],
    }
    path = RESULTS_DIR / f"benchmark-{stamp}.json"
    latest = RESULTS_DIR / "benchmark-latest.json"
    path.write_text(json.dumps(output, indent=2))
    latest.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {path}")
    print(f"Saved: {latest}")

    return 0 if all(result.success for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
