#!/usr/bin/env python3
"""One-request-at-a-time prompt ladder with portable GPU-fault gates."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_broker.broker_profiles import ProfileRegistry  # noqa: E402


MAX_TARGET_TOKENS = 262_000


def request_prompt(target_tokens: int, timeout: float, model: str) -> dict[str, object]:
    # " pad" is one token in the Qwen tokenizer after the first occurrence.
    # Leave room for the chat template and fixed instruction, then trust the
    # server-reported prompt_tokens as the measured value.
    repeats = max(1, target_tokens - 96)
    marker = f"SAFETY-LADDER-{target_tokens}"
    prompt = (
        f"Remember marker {marker}. "
        + (" pad" * repeats)
        + "\nReturn the marker from the beginning exactly, with no other text."
    )
    payload = {
        "model": model,
        "seed": 1,
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": "This is a bounded prompt-processing stability probe.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout)
    started = time.monotonic()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        raw = response.read()
        elapsed = time.monotonic() - started
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"target {target_tokens}: non-JSON HTTP {response.status}: {raw[:500]!r}"
            ) from error
        if response.status != 200:
            raise RuntimeError(
                f"target {target_tokens}: HTTP {response.status}: {decoded}"
            )
        usage = decoded.get("usage", {})
        timings = decoded.get("timings", {})
        choice = decoded.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        return {
            "target_tokens": target_tokens,
            "measured_prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "decode_tokens_per_second": timings.get("predicted_per_second"),
            "elapsed_seconds": round(elapsed, 3),
            "finish_reason": choice.get("finish_reason"),
            "marker_recalled": marker in content,
            "content_preview": content[:160],
            "reasoning_preview": reasoning[:160],
            "profile_header": response.getheader("X-Llama-Broker-Profile"),
        }
    finally:
        connection.close()


def windows_live_kernel_reports(start_iso: str) -> list[dict[str, str]]:
    """Optional WSL/Windows diagnostic.  Never used unless explicitly requested."""

    if shutil.which("powershell.exe") is None:
        raise RuntimeError("--windows-wer requested, but powershell.exe is unavailable")
    command = (
        "$s=[datetime]'" + start_iso + "';"
        "$f=Get-ChildItem 'C:\\WINDOWS\\LiveKernelReports' -Recurse -File "
        "-ErrorAction SilentlyContinue|Where-Object{$_.LastWriteTime -ge $s}|"
        "Select-Object FullName,@{n='LastWriteTime';e={$_.LastWriteTime.ToString('o')}};"
        "@($f)|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    raw = result.stdout.strip()
    if not raw:
        return []
    payload = json.loads(raw)
    records = payload if isinstance(payload, list) else [payload]
    return [
        {
            "source": "windows-wer",
            "message": str(record.get("FullName", record)),
            "time": str(record.get("LastWriteTime", "")),
        }
        for record in records
    ]


_GPU_FAULT_RE = re.compile(
    r"(?:NVRM:.*Xid|\bXid\b|GPU has fallen off|CUDA error|CUDA.*unknown error|"
    r"device-side assert|ggml_cuda.*failed|Out of memory: Killed process|"
    r"oom-kill:.*llama|Killed process \d+ \(llama-server\))",
    re.IGNORECASE,
)


def linux_gpu_faults(
    start_iso: str, profile_id: str = "example-model-7b"
) -> list[dict[str, str]]:
    """Collect portable Linux/kernel and backend-service fault evidence."""

    commands = [
        ("kernel", ["journalctl", "-k", "--since", start_iso, "--no-pager", "-o", "cat"]),
        (
            "llama-server",
            [
                "journalctl",
                "--user",
                "-u",
                f"llama-broker-profile-{profile_id}.service",
                "--since",
                start_iso,
                "--no-pager",
                "-o",
                "cat",
            ],
        ),
    ]
    faults: list[dict[str, str]] = []
    for source, command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in result.stdout.splitlines():
            if _GPU_FAULT_RE.search(line):
                faults.append({"source": source, "message": line.strip()})
    return faults


def collect_faults(
    start_iso: str,
    *,
    profile_id: str = "example-model-7b",
    include_windows_wer: bool = False,
) -> list[dict[str, str]]:
    faults = linux_gpu_faults(start_iso, profile_id)
    if include_windows_wer:
        faults.extend(windows_live_kernel_reports(start_iso))
    return faults


def gpu_state() -> None:
    """GPU memory and utilization are unknown without a portable provider."""
    return None


def backend_pid(profile_id: str) -> int:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            f"llama-broker-profile-{profile_id}.service",
            "--property=MainPID",
            "--value",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model", default="example-model-7b")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "profiles.json",
    )
    parser.add_argument(
        "--windows-wer",
        action="store_true",
        help="on WSL, additionally fail on new Windows LiveKernelReport files",
    )
    args = parser.parse_args()
    registry = ProfileRegistry.from_path(args.profiles)
    profile = registry.resolve(args.model)
    if args.targets != sorted(set(args.targets)):
        parser.error("targets must be unique and increasing")
    profile_limit = min(MAX_TARGET_TOKENS, profile.context_size - 144)
    if any(target < 128 or target > profile_limit for target in args.targets):
        parser.error(f"targets must be between 128 and {profile_limit} for {profile.id}")

    started = datetime.now().astimezone().isoformat()
    initial_pid = backend_pid(profile.id)
    if initial_pid <= 0:
        raise SystemExit(f"ABORT: backend profile {profile.id} is not active")
    print(
        json.dumps(
            {
                "event": "start",
                "time": started,
                "profile": profile.id,
                "backend_pid": initial_pid,
                "gpu": gpu_state(),
            }
        ),
        flush=True,
    )
    for target in args.targets:
        result = request_prompt(target, args.timeout, args.model)
        faults = collect_faults(
            started,
            profile_id=profile.id,
            include_windows_wer=args.windows_wer,
        )
        current_pid = backend_pid(profile.id)
        measured = result["measured_prompt_tokens"]
        validation_errors: list[str] = []
        if not isinstance(measured, int) or measured < target - 512:
            validation_errors.append(
                f"target {target} only measured {measured!r} prompt tokens"
            )
        if result["profile_header"] != profile.id:
            validation_errors.append(
                "broker profile header mismatch: "
                f"expected {profile.id!r}, got {result['profile_header']!r}"
            )
        if not result["marker_recalled"]:
            validation_errors.append(
                f"target {target} did not recall its opening marker"
            )
        if current_pid != initial_pid:
            validation_errors.append(
                f"backend PID changed from {initial_pid} to {current_pid}"
            )
        result["backend_pid"] = current_pid
        result["gpu"] = gpu_state()
        result["gpu_faults"] = faults
        result["validation_errors"] = validation_errors
        print(json.dumps(result, sort_keys=True), flush=True)
        if faults:
            raise SystemExit("ABORT: a new GPU fault was detected")
        if validation_errors:
            raise SystemExit("ABORT: " + "; ".join(validation_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
