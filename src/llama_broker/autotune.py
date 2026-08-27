"""Safe, offline-testable profile autotuning primitives.

The module deliberately keeps model execution behind injected lifecycle/request
ports.  Core tuning decisions are standard-library-only and never infer a
hardware fingerprint or modify profile identity, endpoint, safety, or security
configuration.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import itertools
import json
import math
import os
import re
import signal
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence



MAX_CANDIDATES = 64
MAX_PROFILE_SECONDS = 30 * 60
WARMUPS_PER_MODE = 2
SERIAL_REQUESTS_PER_MODE = 3
CONCURRENCY_WAVES_PER_MODE = 3
MAX_SCORED_REQUESTS_PER_MODE = 32
MAX_HIGH_VARIANCE_REPEATS = 7
HIGH_VARIANCE_CV = 0.20

# This is the only set of fields the engine may alter.  The omitted fields are
# intentionally forbidden even if a runtime exposes a similarly named flag.
LLAMACPP_KNOBS = (
    "parallel",
    "batch_size",
    "ubatch_size",
    "flash_attention",
    "gpu_layers",
    "max_output_tokens",
)
VLLM_KNOBS = (
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_output_tokens",
)
FORBIDDEN_FIELDS = frozenset(
    {
        "id",
        "runtime",
        "runtime_build",
        "server_binary",
        "server_sha256",
        "model_path",
        "model_sha256",
        "mmproj_path",
        "mmproj_sha256",
        "alias",
        "aliases",
        "host",
        "port",
        "endpoint",
        "credentials",
        "tls",
        "environment",
        "safety",
        "service",
        "supervision",
        "pid_file",
        "log_file",
    }
)

# Fallback values are finite, bounded, and documented here rather than guessed
# from arbitrary help text.  A fallback knob is admitted only if its flag is
# present in --help.
FALLBACK_DOCUMENTED: Mapping[str, Mapping[str, tuple[Any, ...]]] = {
    "llamacpp": {
        "context_size": (2048, 4096, 8192, 16384, 32768),
        "batch_size": (128, 256, 512),
        "ubatch_size": (32, 64, 128),
        "parallel": (1, 2),
        "flash_attention": ("auto", "on", "off"),
        "cache_type_k": ("f16", "q8_0", "q4_0"),
        "cache_type_v": ("f16", "q8_0", "q4_0"),
        "max_output_tokens": (64, 128, 256, 512, 1024),
    },
    "vllm": {
        "context_size": (2048, 4096, 8192, 16384, 32768),
        "tensor_parallel_size": (1, 2),
        "gpu_memory_utilization": (0.80, 0.90, 0.95),
        "kv_cache_dtype": ("auto", "float16", "bfloat16", "fp8"),
        "max_output_tokens": (64, 128, 256, 512, 1024),
    },
}
HELP_FLAGS: Mapping[str, Mapping[str, str]] = {
    "llamacpp": {
        "context_size": r"(?:^|\s)-c\b|--ctx-size|--context-size",
        "batch_size": r"(?:^|\s)-b\b|--batch-size",
        "ubatch_size": r"(?:^|\s)-ub\b|--ubatch-size",
        "parallel": r"(?:^|\s)-np\b|--parallel",
        "flash_attention": r"--flash-attn",
        "cache_type_k": r"-ctk\b|--cache-type-k",
        "cache_type_v": r"-ctv\b|--cache-type-v",
    },
    "vllm": {
        "context_size": r"--max-model-len",
        "tensor_parallel_size": r"--tensor-parallel-size",
        "gpu_memory_utilization": r"--gpu-memory-utilization",
        "kv_cache_dtype": r"--kv-cache-dtype",
    },
}

AUTOTUNE_EXTRA_FLAGS = frozenset(
    {
        "--batch-size",
        "--ubatch-size",
        "--parallel",
        "--flash-attn",
        "--cache-type-k",
        "--cache-type-v",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--kv-cache-dtype",
        "--max-model-len",
    }
)


class AutotuneError(RuntimeError):
    """Base class for fail-closed autotune errors."""


class BaselineRejected(AutotuneError):
    """The current profile is not safe to tune."""


class CandidateRejected(AutotuneError):
    """A candidate failed an operational reliability gate."""


class MaintenanceLeaseExpired(AutotuneError):
    """The CLI-owned maintenance lease expired during a run."""


class ReloadCompensationError(AutotuneError):
    """Manifest reload failed and compensation could not restore service."""


@dataclass(frozen=True)
class KnobSpec:
    name: str
    values: tuple[Any, ...]
    source: str
    confidence: str = "high"
    coupled_group: str | None = None


@dataclass(frozen=True)
class TuningSchema:
    runtime: str
    knobs: tuple[KnobSpec, ...]
    confidence: str
    source: str
    fit_enabled: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(knob.name for knob in self.knobs)


@dataclass(frozen=True)
class Candidate:
    index: int
    changes: Mapping[str, Any]
    extra_args: tuple[str, ...]
    strategy: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseMeasurement:
    generated_tokens: int
    end_to_end_seconds: float
    decode_seconds: float | None = None
    ttft_seconds: float | None = None
    error_class: str | None = None
    malformed: bool = False
    truncated: bool = False
    context_rejected: bool = False

    @property
    def valid(self) -> bool:
        return (
            self.error_class is None
            and not self.malformed
            and not self.truncated
            and not self.context_rejected
            and self.generated_tokens > 0
            and self.end_to_end_seconds > 0
        )


@dataclass(frozen=True)
class ModeResult:
    mode: str
    scored_requests: int
    aggregate_tokens_per_second: float
    p95_latency_seconds: float
    p95_ttft_seconds: float | None
    streaming_decode_tokens_per_second: float | None
    operational_errors: tuple[str, ...]
    valid: bool
    repeats: int


@dataclass(frozen=True)
class BenchmarkResult:
    candidate: Candidate
    modes: tuple[ModeResult, ...]
    valid: bool
    operational_errors: tuple[str, ...]
    aggregate_tokens_per_second: float
    p95_latency_seconds: float
    best_latency_seconds: float
    baseline: bool = False

    @property
    def error_classes(self) -> frozenset[str]:
        return frozenset(self.operational_errors)


@dataclass(frozen=True)
class SearchResult:
    baseline: BenchmarkResult
    winner: BenchmarkResult
    best_latency: BenchmarkResult
    evaluated: tuple[BenchmarkResult, ...]
    schema: TuningSchema
    stopped_reason: str


class CandidateSession(Protocol):
    def request(self, mode: str, *, streaming: bool, concurrent: bool) -> ResponseMeasurement: ...

    def stop(self) -> None: ...


class CandidateLifecycle(Protocol):
    def start(self, candidate: Candidate) -> CandidateSession: ...


class RequestProbe(Protocol):
    def __call__(
        self, session: CandidateSession, mode: str, *, streaming: bool, concurrent: bool
    ) -> ResponseMeasurement: ...


ProgressCallback = Callable[[str, Candidate, int, int, BenchmarkResult | None], None]


class HttpCandidateSession:
    """Deterministic local OpenAI-compatible workload for one candidate."""

    _TEXT_PROMPT = (
        "Write a Python function named safe_divide(a: float, b: float) -> "
        "float | None. Return None on division by zero. Output only code."
    )
    _TOOLS_PROMPT = (
        'Return JSON for tool_calls: get_weather(city="Boston") and '
        'get_stock(symbol="AAPL"). Use the declared tool schema.'
    )
    _VISION_PROMPT = "Describe the provided image in one short sentence."

    def __init__(
        self,
        base_url: str,
        alias: str,
        *,
        image_fixture: Path | None = None,
        timeout: float = 240.0,
        ca_file: str | Path | None = None,
        ssl_context: Any = None,
        api_key: str | None = None,
        max_output_tokens: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.alias = alias
        self.image_fixture = image_fixture
        self.timeout = timeout
        self.ca_file = ca_file
        self.ssl_context = ssl_context
        self.api_key = api_key
        self.max_output_tokens = max(1, int(max_output_tokens))

    def request(self, mode: str, *, streaming: bool, concurrent: bool) -> ResponseMeasurement:
        del concurrent  # The harness supplies bounded waves; requests remain isolated here.
        started = time.monotonic()
        connection: http.client.HTTPConnection | None = None
        try:
            payload = self._payload(mode, streaming=streaming)
            connection = self._connection()
            headers = {"Content-Type": "application/json", "Connection": "close"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            response = connection.getresponse()
            body = response.read()
            elapsed = max(time.monotonic() - started, 1e-9)
            if response.status < 200 or response.status >= 300:
                text = body.decode("utf-8", errors="replace").lower()
                return ResponseMeasurement(
                    0,
                    elapsed,
                    error_class=f"http_{response.status}",
                    context_rejected=response.status in {400, 413}
                    and any(word in text for word in ("context", "token", "prompt")),
                )
            if streaming:
                return self._stream_measurement(body, elapsed)
            return self._json_measurement(body, elapsed)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            elapsed = max(time.monotonic() - started, 1e-9)
            return ResponseMeasurement(0, elapsed, error_class=type(error).__name__)
        finally:
            if connection is not None:
                connection.close()

    def _connection(self) -> http.client.HTTPConnection:
        from .tools.broker_backend import make_connection

        return make_connection(
            self.base_url,
            timeout=self.timeout,
            ca_file=self.ca_file,
            ssl_context=self.ssl_context,
        )

    def _payload(self, mode: str, *, streaming: bool) -> dict[str, Any]:
        if mode == "tools-json":
            content: Any = self._TOOLS_PROMPT
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock",
                        "description": "Get a stock quote",
                        "parameters": {
                            "type": "object",
                            "properties": {"symbol": {"type": "string"}},
                            "required": ["symbol"],
                        },
                    },
                },
            ]
        elif mode == "vision":
            if self.image_fixture is None or not self.image_fixture.is_file():
                raise FileNotFoundError("vision fixture missing")
            encoded = base64.b64encode(self.image_fixture.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": self._VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
            tools = None
        else:
            content = self._TEXT_PROMPT
            tools = None
        payload: dict[str, Any] = {
            "model": self.alias,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_output_tokens,
            "temperature": 0.0,
            "seed": 7,
            "stream": streaming,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _token_count(value: Any) -> int:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            return max(1, len(value.split())) if value.strip() else 0
        return 0

    def _json_measurement(self, body: bytes, elapsed: float) -> ResponseMeasurement:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ResponseMeasurement(0, elapsed, error_class="malformed_json", malformed=True)
        if not isinstance(payload, dict):
            return ResponseMeasurement(0, elapsed, error_class="malformed_json", malformed=True)
        usage = payload.get("usage")
        choices = payload.get("choices")
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", choices[0])
            if isinstance(message, dict):
                raw_content = message.get("content", choices[0].get("text", ""))
                if isinstance(raw_content, str):
                    content = raw_content
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        tokens = self._token_count(tokens) or self._token_count(content)
        if not choices or tokens <= 0:
            return ResponseMeasurement(0, elapsed, error_class="malformed_response", malformed=True)
        return ResponseMeasurement(tokens, elapsed, decode_seconds=elapsed, ttft_seconds=elapsed)

    def _stream_measurement(self, body: bytes, elapsed: float) -> ResponseMeasurement:
        content_parts: list[str] = []
        usage_tokens: int | None = None
        try:
            for raw_line in body.decode("utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    payload = json.loads(data)
                    if not isinstance(payload, dict):
                        continue
                    usage = payload.get("usage")
                    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                        usage_tokens = usage["completion_tokens"]
                    choices = payload.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta", choices[0])
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            content_parts.append(delta["content"])
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return ResponseMeasurement(0, elapsed, error_class="malformed_sse", malformed=True)
        tokens = self._token_count(usage_tokens) or self._token_count("".join(content_parts))
        if tokens <= 0:
            return ResponseMeasurement(0, elapsed, error_class="malformed_stream", malformed=True)
        return ResponseMeasurement(tokens, elapsed, decode_seconds=elapsed, ttft_seconds=elapsed)


class RuntimeAdapter(Protocol):
    name: str
    knob_order: tuple[str, ...]

    def schema(
        self, profile: Any, *, help_text: str | None = None, include_fit: bool = False
    ) -> TuningSchema: ...

    def fallback_schema(self, help_text: str, profile: Any) -> TuningSchema: ...

    def apply(self, record: Mapping[str, Any], candidate: Candidate) -> dict[str, Any]: ...


class _Adapter:
    name = ""
    knob_order: tuple[str, ...] = ()

    def schema(
        self, profile: Any, *, help_text: str | None = None, include_fit: bool = False
    ) -> TuningSchema:
        # Adapter-defined schemas are primary.  They are still finite and
        # bounded, with values anchored to the current profile where useful.
        knobs: list[KnobSpec] = []
        documented = FALLBACK_DOCUMENTED[self.name]
        for name in self.knob_order:
            values = documented.get(name)
            if values is None:
                continue
            if help_text is not None:
                pattern = HELP_FLAGS.get(self.name, {}).get(name)
                if pattern and not re.search(pattern, help_text, re.MULTILINE):
                    continue
            knobs.append(
                KnobSpec(
                    name=name,
                    values=tuple(values),
                    source="adapter-defined" if help_text is None else "adapter-defined-help-confirmed",
                    confidence="high",
                    coupled_group="batch" if name in {"batch_size", "ubatch_size"} else None,
                )
            )
        return TuningSchema(
            runtime=self.name,
            knobs=tuple(knobs),
            confidence="high",
            source="adapter-defined",
            fit_enabled=include_fit,
        )

    def fallback_schema(self, help_text: str, profile: Any) -> TuningSchema:
        knobs: list[KnobSpec] = []
        for name in self.knob_order:
            pattern = HELP_FLAGS.get(self.name, {}).get(name)
            values = FALLBACK_DOCUMENTED[self.name].get(name)
            if pattern and values and re.search(pattern, help_text, re.MULTILINE):
                knobs.append(
                    KnobSpec(
                        name=name,
                        values=tuple(values),
                        source="runtime-help",
                        confidence="low",
                        coupled_group="batch" if name in {"batch_size", "ubatch_size"} else None,
                    )
                )
        return TuningSchema(
            runtime=self.name,
            knobs=tuple(knobs),
            confidence="low",
            source="runtime-help-fallback",
            fit_enabled=False,
        )

    def apply(self, record: Mapping[str, Any], candidate: Candidate) -> dict[str, Any]:
        output = dict(record)
        for field_name, value in candidate.changes.items():
            if field_name in FORBIDDEN_FIELDS or field_name not in self.knob_order:
                raise CandidateRejected(f"forbidden autotune field: {field_name}")
            output[field_name] = value
        original_args = tuple(str(value) for value in record.get("extra_args", ()))
        desired_args = tuple(str(value) for value in candidate.extra_args)
        original_pairs = dict(_split_extra_args(original_args))
        desired_pairs = dict(_split_extra_args(desired_args))
        changed = {
            name
            for name in set(original_pairs) | set(desired_pairs)
            if original_pairs.get(name) != desired_pairs.get(name)
        }
        if changed - AUTOTUNE_EXTRA_FLAGS:
            raise CandidateRejected(
                "candidate changes non-autotune extra_args: "
                f"{sorted(changed - AUTOTUNE_EXTRA_FLAGS)}"
            )
        output["extra_args"] = desired_args
        return output


class LlamaCppAdapter(_Adapter):
    name = "llamacpp"
    knob_order = LLAMACPP_KNOBS


class VllmAdapter(_Adapter):
    name = "vllm"
    knob_order = VLLM_KNOBS


ADAPTERS: Mapping[str, _Adapter] = {
    "llamacpp": LlamaCppAdapter(),
    "vllm": VllmAdapter(),
}


def discover_schema(
    profile: Any,
    *,
    help_text: str | None = None,
    include_fit: bool = False,
) -> TuningSchema:
    """Return a safe adapter schema or a lower-confidence help fallback."""

    runtime = str(getattr(profile, "runtime", ""))
    adapter = ADAPTERS.get(runtime)
    if adapter is None:
        raise AutotuneError(f"unsupported autotune runtime: {runtime!r}")
    if help_text is None:
        return adapter.schema(profile, include_fit=include_fit)
    # The adapter schema remains preferred.  Fallback exists for callers that
    # explicitly report no adapter schema; this branch is intentionally narrow.
    schema = adapter.schema(profile, help_text=help_text, include_fit=include_fit)
    if schema.knobs:
        return schema
    return adapter.fallback_schema(help_text, profile)


def parse_runtime_help(
    runtime: str,
    help_text: str,
    *,
    profile: Any = None,
) -> TuningSchema:
    """Parse only explicitly documented, bounded options from --help."""

    adapter = ADAPTERS.get(runtime)
    if adapter is None:
        raise AutotuneError(f"unsupported autotune runtime: {runtime!r}")
    return adapter.fallback_schema(help_text, profile)


def _split_extra_args(args: Sequence[str]) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []
    index = 0
    while index < len(args):
        flag = str(args[index])
        if flag.startswith("--") and index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            result.append((flag, str(args[index + 1])))
            index += 2
        else:
            result.append((flag, None))
            index += 1
    return result


def patch_autotune_extra_args(
    original: Sequence[str], patch: Sequence[str]
) -> tuple[str, ...]:
    """Replace only flags present in the patch, preserving unrelated arguments."""

    patch_pairs = _split_extra_args(patch)
    patch_names = {name for name, _ in patch_pairs}
    if any(not name.startswith("--") for name in patch_names):
        raise CandidateRejected("autotune extra_args patch contains a non-flag token")
    forbidden = patch_names - AUTOTUNE_EXTRA_FLAGS
    if forbidden:
        raise CandidateRejected(
            f"extra_args patch attempts to modify non-autotune flags: {sorted(forbidden)}"
        )
    original_pairs = _split_extra_args(original)
    output: list[str] = []
    consumed: set[str] = set()
    for name, value in original_pairs:
        if name in patch_names:
            if name in consumed:
                continue
            replacement = next(value for flag, value in patch_pairs if flag == name)
            output.append(name)
            if replacement is not None:
                output.append(replacement)
            consumed.add(name)
        else:
            output.append(name)
            if value is not None:
                output.append(value)
    for name, value in patch_pairs:
        if name not in consumed and name not in {flag for flag, _ in original_pairs}:
            output.append(name)
            if value is not None:
                output.append(value)
    return tuple(output)


def generate_candidates(
    profile: Any,
    schema: TuningSchema,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[Candidate, ...]:
    """Generate baseline, stable one-knob refinements, then limited interactions."""

    if max_candidates <= 0 or max_candidates > MAX_CANDIDATES:
        raise AutotuneError(f"max_candidates must be in 1..{MAX_CANDIDATES}")
    baseline_changes = {
        knob.name: getattr(profile, knob.name)
        for knob in schema.knobs
        if hasattr(profile, knob.name) and getattr(profile, knob.name) is not None
    }
    original_extra = tuple(str(value) for value in getattr(profile, "extra_args", ()))
    candidates: list[Candidate] = [
        Candidate(0, baseline_changes, original_extra, "baseline", {"confidence": schema.confidence})
    ]
    index = 1
    for knob in schema.knobs:
        current = baseline_changes.get(knob.name)
        for value in knob.values:
            if value == current:
                continue
            if index >= max_candidates:
                return tuple(candidates)
            candidates.append(
                Candidate(
                    index,
                    {knob.name: value},
                    original_extra,
                    "one-knob",
                    {"knob": knob.name, "confidence": knob.confidence},
                )
            )
            index += 1
    coupled = [knob for knob in schema.knobs if knob.coupled_group == "batch"]
    if len(coupled) >= 2:
        first, second = coupled[:2]
        for left, right in itertools.product(first.values, second.values):
            if left == baseline_changes.get(first.name) and right == baseline_changes.get(second.name):
                continue
            if index >= max_candidates:
                break
            candidates.append(
                Candidate(
                    index,
                    {first.name: left, second.name: right},
                    original_extra,
                    "interaction",
                    {"knobs": (first.name, second.name), "confidence": schema.confidence},
                )
            )
            index += 1
    return tuple(candidates)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class BenchmarkHarness:
    """Execute protocol measurements using a fresh lifecycle for each candidate."""

    def __init__(
        self,
        lifecycle: CandidateLifecycle,
        *,
        image_fixture: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        request_probe: RequestProbe | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.image_fixture = image_fixture
        self.clock = clock
        self.request_probe = request_probe or (
            lambda session, mode, *, streaming, concurrent: session.request(
                mode, streaming=streaming, concurrent=concurrent
            )
        )

    def run(
        self,
        candidate: Candidate,
        *,
        capabilities: Iterable[str],
        parallelism: int,
        baseline: bool = False,
    ) -> BenchmarkResult:
        capabilities = tuple(capabilities)
        if "vision" in capabilities and (
            self.image_fixture is None or not self.image_fixture.is_file()
        ):
            return self._invalid(candidate, "missing_vision_fixture", baseline=baseline)
        if parallelism <= 0:
            return self._invalid(candidate, "invalid_parallelism", baseline=baseline)
        session: CandidateSession | None = None
        try:
            # CandidateLifecycle is required to create a fresh backend process.
            session = self.lifecycle.start(candidate)
            modes: list[ModeResult] = []
            for capability in capabilities:
                mode = "tools-json" if capability == "tools" else capability
                measurements: list[ResponseMeasurement] = []
                errors: list[str] = []
                for _ in range(WARMUPS_PER_MODE):
                    measurement = self.request_probe(
                        session, mode, streaming=False, concurrent=False
                    )
                    self._collect(measurement, [], errors)
                for _ in range(SERIAL_REQUESTS_PER_MODE):
                    measurement = self.request_probe(
                        session, mode, streaming=False, concurrent=False
                    )
                    self._collect(measurement, measurements, errors)
                scored = SERIAL_REQUESTS_PER_MODE
                wave_width = min(max(1, parallelism), MAX_SCORED_REQUESTS_PER_MODE)
                for _ in range(CONCURRENCY_WAVES_PER_MODE):
                    remaining = MAX_SCORED_REQUESTS_PER_MODE - scored
                    request_count = min(wave_width, remaining)
                    if request_count <= 0:
                        break
                    with ThreadPoolExecutor(max_workers=request_count) as executor:
                        futures = [
                            executor.submit(
                                self.request_probe,
                                session,
                                mode,
                                streaming=True,
                                concurrent=True,
                            )
                            for _ in range(request_count)
                        ]
                        for future in as_completed(futures):
                            measurement = future.result()
                            self._collect(measurement, measurements, errors)
                            scored += 1
                repeats = 1
                while measurements and self._high_variance(measurements) and repeats < MAX_HIGH_VARIANCE_REPEATS:
                    measurement = self.request_probe(
                        session, mode, streaming=False, concurrent=False
                    )
                    self._collect(measurement, measurements, errors)
                    repeats += 1
                modes.append(self._mode_result(mode, measurements, errors, repeats))
            operational_errors = tuple(sorted(set(error for mode in modes for error in mode.operational_errors)))
            valid = bool(modes) and all(mode.valid for mode in modes)
            aggregate = statistics.fmean(
                [mode.aggregate_tokens_per_second for mode in modes]
            ) if modes else 0.0
            p95 = max((mode.p95_latency_seconds for mode in modes), default=math.inf)
            return BenchmarkResult(
                candidate=candidate,
                modes=tuple(modes),
                valid=valid,
                operational_errors=operational_errors,
                aggregate_tokens_per_second=aggregate,
                p95_latency_seconds=p95,
                best_latency_seconds=min(
                    (mode.p95_latency_seconds for mode in modes), default=math.inf
                ),
                baseline=baseline,
            )
        except MaintenanceLeaseExpired:
            raise
        except (AutotuneError, OSError, subprocess.SubprocessError, TimeoutError, RuntimeError) as error:
            return self._invalid(candidate, type(error).__name__, baseline=baseline)
        finally:
            if session is not None:
                session.stop()

    @staticmethod
    def _collect(
        measurement: ResponseMeasurement,
        measurements: list[ResponseMeasurement],
        errors: list[str],
    ) -> None:
        if measurement.error_class:
            errors.append(measurement.error_class)
        if measurement.malformed:
            errors.append("malformed_response")
        if measurement.truncated:
            errors.append("truncation")
        if measurement.context_rejected:
            errors.append("context_rejection")
        if measurement.valid:
            measurements.append(measurement)

    @staticmethod
    def _high_variance(measurements: Sequence[ResponseMeasurement]) -> bool:
        values = [measurement.end_to_end_seconds for measurement in measurements]
        if len(values) < 3:
            return False
        mean = statistics.fmean(values)
        return mean > 0 and statistics.pstdev(values) / mean > HIGH_VARIANCE_CV

    @staticmethod
    def _mode_result(
        mode: str,
        measurements: Sequence[ResponseMeasurement],
        errors: Sequence[str],
        repeats: int,
    ) -> ModeResult:
        if not measurements:
            return ModeResult(mode, 0, 0.0, math.inf, None, None, tuple(sorted(set(errors))), False, repeats)
        tokens = sum(item.generated_tokens for item in measurements)
        duration = sum(item.decode_seconds or item.end_to_end_seconds for item in measurements)
        ttfts = [item.ttft_seconds for item in measurements if item.ttft_seconds is not None]
        decode_values = [
            item.generated_tokens / item.decode_seconds
            for item in measurements
            if item.decode_seconds and item.decode_seconds > 0
        ]
        return ModeResult(
            mode=mode,
            scored_requests=len(measurements),
            aggregate_tokens_per_second=tokens / duration if duration else 0.0,
            p95_latency_seconds=_percentile(
                [item.end_to_end_seconds for item in measurements], 0.95
            ),
            p95_ttft_seconds=_percentile(ttfts, 0.95) if ttfts else None,
            streaming_decode_tokens_per_second=statistics.fmean(decode_values) if decode_values else None,
            operational_errors=tuple(sorted(set(errors))),
            valid=not errors,
            repeats=repeats,
        )

    @staticmethod
    def _invalid(candidate: Candidate, error: str, *, baseline: bool) -> BenchmarkResult:
        return BenchmarkResult(
            candidate=candidate,
            modes=(),
            valid=False,
            operational_errors=(error,),
            aggregate_tokens_per_second=0.0,
            p95_latency_seconds=math.inf,
            best_latency_seconds=math.inf,
            baseline=baseline,
        )


class LeaseBusy(AutotuneError):
    """Another autotune maintenance operation owns the lease."""


@dataclass(frozen=True)
class LeaseRecord:
    owner_pid: int
    token: str
    expires_at: float


class MaintenanceLease:
    """A local, expiring CLI-owned maintenance lease.

    The lease file is deliberately separate from the broker manifest.  A
    daemon can report an expired lease without trusting a still-running CLI,
    and the next owner can reclaim records whose PID no longer exists.
    """

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float = 60.0,
        pid: int | None = None,
        clock: Callable[[], float] = time.time,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("lease ttl must be positive")
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.pid = os.getpid() if pid is None else int(pid)
        self.clock = clock
        self.pid_alive = pid_alive or self._default_pid_alive
        self._token: str | None = None

    @staticmethod
    def _default_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False
        return True

    def _read(self) -> LeaseRecord | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return LeaseRecord(
                owner_pid=int(payload["owner_pid"]),
                token=str(payload["token"]),
                expires_at=float(payload["expires_at"]),
            )
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def status(self) -> dict[str, Any]:
        record = self._read()
        if record is None:
            return {"held": False, "expired": False}
        expired = record.expires_at <= self.clock() or not self.pid_alive(record.owner_pid)
        return {
            "held": not expired,
            "expired": expired,
            "owner_pid": record.owner_pid,
            "expires_at": record.expires_at,
        }

    def acquire(self) -> LeaseRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read()
        if existing is not None:
            expired = existing.expires_at <= self.clock() or not self.pid_alive(existing.owner_pid)
            if not expired:
                raise LeaseBusy(f"autotune maintenance lease held by PID {existing.owner_pid}")
            self.path.unlink(missing_ok=True)
        token = f"{self.pid:x}-{time.time_ns():x}"
        record = LeaseRecord(self.pid, token, self.clock() + self.ttl_seconds)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise LeaseBusy("autotune maintenance lease was acquired concurrently") from error
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record.__dict__, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        self._token = token
        return record

    def heartbeat(self) -> LeaseRecord:
        record = self._read()
        if record is None or record.token != self._token or record.owner_pid != self.pid:
            raise MaintenanceLeaseExpired("autotune maintenance lease is no longer owned")
        updated = LeaseRecord(self.pid, record.token, self.clock() + self.ttl_seconds)
        atomic_write_json(self.path, updated.__dict__)
        return updated

    def ensure_valid(self) -> None:
        record = self._read()
        if (
            record is None
            or record.token != self._token
            or record.owner_pid != self.pid
            or record.expires_at <= self.clock()
            or not self.pid_alive(record.owner_pid)
        ):
            raise MaintenanceLeaseExpired("autotune maintenance lease expired")

    def release(self) -> None:
        record = self._read()
        if record is not None and record.token == self._token and record.owner_pid == self.pid:
            self.path.unlink(missing_ok=True)
        self._token = None

    def __enter__(self) -> "MaintenanceLease":
        self.acquire()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.release()


class _SystemdCandidateSession:
    def __init__(
        self,
        unit_name: str,
        request_factory: Callable[[str, str, bool, bool], ResponseMeasurement],
        stop_callback: Callable[[], None],
    ) -> None:
        self.unit_name = unit_name
        self._request_factory = request_factory
        self._stop_callback = stop_callback
        self._stopped = False

    def request(self, mode: str, *, streaming: bool, concurrent: bool) -> ResponseMeasurement:
        if self._stopped:
            raise RuntimeError("candidate session is stopped")
        return self._request_factory(self.unit_name, mode, streaming, concurrent)

    def stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self._stop_callback()


class SystemdCandidateLifecycle:
    """Create one unique systemd user unit per candidate and always clean it."""

    def __init__(
        self,
        command_for: Callable[[Candidate], Sequence[str]],
        request_factory: Callable[[str, str, bool, bool], ResponseMeasurement],
        *,
        unit_dir: Path,
        runner: Callable[[Sequence[str]], Any] | None = None,
        readiness_probe: Callable[[Candidate], bool] | None = None,
        unit_directives_for: Callable[[Candidate], Sequence[str]] | None = None,
        clock: Callable[[], float] = time.time,
        command_timeout: float = 45.0,
    ) -> None:
        self.command_for = command_for
        self.request_factory = request_factory
        self.unit_dir = Path(unit_dir)
        self.readiness_probe = readiness_probe or (lambda _candidate: True)
        self.unit_directives_for = unit_directives_for or (lambda _candidate: ())
        self.clock = clock
        if command_timeout <= 0:
            raise ValueError("candidate command timeout must be positive")
        self.command_timeout = float(command_timeout)
        if runner is None:
            self.runner = lambda command: subprocess.run(
                command, check=True, timeout=self.command_timeout
            )
        else:
            self.runner = runner

    def _run(self, *command: str) -> None:
        self.runner(command)

    def _cleanup(self, unit_name: str, unit_path: Path, *, stop: bool) -> None:
        try:
            # Clear a failed result while systemd still has the candidate unit
            # loaded.  Stopping first can garbage-collect the transient unit,
            # which makes reset-failed produce a misleading "not loaded" line.
            self._run("systemctl", "--user", "reset-failed", unit_name)
        except (OSError, subprocess.SubprocessError):
            pass
        if stop:
            try:
                self._run("systemctl", "--user", "stop", unit_name)
            except (OSError, subprocess.SubprocessError):
                pass
        unit_path.unlink(missing_ok=True)
        try:
            self._run("systemctl", "--user", "daemon-reload")
        except (OSError, subprocess.SubprocessError):
            pass

    def start(self, candidate: Candidate) -> CandidateSession:
        import shlex

        self.unit_dir.mkdir(parents=True, exist_ok=True)
        unit_name = f"llama-broker-autotune-{candidate.index}-{time.time_ns()}.service"
        unit_path = self.unit_dir / unit_name
        command = tuple(str(part) for part in self.command_for(candidate))
        if not command:
            raise AutotuneError("candidate command cannot be empty")
        directives = [str(line) for line in self.unit_directives_for(candidate)]
        unit_path.write_text(
            "[Unit]\nDescription=Llama Broker autotune candidate\n"
            "[Service]\nType=exec\n"
            f"ExecStart={shlex.join(command)}\n"
            + "".join(f"{line.rstrip()}\n" for line in directives)
            + "KillSignal=SIGINT\nTimeoutStopSec=20\n",
            encoding="utf-8",
        )
        started = False
        try:
            self._run("systemctl", "--user", "daemon-reload")
            self._run("systemctl", "--user", "start", unit_name)
            started = True
            if not self.readiness_probe(candidate):
                raise AutotuneError("candidate backend failed readiness")
            return _SystemdCandidateSession(
                unit_name,
                self.request_factory,
                lambda: self._cleanup(unit_name, unit_path, stop=True),
            )
        except BaseException:
            self._cleanup(unit_name, unit_path, stop=started)
            raise


class AutotuneEngine:
    """Run baseline-first bounded search with a reliability gate."""

    def __init__(
        self,
        *,
        schema: TuningSchema,
        candidates: Sequence[Candidate],
        benchmark: Callable[[Candidate, bool], BenchmarkResult],
        clock: Callable[[], float] = time.monotonic,
        max_seconds: float = MAX_PROFILE_SECONDS,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.schema = schema
        self.candidates = tuple(candidates)
        self.benchmark = benchmark
        self.clock = clock
        self.max_seconds = min(float(max_seconds), float(MAX_PROFILE_SECONDS))
        self.progress = progress

    def _progress(
        self,
        phase: str,
        candidate: Candidate,
        completed: int,
        total: int,
        result: BenchmarkResult | None = None,
    ) -> None:
        if self.progress is not None:
            self.progress(phase, candidate, completed, total, result)

    def run(self) -> SearchResult:
        if not self.candidates or self.candidates[0].strategy != "baseline":
            raise BaselineRejected("baseline must be the first candidate")
        started = self.clock()
        total = min(len(self.candidates), MAX_CANDIDATES)
        self._progress("started", self.candidates[0], 0, total)
        baseline = self.benchmark(self.candidates[0], True)
        self._progress("completed", self.candidates[0], 1, total, baseline)
        if not baseline.valid:
            raise BaselineRejected(
                f"baseline rejected: {', '.join(baseline.operational_errors) or 'invalid'}"
            )
        evaluated = [baseline]
        stopped_reason = "candidate_limit"
        for candidate in self.candidates[1:]:
            if len(evaluated) >= MAX_CANDIDATES:
                break
            if self.clock() - started >= self.max_seconds:
                stopped_reason = "time_limit"
                break
            self._progress("started", candidate, len(evaluated), total)
            result = self.benchmark(candidate, False)
            evaluated.append(result)
            self._progress("completed", candidate, len(evaluated), total, result)
        reliable = [result for result in evaluated if self._reliable(result, baseline)]
        if not reliable:
            raise CandidateRejected("no reliable autotune candidates")
        winner = max(
            reliable,
            key=lambda result: (
                result.aggregate_tokens_per_second,
                -result.p95_latency_seconds,
            ),
        )
        best_latency = min(
            reliable,
            key=lambda result: (result.p95_latency_seconds, -result.aggregate_tokens_per_second),
        )
        return SearchResult(
            baseline=baseline,
            winner=winner,
            best_latency=best_latency,
            evaluated=tuple(evaluated),
            schema=self.schema,
            stopped_reason=stopped_reason,
        )

    @staticmethod
    def _reliable(result: BenchmarkResult, baseline: BenchmarkResult) -> bool:
        if not result.valid:
            return False
        if not baseline.operational_errors:
            return not result.operational_errors
        return result.error_classes <= baseline.error_classes


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist JSON with fsync and atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a binary/text snapshot while preserving durability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def profile_revision(record: Mapping[str, Any]) -> str:
    """Return a stable digest for configuration owned by the profile itself."""

    canonical = json.loads(json.dumps(record))
    if isinstance(canonical, dict):
        canonical.pop("autotune", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_winner_to_manifest(
    manifest: Mapping[str, Any],
    profile_id: str,
    winner: Candidate,
    *,
    adapter: RuntimeAdapter,
    autotune_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copied manifest containing only whitelisted winner changes."""

    output = json.loads(json.dumps(manifest))
    records = output.get("profiles")
    if not isinstance(records, list):
        raise AutotuneError("manifest profiles must be a list")
    for record in records:
        if isinstance(record, dict) and record.get("id") == profile_id:
            updated = adapter.apply(record, winner)
            updated["autotune"] = dict(
                autotune_metadata
                or {
                    "status": "complete",
                    "winner_candidate": winner.index,
                    "strategy": winner.strategy,
                    "confidence": winner.metadata.get("confidence", "high"),
                    "accuracy": "not_measured",
                }
            )
            record.clear()
            record.update(updated)
            return output
    raise AutotuneError(f"profile not found: {profile_id}")


class AutotuneTransaction:
    """Commit a winner, reload once, and compensate on reload failure."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        reload_registry: Callable[[], None],
        restore_active: Callable[[str | None], None],
        quiescent: Callable[[], bool] = lambda: True,
        mark_degraded: Callable[[str], None] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.reload_registry = reload_registry
        self.restore_active = restore_active
        self.quiescent = quiescent
        self.mark_degraded = mark_degraded or (lambda _reason: None)

    def commit(
        self,
        manifest: Mapping[str, Any],
        *,
        profile_id: str,
        winner: Candidate,
        adapter: RuntimeAdapter,
        previous_active: str | None,
        state_path: Path | None = None,
    ) -> dict[str, Any]:
        result = self.commit_many(
            manifest,
            winners={profile_id: (winner, adapter)},
            previous_active=previous_active,
            state_path=state_path,
        )
        result["profile_id"] = profile_id
        result["winner_candidate"] = winner.index
        return result

    def commit_many(
        self,
        manifest: Mapping[str, Any],
        *,
        winners: Mapping[str, tuple[Candidate, RuntimeAdapter]],
        previous_active: str | None,
        state_path: Path | None = None,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Commit several profile winners with one reload and compensation boundary."""

        if not winners:
            raise AutotuneError("at least one profile winner is required")
        if not self.quiescent():
            raise AutotuneError("coordinator is not quiescent for autotune reload")
        original = self.manifest_path.read_bytes()
        updated: Mapping[str, Any] = manifest
        for profile_id, (winner, adapter) in winners.items():
            updated = apply_winner_to_manifest(
                updated,
                profile_id,
                winner,
                adapter=adapter,
                autotune_metadata=(metadata or {}).get(profile_id),
            )
        atomic_write_json(self.manifest_path, updated)
        result = {
            "profile_ids": list(winners),
            "winner_candidates": {
                profile_id: winner.index
                for profile_id, (winner, _adapter) in winners.items()
            },
            "reloads": 0,
            "active_profile": previous_active,
            "status": "committed",
        }
        try:
            self.reload_registry()
            result["reloads"] = 1
            self.restore_active(previous_active)
            if state_path is not None:
                atomic_write_json(state_path, result)
            return result
        except BaseException as error:
            recovery_errors: list[str] = []
            try:
                atomic_write_bytes(self.manifest_path, original)
                self.reload_registry()
            except BaseException as recovery_error:
                recovery_errors.append(f"reload old registry: {recovery_error}")
            try:
                self.restore_active(previous_active)
            except BaseException as recovery_error:
                recovery_errors.append(f"restore active profile: {recovery_error}")
            if recovery_errors:
                reason = f"reload failed: {error}; compensation failed: {'; '.join(recovery_errors)}"
                self.mark_degraded(reason)
                raise ReloadCompensationError(reason) from error
            raise AutotuneError(f"autotune reload failed and was compensated: {error}") from error


__all__ = [
    "ADAPTERS",
    "AutotuneEngine",
    "AutotuneError",
    "AutotuneTransaction",
    "BaselineRejected",
    "BenchmarkHarness",
    "BenchmarkResult",
    "Candidate",
    "CandidateRejected",
    "FALLBACK_DOCUMENTED",
    "HttpCandidateSession",
    "KnobSpec",
    "LlamaCppAdapter",
    "MaintenanceLeaseExpired",
    "LeaseBusy",
    "LeaseRecord",
    "ModeResult",
    "ReloadCompensationError",
    "ResponseMeasurement",
    "SearchResult",
    "SystemdCandidateLifecycle",
    "TuningSchema",
    "VllmAdapter",
    "apply_winner_to_manifest",
    "atomic_write_json",
    "atomic_write_bytes",
    "discover_schema",
    "generate_candidates",
    "parse_runtime_help",
    "patch_autotune_extra_args",
]
