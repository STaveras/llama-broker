"""Optional "system 1" decision model support.

A system 1 model answers typed questions about a state in a single forward
pass and returns probabilities instead of generated tokens.  The first
supported model is Liquid AI's d1 (``LiquidAI/d1-3B-GGUF``), which
``llama-server`` exposes on ``POST /v1/systemone``.

The decision model runs as a small, always-resident side backend in its own
systemd user unit.  It is deliberately outside the profile transition
lifecycle: the broker never drains, swaps or stops it while switching the main
backend, and its unit name does not match the profile service prefix, so the
controller's quiescence checks never treat it as a stray profile backend.

The broker uses it for two things, both off unless ``system1.json`` enables
them:

* ``POST /v1/systemone`` is proxied to the side backend so clients can ask
  their own fast questions through the broker's authenticated port.
* Requests whose ``model`` is a routing alias (``auto`` by default) are routed
  by asking d1 which configured profile should handle the request.
"""

from __future__ import annotations

import http.client
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .backend_controller import _unit_quote
from .paths import CONFIG_DIR, SYSTEMD_USER_DIR
from .tools.broker_backend import make_connection


SYSTEM1_CONFIG_PATH = Path(
    os.environ.get("LLAMA_BROKER_SYSTEM1_CONFIG", CONFIG_DIR / "system1.json")
).expanduser()
SYSTEM1_UNIT_NAME = "llama-broker-system1.service"
SYSTEM1_ENDPOINT = "/v1/systemone"
ROUTE_QUESTION = "route"

_DEFAULT_ROUTE_INSTRUCTIONS = (
    "Which model should handle this request? Prefer the smallest model that "
    "can do the job well."
)
_MAX_STATE_CHARS_LIMIT = 200_000


class System1Error(RuntimeError):
    """The decision model is unconfigured, unreachable or answered unusably."""


class System1ConfigError(ValueError):
    pass


def _positive_int(record: Mapping[str, Any], key: str, default: int) -> int:
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise System1ConfigError(f"system1.{key} must be a positive integer")
    return value


def _positive_float(record: Mapping[str, Any], key: str, default: float) -> float:
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise System1ConfigError(f"system1.{key} must be a positive number")
    return float(value)


def _probability(record: Mapping[str, Any], key: str, default: float) -> float:
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise System1ConfigError(f"system1.routing.{key} must be between 0 and 1")
    return float(value)


def _string_list(record: Mapping[str, Any], key: str, default: Sequence[str]) -> tuple[str, ...]:
    value = record.get(key, list(default))
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise System1ConfigError(f"system1.{key} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class System1RoutingConfig:
    enabled: bool = True
    aliases: tuple[str, ...] = ("auto",)
    candidates: tuple[str, ...] = ()
    descriptions: Mapping[str, str] = field(default_factory=dict)
    instructions: str = _DEFAULT_ROUTE_INSTRUCTIONS
    min_confidence: float = 0.5
    # A cross-profile switch drains requests and reloads VRAM, so a different
    # profile must beat the active one by this much before d1 can cause it.
    switch_margin: float = 0.15
    fallback: str = "active"
    max_state_chars: int = 4000

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "System1RoutingConfig":
        if not isinstance(record, Mapping):
            raise System1ConfigError("system1.routing must be an object")
        enabled = record.get("enabled", True)
        if not isinstance(enabled, bool):
            raise System1ConfigError("system1.routing.enabled must be a boolean")
        descriptions = record.get("descriptions", {})
        if not isinstance(descriptions, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in descriptions.items()
        ):
            raise System1ConfigError(
                "system1.routing.descriptions must map profile ids to strings"
            )
        instructions = record.get("instructions", _DEFAULT_ROUTE_INSTRUCTIONS)
        if not isinstance(instructions, str) or not instructions.strip():
            raise System1ConfigError("system1.routing.instructions must be a string")
        fallback = record.get("fallback", "active")
        if not isinstance(fallback, str) or not fallback.strip():
            raise System1ConfigError(
                "system1.routing.fallback must be 'active', 'error' or a profile id"
            )
        max_state_chars = _positive_int(record, "max_state_chars", 4000)
        if max_state_chars > _MAX_STATE_CHARS_LIMIT:
            raise System1ConfigError(
                f"system1.routing.max_state_chars cannot exceed {_MAX_STATE_CHARS_LIMIT}"
            )
        aliases = _string_list(record, "aliases", ("auto",))
        if not aliases:
            raise System1ConfigError("system1.routing.aliases must not be empty")
        return cls(
            enabled=enabled,
            aliases=aliases,
            candidates=_string_list(record, "candidates", ()),
            descriptions=dict(descriptions),
            instructions=instructions.strip(),
            min_confidence=_probability(record, "min_confidence", 0.5),
            switch_margin=_probability(record, "switch_margin", 0.15),
            fallback=fallback.strip(),
            max_state_chars=max_state_chars,
        )


@dataclass(frozen=True)
class System1Config:
    """Validated ``system1.json``."""

    model: str = "d1-3B"
    managed: bool = True
    url: str = "http://127.0.0.1:10001"
    server_binary: Path | None = None
    model_path: Path | None = None
    context_size: int = 8192
    gpu_layers: str = "all"
    extra_args: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    memory_max_mib: int = 8192
    timeout: float = 5.0
    startup_timeout: float = 120.0
    routing: System1RoutingConfig = field(default_factory=System1RoutingConfig)

    @property
    def listen_host(self) -> str:
        return urlsplit(self.url).hostname or "127.0.0.1"

    @property
    def listen_port(self) -> int:
        return urlsplit(self.url).port or 80

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "System1Config":
        if not isinstance(record, Mapping):
            raise System1ConfigError("system1 configuration must be an object")
        managed = record.get("managed", True)
        if not isinstance(managed, bool):
            raise System1ConfigError("system1.managed must be a boolean")
        url = record.get("url", "http://127.0.0.1:10001")
        if not isinstance(url, str):
            raise System1ConfigError("system1.url must be a string")
        url = url.rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise System1ConfigError("system1.url must be an http(s) URL")
        model = record.get("model", "d1-3B")
        if not isinstance(model, str) or not model.strip():
            raise System1ConfigError("system1.model must be a non-empty string")

        def optional_path(key: str) -> Path | None:
            value = record.get(key)
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise System1ConfigError(f"system1.{key} must be a path string")
            return Path(value).expanduser()

        server_binary = optional_path("server_binary")
        model_path = optional_path("model_path")
        if managed:
            if server_binary is None or model_path is None:
                raise System1ConfigError(
                    "a managed system1 backend needs server_binary and model_path"
                )
            if parts.scheme != "http":
                raise System1ConfigError("a managed system1 backend listens on plain http")
            if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise System1ConfigError(
                    "a managed system1 backend must listen on loopback"
                )
            if parts.port is None:
                raise System1ConfigError("system1.url needs an explicit port")

        gpu_layers = record.get("gpu_layers", "all")
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, (int, str)):
            raise System1ConfigError("system1.gpu_layers must be an integer or string")
        environment = record.get("environment", {})
        if not isinstance(environment, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise System1ConfigError("system1.environment must map names to strings")
        return cls(
            model=model.strip(),
            managed=managed,
            url=url,
            server_binary=server_binary,
            model_path=model_path,
            context_size=_positive_int(record, "context_size", 8192),
            gpu_layers=str(gpu_layers),
            extra_args=_string_list(record, "extra_args", ()),
            environment=dict(environment),
            memory_max_mib=_positive_int(record, "memory_max_mib", 8192),
            timeout=_positive_float(record, "timeout", 5.0),
            startup_timeout=_positive_float(record, "startup_timeout", 120.0),
            routing=System1RoutingConfig.from_mapping(record.get("routing", {})),
        )


def load_system1_config(path: Path | None = None) -> System1Config | None:
    """Return the enabled configuration, or ``None`` when system 1 is off."""

    path = path or SYSTEM1_CONFIG_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError as error:
        raise System1ConfigError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(record, Mapping):
        raise System1ConfigError(f"{path}: system1 configuration must be an object")
    enabled = record.get("enabled", False)
    if not isinstance(enabled, bool):
        raise System1ConfigError("system1.enabled must be a boolean")
    if not enabled:
        return None
    return System1Config.from_mapping(record)


class System1Client:
    """Minimal client for a llama-server ``/v1/systemone`` endpoint."""

    def __init__(self, url: str, *, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def post_raw(self, body: bytes) -> tuple[int, str, bytes]:
        connection = make_connection(self.url, timeout=self.timeout)
        try:
            connection.request(
                "POST",
                SYSTEM1_ENDPOINT,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            data = response.read()
            return (
                response.status,
                response.getheader("Content-Type", "application/json"),
                data,
            )
        finally:
            connection.close()

    def decide(
        self,
        state: str | None,
        questions: Mapping[str, Any],
        *,
        images: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Ask *questions* about *state* and return the ``answers`` object."""

        request: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if images:
            request["images"] = list(images)
        body = json.dumps(request, separators=(",", ":")).encode("utf-8")
        try:
            status, _content_type, data = self.post_raw(body)
        except (OSError, http.client.HTTPException) as error:
            raise System1Error(f"system1 backend unreachable: {error}") from error
        if status != 200:
            raise System1Error(f"system1 backend returned HTTP {status}")
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError) as error:
            raise System1Error("system1 backend returned invalid JSON") from error
        answers = payload.get("answers") if isinstance(payload, Mapping) else None
        if not isinstance(answers, Mapping):
            raise System1Error("system1 response has no answers object")
        return dict(answers)

    def healthy(self) -> bool:
        try:
            connection = make_connection(self.url, timeout=min(self.timeout, 0.5))
            try:
                connection.request("GET", "/health", headers={"Connection": "close"})
                response = connection.getresponse()
                response.read()
                return response.status == 200
            finally:
                connection.close()
        except (OSError, http.client.HTTPException, ValueError):
            return False


def _as_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def choice_probabilities(answer: Any) -> dict[str, float]:
    """Normalise a ``choice`` answer to ``{label: probability}``.

    d1 nests each answer under its question type
    (``answers[name]["choice"]``).  The probabilities are accepted either as a
    label-to-probability object or as a list of ``{label, probability}``
    records so a minor server format change does not break routing.
    """

    if isinstance(answer, Mapping) and "choice" in answer:
        answer = answer["choice"]
    if isinstance(answer, Mapping):
        for key in ("probabilities", "probs", "scores"):
            if key in answer:
                answer = answer[key]
                break
    result: dict[str, float] = {}
    if isinstance(answer, Mapping):
        for label, value in answer.items():
            probability = _as_probability(value)
            if isinstance(label, str) and probability is not None:
                result[label] = probability
    elif isinstance(answer, list):
        for item in answer:
            if not isinstance(item, Mapping):
                continue
            label = item.get("label", item.get("name"))
            probability = _as_probability(
                item.get("probability", item.get("prob", item.get("score")))
            )
            if isinstance(label, str) and probability is not None:
                result[label] = probability
    if not result:
        raise System1Error("system1 choice answer has no label probabilities")
    return result


def _content_text_and_images(content: Any) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    images: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind in {"text", "input_text"} and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif kind in {"image_url", "input_image"}:
                image = part.get("image_url")
                if isinstance(image, Mapping):
                    image = image.get("url")
                if isinstance(image, str):
                    images.append(image)
    return texts, images


def request_state(payload: Mapping[str, Any], *, max_chars: int) -> tuple[str, bool]:
    """Return the text d1 judges for a routing decision, and whether it has images.

    The state is the system prompt (trimmed) plus the latest user turn, since
    that is what decides which model is needed.  Long states keep their tail,
    where the actual ask usually is.
    """

    has_images = False
    sections: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        system_text = ""
        last_user = ""
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            texts, images = _content_text_and_images(message.get("content"))
            has_images = has_images or bool(images)
            if message.get("role") in {"system", "developer"} and not system_text:
                system_text = "\n".join(texts)
            elif message.get("role") == "user":
                last_user = "\n".join(texts)
        if system_text:
            sections.append("System prompt: " + system_text[: max_chars // 4])
        if last_user:
            sections.append("User request: " + last_user)
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            sections.append(f"The request offers {len(tools)} tool(s).")
    else:
        prompt = payload.get("prompt", payload.get("input"))
        if isinstance(prompt, str):
            sections.append("User request: " + prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, Mapping):
                    texts, images = _content_text_and_images(item.get("content"))
                    has_images = has_images or bool(images)
                    sections.extend(texts)
                elif isinstance(item, str):
                    sections.append(item)
    if has_images:
        sections.append("The request includes images.")
    state = "\n\n".join(sections)
    if len(state) > max_chars:
        state = "…" + state[-(max_chars - 1):]
    return state, has_images


@dataclass(frozen=True)
class RouteCandidate:
    profile_id: str
    alias: str
    description: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class RouteDecision:
    profile_id: str | None
    reason: str
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


def describe_profile(profile: Any, override: str | None = None) -> str:
    if override:
        return override
    name = str(getattr(profile, "display_name", "") or getattr(profile, "id", ""))
    capabilities = ", ".join(getattr(profile, "capabilities", ()) or ("text",))
    context = getattr(profile, "context_size", None)
    details = f"capabilities: {capabilities}"
    if isinstance(context, int) and context > 0:
        details += f"; context {context} tokens"
    return f"{name} ({details})"


class System1Router:
    """Choose a profile for a routing-alias request by asking d1."""

    def __init__(self, config: System1RoutingConfig, client: System1Client) -> None:
        self.config = config
        self.client = client

    def handles(self, model: Any) -> bool:
        return (
            self.config.enabled
            and isinstance(model, str)
            and model.strip() in self.config.aliases
        )

    def candidates(
        self,
        profiles: Iterable[Any],
        *,
        needs_vision: bool,
        available: Callable[[Any], bool],
    ) -> list[RouteCandidate]:
        allowed = set(self.config.candidates)
        result: list[RouteCandidate] = []
        for profile in profiles:
            if allowed and profile.id not in allowed:
                continue
            capabilities = tuple(getattr(profile, "capabilities", ()) or ("text",))
            if needs_vision and "vision" not in capabilities:
                continue
            if not available(profile):
                continue
            result.append(
                RouteCandidate(
                    profile_id=profile.id,
                    alias=profile.alias,
                    description=describe_profile(
                        profile, self.config.descriptions.get(profile.id)
                    ),
                    capabilities=capabilities,
                )
            )
        return result

    def decide(
        self,
        payload: Mapping[str, Any],
        profiles: Iterable[Any],
        *,
        active_profile: str | None,
        available: Callable[[Any], bool],
    ) -> RouteDecision:
        state, needs_vision = request_state(
            payload, max_chars=self.config.max_state_chars
        )
        candidates = self.candidates(
            profiles, needs_vision=needs_vision, available=available
        )
        if not candidates:
            return self._fallback(
                "no candidate profile can serve this request",
                active_profile,
                candidates,
            )
        if len(candidates) == 1:
            return RouteDecision(candidates[0].profile_id, "only_candidate", 1.0)
        question = {
            ROUTE_QUESTION: {
                "type": "choice",
                "instructions": self.config.instructions,
                "criteria": {
                    candidate.profile_id: candidate.description
                    for candidate in candidates
                },
            }
        }
        try:
            answers = self.client.decide(state or None, question)
            probabilities = choice_probabilities(answers.get(ROUTE_QUESTION))
        except System1Error as error:
            return self._fallback(str(error), active_profile, candidates)
        ids = {candidate.profile_id for candidate in candidates}
        probabilities = {
            label: value for label, value in probabilities.items() if label in ids
        }
        if not probabilities:
            return self._fallback(
                "system1 answered with no known profile", active_profile, candidates
            )
        best = max(probabilities, key=lambda label: probabilities[label])
        confidence = probabilities[best]
        if confidence < self.config.min_confidence:
            decision = self._fallback(
                f"low confidence {confidence:.2f} for {best}",
                active_profile,
                candidates,
            )
            return RouteDecision(
                decision.profile_id, decision.reason, confidence, probabilities
            )
        if (
            active_profile in probabilities
            and best != active_profile
            and confidence - probabilities[active_profile] < self.config.switch_margin
        ):
            return RouteDecision(
                active_profile,
                "kept_active_within_switch_margin",
                probabilities[active_profile],
                probabilities,
            )
        return RouteDecision(best, "system1", confidence, probabilities)

    def _fallback(
        self,
        reason: str,
        active_profile: str | None,
        candidates: Sequence[RouteCandidate],
    ) -> RouteDecision:
        fallback = self.config.fallback
        ids = [candidate.profile_id for candidate in candidates]
        if fallback == "error":
            return RouteDecision(None, reason)
        if fallback == "active":
            if active_profile is not None and (not ids or active_profile in ids):
                return RouteDecision(active_profile, f"fallback_active: {reason}")
            if ids:
                return RouteDecision(ids[0], f"fallback_first_candidate: {reason}")
            return RouteDecision(None, reason)
        return RouteDecision(fallback, f"fallback_profile: {reason}")


class System1Sidecar:
    """Install and start the always-resident decision backend unit."""

    def __init__(
        self,
        config: System1Config,
        *,
        command_runner: Callable[[tuple[str, ...]], Any],
        systemd_dir: Path = SYSTEMD_USER_DIR,
        unit_name: str = SYSTEM1_UNIT_NAME,
    ) -> None:
        self.config = config
        self.command_runner = command_runner
        self.systemd_dir = Path(systemd_dir)
        self.unit_name = unit_name

    @property
    def unit_path(self) -> Path:
        return self.systemd_dir / self.unit_name

    def command(self) -> tuple[str, ...]:
        config = self.config
        if config.server_binary is None or config.model_path is None:
            raise System1ConfigError("system1 backend is not managed by the broker")
        return (
            str(config.server_binary),
            "-m",
            str(config.model_path),
            "--alias",
            config.model,
            "--host",
            config.listen_host,
            "--port",
            str(config.listen_port),
            "--ctx-size",
            str(config.context_size),
            "--n-gpu-layers",
            config.gpu_layers,
            "--parallel",
            "1",
            *config.extra_args,
        )

    def render_unit(self) -> str:
        lines = [
            "[Unit]",
            f"Description=llama-broker system 1 decision backend: {self.config.model}",
            "After=network-online.target",
            "Wants=network-online.target",
            "X-Llama-Broker-System1=1",
            "",
            "[Service]",
            "Type=exec",
            f"MemoryMax={self.config.memory_max_mib}M",
            "KillSignal=SIGINT",
            "TimeoutStopSec=20s",
            "KillMode=control-group",
            "ExecStart="
            + " ".join(_unit_quote(word, escape_dollar=True) for word in self.command()),
        ]
        lines.extend(
            f"Environment={_unit_quote(f'{name}={value}')}"
            for name, value in sorted(self.config.environment.items())
        )
        lines.extend(
            [
                "Restart=on-failure",
                "RestartSec=5s",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        return "\n".join(lines)

    def verify_artifacts(self) -> None:
        config = self.config
        for label, path in (
            ("server_binary", config.server_binary),
            ("model_path", config.model_path),
        ):
            if path is None or not path.is_file():
                raise System1ConfigError(f"system1.{label} does not exist: {path}")
        if not os.access(config.server_binary, os.X_OK):  # type: ignore[arg-type]
            raise System1ConfigError(
                f"system1.server_binary is not executable: {config.server_binary}"
            )

    def ensure_running(self) -> None:
        """Write the unit and start it; a running unit is left untouched."""

        self.verify_artifacts()
        self.systemd_dir.mkdir(parents=True, exist_ok=True)
        text = self.render_unit()
        changed = not self.unit_path.is_file() or self.unit_path.read_text() != text
        if changed:
            temporary = self.unit_path.with_suffix(".service.tmp")
            temporary.write_text(text)
            os.replace(temporary, self.unit_path)
            self.command_runner(("systemctl", "--user", "daemon-reload"))
            self.command_runner(("systemctl", "--user", "restart", self.unit_name))
        else:
            self.command_runner(("systemctl", "--user", "start", self.unit_name))

    def stop(self) -> None:
        self.command_runner(("systemctl", "--user", "stop", self.unit_name))


class System1Service:
    """What the broker application holds when system 1 is enabled."""

    def __init__(
        self,
        config: System1Config,
        *,
        client: System1Client | None = None,
        sidecar: System1Sidecar | None = None,
    ) -> None:
        self.config = config
        self.client = client or System1Client(config.url, timeout=config.timeout)
        self.router = System1Router(config.routing, self.client)
        self.sidecar = sidecar

    def routing_aliases(self) -> tuple[str, ...]:
        return self.config.routing.aliases if self.config.routing.enabled else ()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": self.config.model,
            "url": self.config.url,
            "managed": self.config.managed,
            "unit": self.sidecar.unit_name if self.sidecar else None,
            "ready": self.client.healthy(),
            "routing": {
                "enabled": self.config.routing.enabled,
                "aliases": list(self.routing_aliases()),
            },
        }
