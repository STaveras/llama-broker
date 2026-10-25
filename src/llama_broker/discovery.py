#!/usr/bin/env python3
"""Local backend discovery with port sweep and dirty tracking.

Scans candidate ports for OpenAI-compatible servers, tracks which ports
have been checked recently, and supports optional broad port sweeps.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in clean minimal installs
    requests = None  # type: ignore[assignment]

# Known candidate ports for local LLM servers
DEFAULT_CANDIDATES = [
    "http://127.0.0.1:10000/v1",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1:1234/v1",
    "http://127.0.0.1:5000/v1",
]

# How long a port check is considered "fresh"
DIRTY_THRESHOLD_SECONDS = 300  # 5 minutes

# State file location
STATE_DIR = Path.home() / ".config" / "llm-broker"
STATE_FILE = STATE_DIR / "port_state.json"


@dataclass
class DiscoveredModel:
    """A model found on a discovered endpoint."""
    id: str
    context_length: Optional[int] = None
    parallel: Optional[int] = None
    slot_context_window: Optional[int] = None
    owned_by: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredEndpoint:
    """An endpoint that responded to probing."""
    base_url: str
    port: int
    models: List[DiscoveredModel] = field(default_factory=list)
    total_slots: Optional[int] = None
    n_ctx: Optional[int] = None
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_props: Dict[str, Any] = field(default_factory=dict)
    raw_slots: Any = None

    @property
    def label(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def default_model(self) -> str:
        return self.models[0].id if self.models else "unknown"

    @property
    def slot_context_window(self) -> Optional[int]:
        if self.models:
            return self.models[0].slot_context_window
        return None


@dataclass
class DiscoveryResult:
    """Result of a discovery scan."""
    endpoints: List[DiscoveredEndpoint] = field(default_factory=list)
    checked_ports: List[int] = field(default_factory=list)
    failed_ports: List[int] = field(default_factory=list)
    new_ports: List[int] = field(default_factory=list)
    duration_ms: float = 0.0


def _normalize_url(url: str) -> str:
    """Normalize a URL to have /v1 suffix."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _port_from_url(url: str) -> int:
    """Extract port from URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _probe_endpoint(base_url: str, timeout: float = 2.0) -> Optional[DiscoveredEndpoint]:
    """Probe a single endpoint for models and metadata."""
    if requests is None:
        raise RuntimeError(
            "backend discovery requires the optional 'discovery' dependency; "
            "install llama-broker[discovery]"
        )
    normalized = _normalize_url(base_url)
    server_root = normalized.rsplit("/v1", 1)[0] if normalized.endswith("/v1") else normalized
    port = _port_from_url(normalized)

    try:
        # Probe /v1/models
        models_resp = requests.get(
            f"{normalized}/models",
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        models_resp.raise_for_status()
        models_data = models_resp.json()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError):
        return None

    # Probe /props (optional)
    props_data = {}
    try:
        props_resp = requests.get(f"{server_root}/props", timeout=timeout)
        if props_resp.status_code == 200:
            props_data = props_resp.json()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        pass

    # Probe /slots (optional)
    slots_data = None
    try:
        slots_resp = requests.get(f"{server_root}/slots", timeout=timeout)
        if slots_resp.status_code == 200:
            slots_data = slots_resp.json()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        pass

    # Extract total slots from props
    total_slots = None
    if isinstance(props_data, dict):
        total_slots = props_data.get("total_slots")
        if total_slots is None:
            default_settings = props_data.get("default_generation_settings", {})
            if isinstance(default_settings, dict):
                total_slots = default_settings.get("n_parallel")

    # Extract per-slot context from slots
    slot_ctx = None
    if isinstance(slots_data, list) and slots_data:
        for slot in slots_data:
            if isinstance(slot, dict):
                ctx = slot.get("n_ctx")
                if ctx and isinstance(ctx, (int, float)) and ctx > 0:
                    slot_ctx = int(ctx)
                    break

    # Parse models
    models_list = models_data.get("data", []) if isinstance(models_data, dict) else []
    discovered_models = []
    for item in models_list:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id", "")
        if not model_id:
            continue

        # Extract context_length from model or meta
        ctx = item.get("context_length") or item.get("max_model_len") or item.get("max_tokens")
        if ctx is None and isinstance(item.get("meta"), dict):
            ctx = item["meta"].get("n_ctx_train")

        # Extract parallel from model or props
        par = item.get("parallel") or total_slots

        # Calculate slot context
        model_slot_ctx = item.get("slot_context_window")
        if (
            isinstance(model_slot_ctx, int)
            and not isinstance(model_slot_ctx, bool)
            and model_slot_ctx > 0
        ):
            s_ctx = model_slot_ctx
        elif slot_ctx:
            s_ctx = slot_ctx
        elif ctx and par and par > 1:
            s_ctx = ctx // par
        else:
            s_ctx = ctx

        discovered_models.append(DiscoveredModel(
            id=model_id,
            context_length=ctx if isinstance(ctx, int) else None,
            parallel=par if isinstance(par, int) else None,
            slot_context_window=s_ctx if isinstance(s_ctx, int) else None,
            owned_by=item.get("owned_by", ""),
            raw=item,
        ))

    if not discovered_models:
        return None

    return DiscoveredEndpoint(
        base_url=normalized,
        port=port,
        models=discovered_models,
        total_slots=total_slots if isinstance(total_slots, int) else None,
        n_ctx=slot_ctx if isinstance(slot_ctx, int) else None,
        raw_props=props_data,
        raw_slots=slots_data,
    )


def _load_state() -> Dict[str, Any]:
    """Load port discovery state from disk."""
    if not STATE_FILE.exists():
        return {"checked": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"checked": {}}


def _save_state(state: Dict[str, Any]) -> None:
    """Save port discovery state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _is_dirty(port_state: Dict[str, Any], threshold_seconds: int = DIRTY_THRESHOLD_SECONDS) -> bool:
    """Check if a port state entry is stale (dirty)."""
    last_checked = port_state.get("last_checked")
    if not last_checked:
        return True
    try:
        last_dt = datetime.fromisoformat(last_checked)
        return (datetime.now() - last_dt).total_seconds() > threshold_seconds
    except (ValueError, TypeError):
        return True


def discover_backends(
    candidates: Optional[Sequence[str]] = None,
    sweep_range: Optional[str] = None,
    dirty_only: bool = False,
    timeout: float = 2.0,
    max_workers: int = 8,
) -> DiscoveryResult:
    """Discover local OpenAI-compatible backends.

    Args:
        candidates: List of known candidate URLs. Uses DEFAULT_CANDIDATES if None.
        sweep_range: Optional port range to scan (e.g. "8000-12000").
        dirty_only: Only recheck ports that haven't been checked recently.
        timeout: HTTP timeout per probe in seconds.
        max_workers: Max concurrent probe threads.

    Returns:
        DiscoveryResult with found endpoints and metadata.
    """
    start_time = time.monotonic()
    state = _load_state()
    checked = state.get("checked", {})

    # Build list of ports to probe
    urls_to_probe: List[str] = []

    # Add candidates
    for url in (candidates or DEFAULT_CANDIDATES):
        normalized = _normalize_url(url)
        port = _port_from_url(normalized)

        if dirty_only and port in checked and not _is_dirty(checked[str(port)]):
            continue  # Skip clean ports

        urls_to_probe.append(normalized)

    # Add sweep range if specified
    new_ports: List[int] = []
    if sweep_range:
        parts = sweep_range.split("-")
        if len(parts) == 2:
            try:
                start_port, end_port = int(parts[0]), int(parts[1])
                for port in range(start_port, end_port + 1):
                    port_str = str(port)
                    if port_str not in checked:
                        new_ports.append(port)
                        urls_to_probe.append(f"http://127.0.0.1:{port}/v1")
            except ValueError:
                pass

    # Probe endpoints
    endpoints: List[DiscoveredEndpoint] = []
    failed: List[int] = []
    checked_ports: List[int] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls_to_probe) or 1)) as executor:
        future_map = {
            executor.submit(_probe_endpoint, url, timeout): url
            for url in urls_to_probe
        }
        for future in as_completed(future_map):
            url = future_map[future]
            port = _port_from_url(url)
            checked_ports.append(port)

            try:
                result = future.result()
                if result:
                    endpoints.append(result)
                    # Update state
                    checked[str(port)] = {
                        "last_checked": datetime.now().isoformat(),
                        "status": "alive",
                        "models": [m.id for m in result.models],
                        "base_url": result.base_url,
                    }
                else:
                    failed.append(port)
                    checked[str(port)] = {
                        "last_checked": datetime.now().isoformat(),
                        "status": "dead",
                        "models": [],
                    }
            except Exception:
                failed.append(port)
                checked[str(port)] = {
                    "last_checked": datetime.now().isoformat(),
                    "status": "error",
                    "models": [],
                }

    # Save updated state
    state["checked"] = checked
    _save_state(state)

    duration_ms = (time.monotonic() - start_time) * 1000

    return DiscoveryResult(
        endpoints=endpoints,
        checked_ports=checked_ports,
        failed_ports=failed,
        new_ports=new_ports,
        duration_ms=duration_ms,
    )


def get_state_summary() -> Dict[str, Any]:
    """Get a summary of the port discovery state."""
    state = _load_state()
    checked = state.get("checked", {})

    alive = []
    dead = []
    unknown = []

    for port_str, info in checked.items():
        status = info.get("status", "unknown")
        entry = {
            "port": int(port_str) if port_str.isdigit() else port_str,
            "status": status,
            "models": info.get("models", []),
            "last_checked": info.get("last_checked"),
        }
        if status == "alive":
            alive.append(entry)
        elif status == "dead":
            dead.append(entry)
        else:
            unknown.append(entry)

    return {
        "alive": alive,
        "dead": dead,
        "unknown": unknown,
        "total_checked": len(checked),
    }
