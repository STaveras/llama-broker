#!/usr/bin/env python3
"""Interactive terminal and curses profile configuration front ends."""

from __future__ import annotations

import copy
import json
import os
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .discovery import discover_backends
from .profile_config import (
    SIMPLE_FIELDS,
    apply_simple_fields,
    clone_record,
    discover_local_ggufs,
    draft_from_endpoint,
    draft_from_gguf,
    find_profile_record,
    load_manifest_source,
    merge_manifest_drafts,
    semantic_diff,
    sha256_file,
    simple_projection,
    validate_manifest,
    write_manifest_atomically,
)
from .runtime_builds import (
    apply_runtime_build,
    catalog_from_manifest,
    infer_runtime_build,
    resolve_runtime_build,
    runtime_build_label,
    serving_backend_label,
)
from .runtimes import get_runtime_info, runtime_names


ApplyCallback = Callable[[dict[str, Any], set[str]], Any]


@dataclass(frozen=True)
class EditorField:
    key: str
    label: str
    kind: str
    advanced: bool = False
    options: tuple[str, ...] = ()


EDITOR_FIELDS: tuple[EditorField, ...] = (
    EditorField("display_name", "Display name", "text"),
    EditorField("alias", "Primary alias", "text"),
    EditorField("aliases", "Additional aliases", "csv"),
    EditorField("role", "Role", "select", options=("persistent", "transient")),
    EditorField("runtime_family", "Runtime family", "select"),
    EditorField("runtime_build", "Backend build", "select"),
    EditorField("runtime", "Serving runtime", "select", options=runtime_names()),
    EditorField("capabilities", "Capabilities", "multi", options=("text", "vision", "tools", "reasoning")),
    EditorField("context_size", "Context size", "int"),
    EditorField("parallel", "Parallel slots", "select", options=("1", "2")),
    EditorField("batch_size", "Batch size", "int"),
    EditorField("ubatch_size", "Micro-batch size", "int"),
    EditorField(
        "cache_type_k",
        "KV cache K",
        "select",
        options=("f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "turbo4", "kvarn4"),
    ),
    EditorField(
        "cache_type_v",
        "KV cache V",
        "select",
        options=("f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "turbo4", "kvarn4"),
    ),
    EditorField("flash_attention", "Flash attention", "select", options=("auto", "on", "off")),
    EditorField("gpu_layers", "GPU layers", "select", options=("auto", "all", "1", "2", "4", "8", "16", "32")),
    EditorField("mmproj_path", "Vision projector path", "path"),
    EditorField("tensor_parallel_size", "vLLM tensor parallel size", "int", advanced=True),
    EditorField("gpu_memory_utilization", "vLLM GPU memory utilization", "text", advanced=True),
    EditorField("kv_cache_dtype", "vLLM KV cache dtype", "text", advanced=True),
    EditorField("tool_call_parser", "vLLM tool-call parser", "text", advanced=True),
    EditorField("tool_parser_plugin", "vLLM tool parser plugin", "path", advanced=True),
    EditorField("chat_template", "External chat template", "path", advanced=True),
    EditorField("limit_mm_per_prompt", "vLLM media limits (JSON)", "json", advanced=True),
    EditorField("allowed_local_media_path", "vLLM local media path", "path", advanced=True),
    EditorField("server_binary", "Runtime binary", "path", advanced=True),
    EditorField("strata_artifacts_manifest_path", "Strata artifact manifest", "path", advanced=True),
    EditorField("strata_launch_config_path", "Strata launch config", "path", advanced=True),
    EditorField("model_path", "Model file", "path", advanced=True),
    EditorField("model_draft", "Draft model file", "path", advanced=True),
    EditorField(
        "spec_type",
        "Speculative decoding",
        "select",
        advanced=True,
        options=("off", "mtp", "draft-mtp", "nextn"),
    ),
    EditorField("draft_max", "Draft max tokens", "int", advanced=True),
    EditorField("draft_min", "Draft min tokens", "int", advanced=True),
    EditorField("native_vision", "Native vision", "toggle", advanced=True),
    EditorField("fit", "Auto-fit to VRAM", "toggle", advanced=True),
    EditorField("fit_target_mib", "Fit target MiB", "int", advanced=True),
    EditorField("fit_ctx", "Fit context size", "int", advanced=True),
    EditorField("ctx_checkpoints", "Fit context checkpoints", "int", advanced=True),
    EditorField("cache_ram_mib", "Fit cache RAM MiB", "int", advanced=True),
    EditorField("startup_timeout", "Startup timeout seconds", "int", advanced=True),
    EditorField("restore_profile", "Restore profile", "text", advanced=True),
    EditorField("environment", "Environment variables (JSON)", "json", advanced=True),
    EditorField("extra_args", "Extra server args (JSON)", "json_list", advanced=True),
    EditorField("safety", "Safety budgets (JSON)", "json", advanced=True),
)


def _default_text(input_fn: Callable[[str], str], prompt: str, current: Any = "") -> str:
    suffix = f" [{current}]" if current not in (None, "") else ""
    value = input_fn(f"{prompt}{suffix}: ").strip()
    return value if value else str(current or "")


def _default_int(input_fn: Callable[[str], str], prompt: str, current: int) -> int:
    while True:
        raw = _default_text(input_fn, prompt, current)
        try:
            return int(raw)
        except ValueError:
            print("Enter a positive integer.")


def _yes(input_fn: Callable[[str], str], prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input_fn(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "on", "true"}


def _choose_local_gguf(
    input_fn: Callable[[str], str], output: Callable[[str], None],
) -> Path:
    candidates = discover_local_ggufs()
    if candidates:
        output("Local GGUF files:")
        for index, candidate in enumerate(candidates[:50], 1):
            output(f"  {index:>2}. {candidate}")
    raw = input_fn("GGUF path (or number from the list): ").strip()
    if raw.isdigit() and candidates and 1 <= int(raw) <= len(candidates):
        return candidates[int(raw) - 1]
    return Path(raw).expanduser()


def _find_discovered_model(result: Any, selector: str | None) -> tuple[Any, Any] | None:
    if not selector:
        return None
    wanted = selector.casefold()
    for endpoint in getattr(result, "endpoints", []):
        for model in getattr(endpoint, "models", []):
            if model.id.casefold() == wanted or wanted in model.id.casefold():
                return endpoint, model
    return None


def _edit_simple_fields(
    record: Mapping[str, Any],
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> dict[str, Any]:
    """Guided simple screen shared conceptually with the curses editor."""

    current = simple_projection(record)
    values: dict[str, Any] = {}
    values["display_name"] = _default_text(input_fn, "Display name", current.get("display_name", ""))
    values["alias"] = _default_text(input_fn, "Alias", current.get("alias", ""))
    values["role"] = _default_text(input_fn, "Role (persistent/transient)", current.get("role", "persistent"))
    values["capabilities"] = _default_text(
        input_fn, "Capabilities (comma-separated)", ",".join(current.get("capabilities", ["text"]))
    )
    values["context_size"] = _default_int(input_fn, "Aggregate context size", int(current.get("context_size", 32768)))
    values["parallel"] = _default_int(input_fn, "Parallel slots", int(current.get("parallel", 1)))
    if str(current.get("runtime", "llamacpp")).strip().lower() == "llamacpp":
        values["batch_size"] = _default_int(input_fn, "Batch size", int(current.get("batch_size", 256)))
        values["ubatch_size"] = _default_int(input_fn, "Micro-batch size", int(current.get("ubatch_size", 64)))
        values["cache_type_k"] = _default_text(input_fn, "KV cache K", current.get("cache_type_k", "f16"))
        values["cache_type_v"] = _default_text(input_fn, "KV cache V", current.get("cache_type_v", "f16"))
        values["gpu_layers"] = _default_text(input_fn, "GPU layers (auto/all/count)", current.get("gpu_layers", "auto"))
        values["mmproj_path"] = _default_text(input_fn, "Vision projector path (blank disables)", current.get("mmproj_path", ""))
    output("Simple serving fields captured; use advanced mode for hashes, safety, and environment.")
    return apply_simple_fields(record, values)


def _edit_advanced_fields(
    record: dict[str, Any], input_fn: Callable[[str], str], output: Callable[[str], None]
) -> dict[str, Any]:
    draft = clone_record(record)
    model_path = _default_text(input_fn, "Model path", draft.get("model_path", ""))
    server_binary = _default_text(input_fn, "llama-server path", draft.get("server_binary", ""))
    if model_path:
        from .profile_config import sha256_file

        draft["model_path"] = str(Path(model_path).expanduser().resolve())
        draft["model_sha256"] = sha256_file(Path(model_path).expanduser())
    if server_binary:
        from .profile_config import sha256_file

        draft["server_binary"] = str(Path(server_binary).expanduser().resolve())
        draft["server_sha256"] = sha256_file(Path(server_binary).expanduser())
    if draft.get("mmproj_path"):
        from .profile_config import sha256_file

        projector = Path(str(draft["mmproj_path"])).expanduser()
        draft["mmproj_path"] = str(projector.resolve())
        draft["mmproj_sha256"] = sha256_file(projector)
    raw_environment = _default_text(
        input_fn,
        "Environment JSON (CUDA/KVarN safety variables)",
        json.dumps(draft.get("environment", {}), sort_keys=True),
    )
    try:
        environment = json.loads(raw_environment)
    except json.JSONDecodeError as error:
        raise ValueError(f"environment must be valid JSON: {error}") from error
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("environment must be a JSON object of string values")
    draft["environment"] = environment
    raw_extra_args = _default_text(
        input_fn,
        "Extra args JSON list",
        json.dumps(draft.get("extra_args", [])),
    )
    try:
        extra_args = json.loads(raw_extra_args)
    except json.JSONDecodeError as error:
        raise ValueError(f"extra_args must be valid JSON: {error}") from error
    if not isinstance(extra_args, list) or not all(isinstance(value, str) for value in extra_args):
        raise ValueError("extra_args must be a JSON string list")
    draft["extra_args"] = extra_args
    raw_safety = _default_text(
        input_fn,
        "Safety budgets JSON",
        json.dumps(draft.get("safety", {}), sort_keys=True),
    )
    try:
        safety = json.loads(raw_safety)
    except json.JSONDecodeError as error:
        raise ValueError(f"safety must be valid JSON: {error}") from error
    if not isinstance(safety, dict):
        raise ValueError("safety must be a JSON object")
    draft["safety"] = safety
    draft["startup_timeout"] = _default_int(
        input_fn, "Startup timeout seconds", int(draft.get("startup_timeout", 240))
    )
    if _yes(input_fn, "Enable auto-fit (fit controls are advanced)", bool(draft.get("fit"))):
        draft["fit"] = True
        draft["fit_target_mib"] = _default_int(input_fn, "Fit target MiB", int(draft.get("fit_target_mib", 8192)))
        draft["fit_ctx"] = _default_int(input_fn, "Fit context", int(draft.get("fit_ctx", draft.get("context_size", 32768))))
    else:
        for key in ("fit", "fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"):
            draft.pop(key, None)
    output("Advanced fields captured and file hashes recomputed.")
    return draft


def run_guided_wizard(
    profiles_path: Path | str,
    selector: str | None = None,
    *,
    apply_callback: ApplyCallback | None = None,
    discover_fn: Callable[..., Any] = discover_backends,
    input_fn: Callable[[str], str] | None = None,
    output: Callable[[str], None] | None = None,
    preview: bool = False,
) -> int:
    """Run a dependency-free guided wizard and optionally apply its draft."""

    path = Path(profiles_path)
    if input_fn is None:
        input_fn = input
    if output is None:
        output = print
    manifest = load_manifest_source(path, allow_missing_artifacts=True)
    if selector is None:
        existing = manifest.get("profiles", [])
        output("Configured profiles:")
        for record in existing if isinstance(existing, list) else []:
            if isinstance(record, dict):
                output(f"  {record.get('id')} ({record.get('alias')})")
        choice = input_fn("Profile id/alias to edit, or 'new' for import [new]: ").strip()
        if choice and choice.casefold() != "new":
            selector = choice
    found = find_profile_record(manifest, selector) if selector else None
    is_new = found is None
    endpoint_model = None

    if is_new:
        try:
            result = discover_fn(timeout=2.0)
        except Exception as error:
            output(f"Endpoint discovery unavailable ({error}); continuing with local GGUF import.")
            result = type("DiscoveryFallback", (), {"endpoints": []})()
        endpoint_model = _find_discovered_model(result, selector)
        if endpoint_model:
            endpoint, model = endpoint_model
            output(f"Discovered {model.id} at {endpoint.label}; adopting it as a broker-owned profile.")
            model_path = _choose_local_gguf(input_fn, output)
            draft = draft_from_endpoint(manifest, endpoint, model, model_path=model_path)
        else:
            if selector:
                output(f"No endpoint matched {selector!r}; using manual GGUF import.")
            model_path = _choose_local_gguf(input_fn, output)
            draft = draft_from_gguf(manifest, model_path, profile_id=selector)
    else:
        assert found is not None
        draft = clone_record(found[1])
        output(f"Editing {draft.get('id')} ({draft.get('display_name', draft.get('alias'))})")

    draft = _edit_simple_fields(draft, input_fn, output)
    if _yes(input_fn, "Edit advanced paths, safety, fit, environment, or extra args", False):
        draft = _edit_advanced_fields(draft, input_fn, output)

    updated = copy.deepcopy(manifest)
    profiles = updated.setdefault("profiles", [])
    if is_new:
        profiles.append(draft)
        changed_ids = {str(draft["id"])}
    else:
        assert found is not None
        profiles[found[0]] = draft
        changed_ids = {str(draft["id"])}

    # Preview and Apply use the same production parser; an attractive draft
    # must not be shown as ready if it would fail the broker's safety gates.
    validate_manifest(updated, directory=path.parent)
    changed = semantic_diff(manifest, updated)
    if not changed:
        output("No changes.")
        return 0
    output("Manifest changes: " + ", ".join(changed))
    if preview:
        output("Preview only; nothing was written.")
        return 0
    if not _yes(input_fn, "Apply and restart the active broker profile if required", True):
        output("Cancelled; no changes written.")
        return 0
    if apply_callback is not None:
        apply_callback(updated, changed_ids)
    else:
        write_manifest_atomically(path, updated)
    output("Profile configuration applied.")
    return 0


_SPECULATIVE_FLAG_BY_FIELD: Mapping[str, str] = {
    "spec_type": "--spec-type",
    "draft_max": "--draft-max",
    "draft_min": "--draft-min",
}
_DRAFT_MAX_ALIASES = ("--draft-max", "--spec-draft-n-max")
_SPECULATIVE_EDITOR_FIELDS = frozenset(_SPECULATIVE_FLAG_BY_FIELD)
_SPECULATIVE_COUNT_FIELDS = _SPECULATIVE_EDITOR_FIELDS - {"spec_type"}
_MANAGED_SPECULATIVE_FLAGS = frozenset(
    ("--spec-type", "--draft-max", "--spec-draft-n-max", "--draft-min")
)
# Flags the editor derives from structured fields instead of letting them be
# hand-edited inside the raw extra_args JSON.
_MANAGED_FLAG_GROUPS = _MANAGED_SPECULATIVE_FLAGS | {"--reasoning"}


def _extra_arg_value(extra_args: Any, flag: str) -> str:
    """Read a flag value without changing the raw extra-args representation."""
    if not isinstance(extra_args, list):
        return ""
    for index, token in enumerate(extra_args):
        if token == flag and index + 1 < len(extra_args):
            value = extra_args[index + 1]
            if isinstance(value, str) and not value.startswith("--"):
                return value
            return ""
    return ""


def _draft_max_value(extra_args: Any) -> str:
    for flag in _DRAFT_MAX_ALIASES:
        value = _extra_arg_value(extra_args, flag)
        if value:
            return value
    return ""


def _without_managed_speculative_args(extra_args: Any) -> list[str]:
    if not isinstance(extra_args, list):
        return []
    result: list[str] = []
    index = 0
    while index < len(extra_args):
        token = extra_args[index]
        if token in _MANAGED_SPECULATIVE_FLAGS:
            index += 1
            if index < len(extra_args) and not extra_args[index].startswith("--"):
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def _without_managed_args(extra_args: Any) -> list[str]:
    """Strip every flag group the editor derives from structured fields."""

    if not isinstance(extra_args, list):
        return []
    result: list[str] = []
    index = 0
    while index < len(extra_args):
        token = extra_args[index]
        if token in _MANAGED_FLAG_GROUPS:
            index += 1
            if index < len(extra_args) and not extra_args[index].startswith("--"):
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def _sync_reasoning_flag(
    draft: dict[str, Any], extra_args: list[str], capabilities: list[str]
) -> list[str]:
    """Derive the --reasoning launch flag from the reasoning capability.

    The flag is managed state: it is owned by the capability toggle, so the
    user cannot enable the capability while leaving the launch argument off
    (and vice versa).
    """

    enabled = "reasoning" in capabilities
    return _replace_extra_arg(
        extra_args, "--reasoning", "on" if enabled else None
    )


def _replace_extra_arg(
    extra_args: Any,
    flag: str,
    value: str | None,
) -> list[str]:
    """Replace or remove one flag/value pair while preserving other arguments."""
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) and item for item in extra_args
    ):
        raise ValueError("extra_args must be a JSON string list")

    updated: list[str] = []
    replaced = False
    index = 0
    while index < len(extra_args):
        token = extra_args[index]
        if token != flag:
            updated.append(token)
            index += 1
            continue

        if value is not None and not replaced:
            updated.extend((flag, value))
            replaced = True
        index += 1
        if index < len(extra_args) and not extra_args[index].startswith("--"):
            index += 1

    if value is not None and not replaced:
        updated.extend((flag, value))
    return updated


class TUIModel:
    """Testable state model for the field-oriented profile editor."""

    _INTEGER_FIELDS = {
        "context_size",
        "batch_size",
        "ubatch_size",
        "tensor_parallel_size",
        "fit_target_mib",
        "fit_ctx",
        "ctx_checkpoints",
        "cache_ram_mib",
        "startup_timeout",
    }
    _OPTIONAL_INTEGER_FIELDS = {"ctx_checkpoints", "cache_ram_mib"}
    _BOOLEAN_FIELDS = {"native_vision", "fit"}
    _PATH_FIELDS = {
        "server_binary",
        "model_path",
        "mmproj_path",
        "model_draft",
        "chat_template",
        "strata_artifacts_manifest_path",
        "strata_launch_config_path",
    }

    def __init__(self, manifest: Mapping[str, Any]):
        self.manifest = copy.deepcopy(dict(manifest))
        self.base_manifest = copy.deepcopy(self.manifest)
        self._catalog = tuple(
            sorted(
                catalog_from_manifest(self.manifest, allow_missing_artifacts=True),
                key=lambda build: build.id,
            )
        )
        self.index = 0
        self.field_index = 0
        self.advanced = False
        self.dirty = False
        self._manual_profile_ids: set[str] = set()
        self._valid_build_cache: tuple[Any, ...] | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        raw = self.manifest.setdefault("profiles", [])
        if not isinstance(raw, list):
            raise ValueError("profiles must be a list")
        return raw

    @property
    def current(self) -> dict[str, Any] | None:
        records = self.records
        if not records:
            return None
        self.index = max(0, min(self.index, len(records) - 1))
        return records[self.index]

    def _current_runtime_build(self, profile: Mapping[str, Any]) -> Any:
        profile_id = str(profile.get("id", ""))
        if profile_id in self._manual_profile_ids:
            return None
        selected = profile.get("runtime_build")
        if selected:
            wanted = str(selected).strip().casefold()
            for build in self._catalog:
                if build.id.casefold() == wanted and build.runtime == str(
                    profile.get("runtime", "llamacpp")
                ).strip().lower():
                    return build
        inferred = infer_runtime_build(
            self.manifest, profile, allow_missing_artifacts=True
        )
        if inferred is not None:
            return next(
                (build for build in self._catalog if build.id == inferred.id), inferred
            )
        return None

    def _valid_builds(self) -> tuple[Any, ...]:
        if self._valid_build_cache is not None:
            return self._valid_build_cache
        valid: list[Any] = []
        for build in self._catalog:
            path = Path(build.server_binary)
            if not path.is_file() or not os.access(path, os.X_OK):
                continue
            try:
                if sha256_file(path) != build.server_sha256:
                    continue
            except OSError:
                continue
            valid.append(build)
        self._valid_build_cache = tuple(valid)
        return self._valid_build_cache

    def _family_options(self) -> tuple[str, ...]:
        families = sorted({build.runtime for build in self._valid_builds()})
        if not families:
            return ("(no valid catalog builds available)", "(custom/manual)")
        return tuple(
            f"{family} — {sum(build.runtime == family for build in self._catalog)} "
            f"{'build' if sum(build.runtime == family for build in self._catalog) == 1 else 'builds'}"
            for family in families
        ) + ("(custom/manual)",)

    def _family_for_value(self, value: Any) -> str:
        selected = str(value).strip()
        if selected == "(custom/manual)":
            return selected
        if selected == "(no valid catalog builds available)":
            raise ValueError(selected)
        for option in self._family_options():
            if option == selected:
                return option.split(" — ", 1)[0]
        if any(build.runtime == selected for build in self._catalog):
            return selected
        return selected

    def _build_for_value(self, value: Any) -> Any | None:
        wanted = str(value).strip().casefold()
        first_line = wanted.splitlines()[0].strip() if wanted else ""
        for build in self._catalog:
            labels = {
                build.id.casefold(),
                runtime_build_label(build).casefold(),
                f"{build.display_name} — {build.id}".casefold(),
            }
            if wanted in labels or first_line in labels:
                return build
        return None

    def _build_option(self, build: Any) -> str:
        return (
            f"{build.display_name} — {build.id}\n"
            f"  runtime: {build.runtime}\n"
            f"  binary: {build.server_binary}\n"
            f"  sha256: {build.server_sha256}"
        )

    def _apply_build(self, draft: dict[str, Any], build: Any) -> dict[str, Any]:
        allowed_fields = get_runtime_info(build.runtime).editable_fields
        updated = {
            key: value
            for key, value in clone_record(draft).items()
            if key in allowed_fields
        }
        updated.update(
            {
                "runtime": build.runtime,
                "runtime_build": build.id,
                "server_binary": str(build.server_binary),
                "server_sha256": build.server_sha256,
                "environment": dict(build.environment),
            }
        )
        return updated

    @property
    def visible_fields(self) -> list[EditorField]:
        current = self.current or {}
        runtime = str(current.get("runtime", "llamacpp")).strip().lower()
        runtime_info = get_runtime_info(runtime)
        inferred = self._current_runtime_build(current)
        family = inferred.runtime if inferred is not None else "(custom/manual)"
        fields = []
        for field in EDITOR_FIELDS:
            if runtime == "strata" and field.key == "environment":
                continue
            if field.key == "runtime_family":
                field = replace(field, options=self._family_options())
            elif field.key == "runtime_build":
                if family == "(custom/manual)":
                    continue
                builds = tuple(build for build in self._catalog if build.runtime == family)
                field = replace(field, options=tuple(self._build_option(build) for build in builds))
            elif field.key == "runtime" and family != "(custom/manual)":
                continue
            elif field.key not in runtime_info.editable_fields and not (
                runtime == "llamacpp" and field.key in _SPECULATIVE_EDITOR_FIELDS
            ):
                continue
            if field.advanced and not self.advanced:
                continue
            if field.key in {"fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"}:
                if not current.get("fit"):
                    continue
            fields.append(field)
        return fields

    @property
    def current_field(self) -> EditorField | None:
        fields = self.visible_fields
        if not fields:
            return None
        self.field_index = max(0, min(self.field_index, len(fields) - 1))
        return fields[self.field_index]

    def move(self, delta: int) -> None:
        if self.records:
            self.index = (self.index + delta) % len(self.records)
            self.field_index = 0

    def move_field(self, delta: int) -> None:
        fields = self.visible_fields
        if fields:
            self.field_index = (self.field_index + delta) % len(fields)

    def toggle_advanced(self) -> None:
        self.advanced = not self.advanced
        self.field_index = 0

    def update_current(self, values: Mapping[str, Any]) -> dict[str, Any]:
        current = self.current
        if current is None:
            raise ValueError("no profile selected")
        updated = apply_simple_fields(current, values)
        self._publish(updated)
        return updated

    def field_value(self, key: str) -> Any:
        current = self.current or {}
        if key == "aliases":
            alias = str(current.get("alias", ""))
            aliases = [str(item) for item in current.get("aliases", []) if str(item) != alias]
            return ", ".join(aliases)
        if key in {"environment", "safety"}:
            return json.dumps(current.get(key, {}), sort_keys=True)
        if key == "extra_args":
            return json.dumps(_without_managed_args(current.get(key, [])))
        if key in _SPECULATIVE_FLAG_BY_FIELD:
            value = (
                _draft_max_value(current.get("extra_args", []))
                if key == "draft_max"
                else _extra_arg_value(current.get("extra_args", []), _SPECULATIVE_FLAG_BY_FIELD[key])
            )
            return value or ("off" if key == "spec_type" else "")
        if key == "runtime_family":
            inferred = self._current_runtime_build(current)
            if inferred is None:
                return "(custom/manual)"
            return next(
                option
                for option in self._family_options()
                if option.startswith(f"{inferred.runtime} — ")
            )
        if key == "runtime_build":
            inferred = self._current_runtime_build(current)
            if inferred is not None:
                return self._build_option(inferred)
            selected = current.get("runtime_build")
            if selected:
                for build in self._catalog:
                    if build.id == str(selected).strip():
                        return self._build_option(build)
            return "(custom/manual)"
        return current.get(key, "")

    def set_field(self, key: str, value: Any) -> dict[str, Any]:
        current = self.current
        if current is None:
            raise ValueError("no profile selected")
        current_value = self.field_value(key)
        if key not in {"runtime_build", *self._PATH_FIELDS} and (
            value == current_value
            or (
                isinstance(value, str)
                and str(value).strip() == str(current_value).strip()
            )
        ):
            return current
        draft = clone_record(current)

        if key in _SPECULATIVE_FLAG_BY_FIELD:
            raw = "" if value is None else str(value).strip()
            if key == "spec_type":
                if raw == "off":
                    replacement = None
                else:
                    requested = tuple(part.strip() for part in raw.split(","))
                    if not requested or any(
                        part not in {"mtp", "draft-mtp", "nextn"} for part in requested
                    ):
                        raise ValueError(f"unsupported speculative decoding type: {raw!r}")
                    replacement = ",".join(requested)
            else:
                replacement = str(int(raw)) if raw else None
            if key == "draft_max":
                draft["extra_args"] = _replace_extra_arg(
                    _replace_extra_arg(draft.get("extra_args", []), "--spec-draft-n-max", None),
                    "--draft-max",
                    replacement,
                )
            else:
                draft["extra_args"] = _replace_extra_arg(
                    draft.get("extra_args", []), _SPECULATIVE_FLAG_BY_FIELD[key], replacement
                )
        elif key == "extra_args":
            raw_args = json.loads(str(value)) if isinstance(value, str) else value
            if not isinstance(raw_args, list) or not all(isinstance(item, str) and item for item in raw_args):
                raise ValueError("extra_args must be a JSON string list")
            preserved: list[str] = []
            current_args = draft.get("extra_args", [])
            for index, token in enumerate(current_args):
                if isinstance(token, str) and token in _MANAGED_SPECULATIVE_FLAGS:
                    preserved.append(token)
                    if index + 1 < len(current_args) and not str(current_args[index + 1]).startswith("--"):
                        preserved.append(current_args[index + 1])
            raw_args = _without_managed_args(raw_args)
            capabilities = draft.get("capabilities", [])
            draft["extra_args"] = _sync_reasoning_flag(
                draft,
                [*raw_args, *preserved],
                capabilities if isinstance(capabilities, list) else [],
            )
        elif key == "runtime_family":
            selected = self._family_for_value(value)
            if not selected or selected == "(custom/manual)":
                draft.pop("runtime_build", None)
                profile_id = str(current.get("id", ""))
                if profile_id:
                    self._manual_profile_ids.add(profile_id)
            else:
                profile_id = str(current.get("id", ""))
                if profile_id:
                    self._manual_profile_ids.discard(profile_id)
                draft["runtime"] = selected
                current_build = self._current_runtime_build(current)
                if current_build is not None and current_build.runtime == selected:
                    draft = self._apply_build(draft, current_build)
                else:
                    fallback = next(
                        (build for build in self._valid_builds() if build.runtime == selected),
                        None,
                    )
                    if fallback is not None:
                        draft = self._apply_build(draft, fallback)
        elif key == "runtime_build":
            build = self._build_for_value(value)
            if build is None:
                raise ValueError(f"unknown runtime build: {value}")
            profile_id = str(current.get("id", ""))
            if profile_id:
                self._manual_profile_ids.discard(profile_id)
            draft = self._apply_build(draft, build)
        elif key == "runtime":
            draft["runtime"] = str(value).strip().lower()
            draft.pop("runtime_build", None)
            profile_id = str(current.get("id", ""))
            if profile_id:
                self._manual_profile_ids.add(profile_id)
        elif key in self._PATH_FIELDS:
            raw = str(value).strip()
            if not raw:
                if key == "mmproj_path":
                    draft = apply_simple_fields(draft, {key: ""})
                else:
                    draft.pop(key, None)
                    hash_key = {
                        "model_path": "model_sha256",
                        "server_binary": "server_sha256",
                        "strata_artifacts_manifest_path": "strata_artifacts_manifest_sha256",
                        "strata_launch_config_path": "strata_launch_config_sha256",
                    }.get(key)
                    if hash_key:
                        draft.pop(hash_key, None)
                self._publish(draft)
                return draft
            path = Path(raw).expanduser()
            runtime = str(current.get("runtime", "llamacpp")).strip().lower()
            if not path.is_file() and not (
                key == "model_path" and runtime == "vllm" and path.is_dir()
            ):
                raise ValueError(f"file does not exist: {path}")
            path = path.resolve()
            if key == "mmproj_path":
                draft = apply_simple_fields(draft, {key: str(path)})
            else:
                draft[key] = str(path)
                hash_key = {
                    "model_path": "model_sha256",
                    "server_binary": "server_sha256",
                    "strata_artifacts_manifest_path": "strata_artifacts_manifest_sha256",
                    "strata_launch_config_path": "strata_launch_config_sha256",
                }.get(key)
                if hash_key:
                    draft[hash_key] = sha256_file(path)
        elif key == "capabilities":
            raw_caps = value if isinstance(value, (list, tuple, set, frozenset)) else str(value)
            if isinstance(raw_caps, (list, tuple, set, frozenset)):
                capabilities = [str(item).strip() for item in raw_caps if str(item).strip()]
            else:
                capabilities = [item.strip() for item in raw_caps.split(",") if item.strip()]
            draft["capabilities"] = capabilities
            draft["extra_args"] = _sync_reasoning_flag(draft, draft.get("extra_args", []), capabilities)
        elif key in SIMPLE_FIELDS:
            draft = apply_simple_fields(draft, {key: value})
            if key == "alias":
                alias = str(value).strip()
                previous_alias = str(current.get("alias", ""))
                aliases = [
                    alias,
                    *[
                        str(item)
                        for item in current.get("aliases", [])
                        if str(item) not in {alias, previous_alias}
                    ],
                ]
                draft["aliases"] = list(dict.fromkeys(item for item in aliases if item))
        elif key == "aliases":
            items = [item.strip() for item in str(value).split(",") if item.strip()]
            alias = str(draft.get("alias", "")).strip()
            draft["aliases"] = list(dict.fromkeys([alias, *items]))
        elif key in self._INTEGER_FIELDS:
            raw = str(value).strip()
            if not raw and key in self._OPTIONAL_INTEGER_FIELDS:
                draft.pop(key, None)
            else:
                draft[key] = int(raw)
        elif key == "gpu_memory_utilization":
            draft[key] = float(str(value).strip())
        elif key in self._BOOLEAN_FIELDS:
            enabled = value if isinstance(value, bool) else str(value).strip().lower() in {"on", "true", "1", "yes"}
            if key == "fit":
                if enabled:
                    draft["fit"] = True
                    draft.setdefault("fit_target_mib", 8192)
                    draft.setdefault("fit_ctx", int(draft.get("context_size", 32768)))
                else:
                    for fit_key in ("fit", "fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"):
                        draft.pop(fit_key, None)
            else:
                draft[key] = enabled
        elif key in {"environment", "safety", "limit_mm_per_prompt"}:
            parsed = json.loads(str(value))
            if not isinstance(parsed, dict):
                raise ValueError(f"{key} must be a JSON object")
            if key == "environment" and not all(
                isinstance(name, str) and isinstance(item, str) for name, item in parsed.items()
            ):
                raise ValueError("environment values must all be strings")
            draft[key] = parsed
        elif key == "extra_args":
            parsed = json.loads(str(value))
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError("extra_args must be a JSON list of strings")
            draft[key] = parsed
        elif key == "restore_profile":
            if str(value).strip():
                draft[key] = str(value).strip()
            else:
                draft.pop(key, None)
        else:
            draft[key] = value

        self._publish(draft)
        return draft

    def _publish(self, draft: dict[str, Any]) -> None:
        self.manifest["profiles"][self.index] = draft
        self.dirty = self.manifest != self.base_manifest


def _truncate_field_value(value: str, width: int) -> str:
    """Keep a root-list value on one row while preserving its useful tail."""

    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    return "..." + value[-(width - 3) :]


def _alt_arrow_from_sequence(sequence: list[int]) -> int | None:
    """Decode an Alt+Arrow sequence that curses did not recognize."""

    import curses

    if not sequence or sequence[0] not in (ord("["), ord("O")):
        return None
    arrow_keys = {
        ord("A"): curses.KEY_UP,
        ord("B"): curses.KEY_DOWN,
        ord("C"): curses.KEY_RIGHT,
        ord("D"): curses.KEY_LEFT,
    }
    final = sequence[-1]
    if final not in arrow_keys:
        return None
    parameters = sequence[1:-1]
    if sequence[0] == ord("O") and parameters:
        return None
    if sequence[0] == ord("[") and any(
        value != ord(";")
        and value != ord("?")
        and not ord("0") <= value <= ord("9")
        for value in parameters
    ):
        return None
    return arrow_keys[final]


def _read_curses_key(stdscr: Any) -> int:
    """Read a key while keeping Alt-modified arrows from becoming Escape."""

    import curses

    key = stdscr.getch()
    if key != 27:
        return key

    sequence: list[int] = []
    try:
        # Alt is commonly encoded as ESC followed by a CSI/SS3 arrow
        # sequence.  curses.keypad() does not decode every modifier variant,
        # so wait briefly for the tail before treating ESC as cancellation.
        stdscr.timeout(50)
        for _ in range(16):
            next_key = stdscr.getch()
            if next_key == -1:
                break
            sequence.append(next_key)
            if len(sequence) == 1 and next_key in (
                curses.KEY_UP,
                curses.KEY_DOWN,
                curses.KEY_LEFT,
                curses.KEY_RIGHT,
            ):
                return next_key
            decoded = _alt_arrow_from_sequence(sequence)
            if decoded is not None:
                return decoded
            if sequence[0] not in (ord("["), ord("O")):
                break
            if sequence[0] == ord("O") and len(sequence) > 1:
                break
            if sequence[0] == ord("[") and any(
                value != ord(";")
                and value != ord("?")
                and not ord("0") <= value <= ord("9")
                and value not in (ord("A"), ord("B"), ord("C"), ord("D"))
                for value in sequence[1:]
            ):
                break
    finally:
        stdscr.timeout(-1)

    # Preserve an unrecognized Escape sequence for the caller instead of
    # dropping the bytes that followed the Escape prefix.
    for value in reversed(sequence):
        curses.ungetch(value)
    return key


def _curses_prompt(
    stdscr: Any,
    prompt: str,
    current: Any = "",
    *,
    allow_empty: bool = False,
) -> str:
    """Edit one selected field with a curses-native single-line editor."""

    import curses

    height, width = stdscr.getmaxyx()
    row = max(height - 2, 0)
    max_width = max(width - 1, 1)
    prefix = f"{prompt}: "
    original = str(current or "")
    buffer = list(original)
    cursor = len(buffer)

    def render() -> None:
        nonlocal cursor
        text = "".join(buffer)
        available = max(max_width - len(prefix), 1)
        start = max(0, min(cursor - available + 1, max(len(text) - available, 0)))
        visible = text[start : start + available]
        line = prefix + visible
        stdscr.move(row, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(row, 0, line, max_width)
        cursor_column = min(max_width - 1, len(prefix) + cursor - start)
        stdscr.move(row, cursor_column)
        stdscr.refresh()

    render()
    while True:
        try:
            key = _read_curses_key(stdscr)
        except KeyboardInterrupt:
            return original

        if key in (10, 13, curses.KEY_ENTER):
            value = "".join(buffer).strip()
            return value if value or allow_empty else original
        if key in (27, 3):  # Escape or Ctrl-C cancels the field edit.
            return original
        if key in (curses.KEY_BACKSPACE, 8, 127):
            if cursor > 0:
                del buffer[cursor - 1]
                cursor -= 1
        elif key == getattr(curses, "KEY_DC", -1):
            if cursor < len(buffer):
                del buffer[cursor]
        elif key in (curses.KEY_LEFT,):
            cursor = max(0, cursor - 1)
        elif key in (curses.KEY_RIGHT,):
            cursor = min(len(buffer), cursor + 1)
        elif key in (curses.KEY_HOME, 1):
            cursor = 0
        elif key in (curses.KEY_END, 5):
            cursor = len(buffer)
        elif key == 21:  # Ctrl-U clears the current line.
            buffer.clear()
            cursor = 0
        elif 0 <= key <= 255:
            character = chr(key)
            if character.isprintable():
                buffer.insert(cursor, character)
                cursor += 1
        render()


def _curses_confirm(stdscr: Any, prompt: str, *, default: bool = True) -> bool:
    """Read a compact Y/n confirmation without requiring a typed word."""

    import curses

    height, width = stdscr.getmaxyx()
    choices = "Y/n" if default else "y/N"
    stdscr.addnstr(max(height - 2, 0), 0, f"{prompt} [{choices}]", max(width - 1, 1))
    stdscr.clrtoeol()
    key = _read_curses_key(stdscr)
    if key in (ord("y"), ord("Y")):
        return True
    if key in (ord("n"), ord("N")):
        return False
    if key in (10, 13, curses.KEY_ENTER):
        return default
    return False


def _curses_select(
    stdscr: Any, label: str, current: Any, options: Iterable[str]
) -> tuple[str, bool]:
    """Choose one value from a keyboard-navigable dropdown."""

    import curses

    choices = list(dict.fromkeys([str(current), *options]))
    selected = choices.index(str(current)) if str(current) in choices else 0
    scroll = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        def draw_line(row: int, column: int, text: str, attr: int = 0) -> None:
            # addnstr() raises curses.error when its starting cell is outside
            # the window.  Dropdown entries can be taller than the viewport,
            # so clip both the row and the available columns explicitly.
            if not (0 <= row < height and 0 <= column < width):
                return
            count = width - column - 1
            if count > 0:
                stdscr.addnstr(row, column, text, count, attr)

        draw_line(0, 0, f"{label} — choose a value", curses.A_BOLD)
        draw_line(1, 0, "↑/↓ move  Enter select  Esc cancel")

        rendered_choices: list[list[str]] = []
        for choice in choices:
            choice_lines: list[str] = []
            for line in str(choice).splitlines() or [""]:
                choice_lines.extend(
                    textwrap.wrap(
                        line,
                        width=max(width - 7, 1),
                        replace_whitespace=False,
                        drop_whitespace=False,
                    )
                    or [""]
                )
            rendered_choices.append(choice_lines or [""])

        viewport_height = max(height - 3, 0)
        if viewport_height:
            starts: list[int] = []
            line_count = 0
            for choice_lines in rendered_choices:
                starts.append(line_count)
                line_count += len(choice_lines)

            selected_start = starts[selected]
            selected_end = selected_start + len(rendered_choices[selected])
            if selected_start < scroll:
                scroll = selected_start
            elif selected_end > scroll + viewport_height:
                scroll = selected_end - viewport_height
            scroll = max(0, min(scroll, max(line_count - viewport_height, 0)))

            line_number = 0
            for choice_index, choice_lines in enumerate(rendered_choices):
                for line_index, line in enumerate(choice_lines):
                    if scroll <= line_number < scroll + viewport_height:
                        marker = (
                            "●" if choice_index == selected else "○"
                        ) if line_index == 0 else " "
                        prefix = f"{marker} " if line_index == 0 else "  "
                        draw_line(
                            3 + line_number - scroll,
                            2,
                            f"{prefix}{line}",
                            curses.A_REVERSE if choice_index == selected else 0,
                        )
                    line_number += 1

        key = _read_curses_key(stdscr)
        if key in (27, ord("q")):
            return str(current), True
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(choices)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(choices)
        elif key in (10, 13):
            if choices[selected] == "(no valid catalog builds available)":
                continue
            return str(choices[selected]), False


def _curses_checklist(
    stdscr: Any, label: str, current: Any, options: Iterable[str]
) -> tuple[list[str], bool]:
    """Choose zero or more values from a keyboard-navigable checklist."""

    import curses

    choices = list(options)
    # Universal base capability: always present and non-removable, so the
    # checklist cannot produce an invalid capabilities list (e.g. dropping
    # "text").
    locked = {"text"} if "text" in choices else set()
    selected = {"text"} | {str(item) for item in (current or [])}
    cursor = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        stdscr.addnstr(0, 0, f"{label} — choose values", max(width - 1, 1), curses.A_BOLD)
        stdscr.addnstr(1, 0, "↑/↓ move  Space toggle  Enter accept  Esc cancel", max(width - 1, 1))
        for row, choice in enumerate(choices, 3):
            attr = curses.A_REVERSE if row - 3 == cursor else 0
            if choice in locked:
                stdscr.addnstr(row, 2, f"[x] {choice} (always on)", max(width - 3, 1), curses.A_DIM | attr)
                continue
            marker = "[x]" if choice in selected else "[ ]"
            stdscr.addnstr(row, 2, f"{marker} {choice}", max(width - 3, 1), attr)
        key = _read_curses_key(stdscr)
        if key == 27:
            return list(current or []), True
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(choices)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(choices)
        elif key == ord(" "):
            choice = choices[cursor]
            if choice in locked:
                continue
            if choice in selected:
                selected.discard(choice)
            else:
                selected.add(choice)
        elif key in (10, 13):
            return [choice for choice in choices if choice in selected], False


def _edit_current_field(stdscr: Any, model: TUIModel) -> None:
    field = model.current_field
    current = model.current
    if field is None or current is None:
        return
    value = model.field_value(field.key)
    cancelled = False
    original_value = value
    if field.kind == "select":
        value, cancelled = _curses_select(stdscr, field.label, value, field.options)
    elif field.kind == "multi":
        value, cancelled = _curses_checklist(stdscr, field.label, current.get(field.key, []), field.options)
    elif field.kind == "toggle":
        value = not bool(value)
    else:
        value = _curses_prompt(
            stdscr,
            field.label,
            value,
            allow_empty=field.key
            in {
                "mmproj_path",
                "model_draft",
                "restore_profile",
                *_SPECULATIVE_COUNT_FIELDS,
            },
        )
    if cancelled:
        return
    if field.kind not in {"select", "multi", "toggle"} and (
        value == original_value
        or (
            isinstance(value, str)
            and str(value).strip() == str(original_value).strip()
        )
    ):
        return
    model.set_field(field.key, value)


def _format_field_value(value: Any) -> str:
    if value in (None, ""):
        return "(unset)"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return " | ".join(line.strip() for line in value.splitlines() if line.strip())
    return str(value)


class OperationsTUIModel:
    """Testable state model for the broker transition-operations view."""

    def __init__(
        self,
        *,
        profiles: Iterable[str],
        list_operations: Callable[[], Mapping[str, Any]],
        refresh_profiles: Callable[[], Iterable[str]] | None = None,
        submit_load: Callable[[str], Mapping[str, Any]],
        submit_restart: Callable[[], Mapping[str, Any]],
        submit_stop: Callable[[], Mapping[str, Any]],
        submit_reload: Callable[[], Mapping[str, Any]],
        status_operation: Callable[[str], Mapping[str, Any]],
        cancel_operation: Callable[[str], Mapping[str, Any]],
        wait_operation: Callable[[str], Mapping[str, Any]],
    ) -> None:
        self.profiles = tuple(profiles)
        self._refresh_profiles = refresh_profiles
        self._list_operations = list_operations
        self._submit_load = submit_load
        self._submit_restart = submit_restart
        self._submit_stop = submit_stop
        self._submit_reload = submit_reload
        self._status_operation = status_operation
        self._cancel_operation = cancel_operation
        self._wait_operation = wait_operation
        self.entries: list[dict[str, Any]] = []
        self.selected_index = 0
        self.selected_request_id: str | None = None
        self.status_message = ""
        self.refresh()

    @property
    def selected(self) -> dict[str, Any] | None:
        if not self.entries:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.entries) - 1))
        return self.entries[self.selected_index]

    def refresh(self) -> list[dict[str, Any]]:
        if self._refresh_profiles is not None:
            self.profiles = tuple(self._refresh_profiles())
        raw_entries = self._list_operations().get("operations", [])
        if not isinstance(raw_entries, list):
            raise ValueError("operation list response is malformed")
        rows: list[dict[str, Any]] = []
        for entry in raw_entries:
            if not isinstance(entry, Mapping):
                continue
            execution = entry.get("execution")
            requests = entry.get("requests", [])
            if not isinstance(execution, Mapping):
                continue
            if not isinstance(requests, list) or not requests:
                rows.append(dict(execution))
                continue
            for request in requests:
                if isinstance(request, Mapping):
                    rows.append({**dict(execution), **dict(request)})
        self.entries = rows
        if self.selected_request_id is not None:
            for index, row in enumerate(rows):
                if row.get("request_id") == self.selected_request_id:
                    self.selected_index = index
                    break
            else:
                self.selected_index = min(self.selected_index, max(len(rows) - 1, 0))
        else:
            self.selected_index = min(self.selected_index, max(len(rows) - 1, 0))
        return self.entries

    def move(self, delta: int) -> None:
        if self.entries:
            self.selected_index = (self.selected_index + delta) % len(self.entries)
            request_id = self.entries[self.selected_index].get("request_id")
            self.selected_request_id = request_id if isinstance(request_id, str) else None

    def submit_profile_load(self, profile_id: str) -> Mapping[str, Any]:
        if profile_id not in self.profiles:
            raise ValueError(f"unknown profile: {profile_id}")
        return self._remember(self._submit_load(profile_id), "load")

    def submit_restart(self) -> Mapping[str, Any]:
        return self._remember(self._submit_restart(), "restart")

    def submit_stop(self) -> Mapping[str, Any]:
        return self._remember(self._submit_stop(), "stop")

    def submit_reload(self) -> Mapping[str, Any]:
        return self._remember(self._submit_reload(), "reload")

    def status_selected(self) -> Mapping[str, Any]:
        result = self._status_operation(self._require_selected_request())
        operation = result.get("operation")
        if isinstance(operation, Mapping):
            self.status_message = self._describe(operation)
        self.refresh()
        return result

    def cancel_selected(self) -> Mapping[str, Any]:
        result = self._cancel_operation(self._require_selected_request())
        operation = result.get("operation")
        self.status_message = self._describe(operation) if isinstance(operation, Mapping) else "Cancel sent."
        self.refresh()
        return result

    def wait_selected(self) -> Mapping[str, Any]:
        result = self._wait_operation(self._require_selected_request())
        operation = result.get("operation")
        self.status_message = self._describe(operation) if isinstance(operation, Mapping) else "Wait completed."
        self.refresh()
        return result

    def _remember(self, result: Mapping[str, Any], action: str) -> Mapping[str, Any]:
        operation = result.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError(f"broker did not return an operation ID for {action}")
        request_id = operation.get("request_id")
        execution_id = operation.get("execution_id")
        if isinstance(request_id, str):
            self.selected_request_id = request_id
        self.status_message = (
            f"{action} request {request_id or '(missing ID)'}; "
            f"execution {execution_id or '(missing ID)'}"
        )
        self.refresh()
        return result

    def _require_selected_request(self) -> str:
        row = self.selected
        request_id = row.get("request_id") if row is not None else self.selected_request_id
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("select an operation request first")
        self.selected_request_id = request_id
        return request_id

    @staticmethod
    def _describe(operation: Mapping[str, Any]) -> str:
        return (
            f"{operation.get('status', 'unknown')} — request "
            f"{operation.get('request_id', '?')} / execution "
            f"{operation.get('execution_id', '?')}"
        )


def run_operations_tui(model: OperationsTUIModel) -> int:
    """Run the asynchronous load/restart/unload operation monitor."""
    import curses

    def draw(stdscr: Any) -> int:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            stdscr.addnstr(0, 0, "Llama-Broker transition operations", max(width - 1, 1), curses.A_BOLD)
            stdscr.addnstr(
                1,
                0,
                "↑/↓ select  l load  R restart  u unload  g reload  s status  c cancel  w wait  r refresh  q quit",
                max(width - 1, 1),
            )
            visible = max(height - 5, 1)
            start = max(0, min(model.selected_index - visible // 2, max(len(model.entries) - visible, 0)))
            for row, entry in enumerate(model.entries[start : start + visible], 3):
                index = start + row - 3
                marker = ">" if index == model.selected_index else " "
                request_id = str(entry.get("request_id", "-"))
                execution_id = str(entry.get("execution_id", "-"))
                target = str(entry.get("target") or "-")
                line = (
                    f"{marker} {entry.get('status', 'unknown'):<10} "
                    f"{entry.get('type', 'operation'):<8} {target:<20} "
                    f"req {request_id[:10]} exec {execution_id[:10]}"
                )
                attr = curses.A_REVERSE if index == model.selected_index else 0
                stdscr.addnstr(row, 0, line, max(width - 1, 1), attr)
            stdscr.addnstr(max(height - 2, 0), 0, model.status_message, max(width - 1, 1), curses.A_BOLD)
            stdscr.refresh()
            key = _read_curses_key(stdscr)
            try:
                if key in (ord("q"), 27):
                    return 0
                if key in (curses.KEY_UP, ord("k")):
                    model.move(-1)
                elif key in (curses.KEY_DOWN, ord("j")):
                    model.move(1)
                elif key == ord("r"):
                    model.refresh()
                    model.status_message = "Operation list refreshed."
                elif key == ord("l"):
                    if not model.profiles:
                        model.status_message = "No profile IDs are available."
                        continue
                    selected, cancelled = _curses_select(stdscr, "Load profile", model.profiles[0], model.profiles)
                    if not cancelled:
                        model.submit_profile_load(str(selected))
                elif key == ord("R"):
                    model.submit_restart()
                elif key == ord("u"):
                    model.submit_stop()
                elif key == ord("g"):
                    model.submit_reload()
                elif key == ord("s"):
                    model.status_selected()
                elif key == ord("c"):
                    model.cancel_selected()
                elif key == ord("w"):
                    model.wait_selected()
            except Exception as error:
                model.status_message = f"Operation request failed: {error}"

    return curses.wrapper(draw)


def run_tui(
    profiles_path: Path | str,
    selector: str | None = None,
    *,
    apply_callback: ApplyCallback | None = None,
) -> int:
    """Run a field-oriented full-screen profile editor (stdlib only)."""

    import curses
    import threading
    import time

    path = Path(profiles_path)
    manifest = load_manifest_source(path, allow_missing_artifacts=True)
    model = TUIModel(manifest)
    if selector:
        found = find_profile_record(model.manifest, selector)
        if found is None:
            print(f"Profile not found: {selector}")
            return 1
        model.index = found[0]

    def draw(stdscr: Any) -> int:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        status = ""
        apply_finished: threading.Event | None = None
        apply_completion: dict[str, BaseException | None] = {}
        pending_manifest: dict[str, Any] | None = None
        apply_started_at = 0.0
        while True:
            if apply_finished is not None and apply_finished.is_set():
                error = apply_completion.get("error")
                if error is not None:
                    status = f"Cannot save: {error}"
                else:
                    assert pending_manifest is not None
                    model.manifest = pending_manifest
                    model.base_manifest = copy.deepcopy(pending_manifest)
                    model.dirty = False
                    status = "Saved successfully; loaded model keeps running."
                apply_finished = None
                pending_manifest = None
            if apply_finished is not None:
                status = (
                    f"Applying changes ({int(time.monotonic() - apply_started_at)}s); "
                    "please wait."
                )
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            current = model.current
            fields = model.visible_fields
            model.field_index = max(0, min(model.field_index, max(len(fields) - 1, 0)))
            stdscr.addnstr(0, 0, "Llama-Broker profile editor", max(width - 1, 1), curses.A_BOLD)
            if current is None:
                stdscr.addnstr(2, 0, "No profiles configured. Press q to quit.", max(width - 1, 1))
                key = _read_curses_key(stdscr)
                if key in (ord("q"), 27):
                    return 0
                continue

            dirty_marker = " *" if model.dirty else ""
            profile_title = f"{current.get('display_name', current.get('id'))}  ({current.get('id')}){dirty_marker}"
            stdscr.addnstr(1, 0, profile_title, max(width - 1, 1), curses.A_UNDERLINE)
            help_text = "↑/↓ field  Enter edit  Tab advanced  [/] profile  s save  q quit"
            stdscr.addnstr(2, 0, help_text, max(width - 1, 1))
            mode = "advanced" if model.advanced else "serving"
            stdscr.addnstr(3, 0, f"Editing {mode} fields; hashes update automatically when paths change.", max(width - 1, 1))

            viewport_height = max(height - 9, 1)
            start = max(0, min(model.field_index - viewport_height // 2, max(len(fields) - viewport_height, 0)))
            for row, field in enumerate(fields[start : start + viewport_height], 5):
                field_index = start + row - 5
                marker = ">" if field_index == model.field_index else " "
                value = _format_field_value(model.field_value(field.key))
                value_column = 33
                value_width = max(width - value_column - 1, 1)
                value = _truncate_field_value(value, value_width)
                line = f"{marker} {field.label:<30} {value}"
                attr = curses.A_REVERSE if field_index == model.field_index else 0
                stdscr.addnstr(row, 0, line, max(width - 1, 1), attr)

            if status:
                stdscr.addnstr(max(height - 3, 0), 0, status, max(width - 1, 1), curses.A_BOLD)
            footer = (
                "Apply is running; wait here for its result."
                if apply_finished is not None
                else "s saves profile config and refreshes aliases; the loaded model keeps running."
            )
            stdscr.addnstr(max(height - 2, 0), 0, footer, max(width - 1, 1))
            stdscr.timeout(250 if apply_finished is not None else -1)
            key = _read_curses_key(stdscr)
            if apply_finished is not None:
                continue
            if key in (ord("q"), 27):
                if model.dirty:
                    if not _curses_confirm(stdscr, "Discard unsaved changes?", default=False):
                        status = "Still editing; changes not discarded."
                        continue
                return 0
            if key in (curses.KEY_UP, ord("k")):
                model.move_field(-1)
            elif key in (curses.KEY_DOWN, ord("j")):
                model.move_field(1)
            elif key in (ord("["),):
                model.move(-1)
            elif key in (ord("]"),):
                model.move(1)
            elif key in (9, ord("x")):
                model.toggle_advanced()
            elif key in (10, 13, ord("e"), ord(" ")):
                try:
                    _edit_current_field(stdscr, model)
                    status = "Field updated; press s to validate and save."
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    status = f"Not changed: {error}"
            elif key == ord("s"):
                if not model.dirty:
                    status = "No changes to save."
                    continue
                try:
                    latest = load_manifest_source(path, allow_missing_artifacts=True)
                    merged, conflicts = merge_manifest_drafts(
                        model.base_manifest, latest, model.manifest
                    )
                    if conflicts:
                        conflict_text = ", ".join(conflicts[:4])
                        if len(conflicts) > 4:
                            conflict_text += ", …"
                        if not _curses_confirm(
                            stdscr,
                            f"External conflicts ({conflict_text}); keep draft values?",
                            default=False,
                        ):
                            status = "Save cancelled; external conflicts remain."
                            continue
                    validate_manifest(merged, directory=path.parent)
                    if not _curses_confirm(stdscr, "Save profile config?", default=True):
                        status = "Save cancelled."
                        continue
                    if apply_callback is not None:
                        changed_ids = {str(current.get("id"))}
                        finished_event = threading.Event()
                        apply_finished = finished_event
                        apply_completion = {}
                        pending_manifest = merged
                        apply_started_at = time.monotonic()

                        def apply_in_background(
                            callback: ApplyCallback = apply_callback,
                            draft: dict[str, Any] = merged,
                            ids: set[str] = changed_ids,
                            completion: dict[str, BaseException | None] = apply_completion,
                            finished: threading.Event = finished_event,
                        ) -> None:
                            try:
                                callback(draft, ids)
                            except BaseException as error:
                                completion["error"] = error
                            finally:
                                finished.set()

                        thread = threading.Thread(
                            target=apply_in_background,
                            name="llama-broker-profile-apply",
                            daemon=False,
                        )
                        try:
                            thread.start()
                        except BaseException:
                            apply_finished = None
                            pending_manifest = None
                            raise
                        status = "Applying profile changes..."
                        continue
                    write_manifest_atomically(path, merged)
                    model.manifest = merged
                    model.base_manifest = copy.deepcopy(merged)
                    model.dirty = False
                    status = "Saved profile configuration."
                except Exception as error:
                    status = f"Cannot save: {error}"
        return 0

    return curses.wrapper(draw)
