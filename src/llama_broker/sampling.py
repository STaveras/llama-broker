"""Per-profile request sampling defaults, independent of backend launch settings."""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
)
SAMPLING_MODES = ("instruct", "thinking")


def validate_sampling_defaults(value: Any) -> Mapping[str, Mapping[str, float | int]]:
    """Validate and freeze a manifest's optional, mode-specific defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("sampling_defaults must be an object")
    unknown = set(value) - set(SAMPLING_MODES)
    if unknown:
        raise ValueError(f"unknown sampling mode: {sorted(unknown, key=str)}")
    result = {}
    for mode, settings in value.items():
        if not isinstance(settings, Mapping):
            raise ValueError(f"sampling_defaults.{mode} must be an object")
        unknown_fields = set(settings) - set(SAMPLING_FIELDS)
        if unknown_fields:
            raise ValueError(f"unknown sampling fields: {sorted(unknown_fields, key=str)}")
        checked = {}
        for key, number in settings.items():
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"sampling_defaults.{mode}.{key} must be a number")
            try:
                finite = math.isfinite(number)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"sampling_defaults.{mode}.{key} must be finite")
            valid = True
            if key == "temperature":
                valid = number >= 0
            elif key in {"top_p", "min_p"}:
                valid = 0 <= number <= 1
            elif key == "top_k":
                valid = isinstance(number, int) and number >= 0
            elif key == "presence_penalty":
                valid = -2 <= number <= 2
            elif key == "repetition_penalty":
                valid = number > 0
            if not valid:
                raise ValueError(f"sampling_defaults.{mode}.{key} is out of range")
            checked[key] = number
        result[mode] = MappingProxyType(checked)
    return MappingProxyType(result)


def qwen27b_sampling_defaults() -> dict[str, dict[str, float | int]]:
    """Return a fresh editable preset; never infer it from a model filename."""
    common = dict(top_k=20, min_p=0.0, repetition_penalty=1.0)
    return {
        "instruct": dict(common, temperature=0.7, top_p=0.80, presence_penalty=1.5),
        "thinking": dict(common, temperature=1.0, top_p=0.95, presence_penalty=0.0),
    }


def apply_sampling_defaults(
    payload: Mapping[str, Any],
    defaults: Mapping[str, Mapping[str, float | int]],
    *,
    runtime: str,
) -> dict[str, Any]:
    """Fill only absent fields, after the existing reasoning policy is resolved.

    llama.cpp's OpenAI-compatible endpoint takes its native repeat_penalty
    extension, not vLLM's repetition_penalty spelling. No new fields are sent
    to runtimes whose sampling contract has not been qualified here.
    """
    result = dict(payload)
    if runtime != "llamacpp" or not defaults:
        return result
    kwargs = result.get("chat_template_kwargs")
    thinking = isinstance(kwargs, Mapping) and kwargs.get("enable_thinking") is True
    settings = defaults.get("thinking" if thinking else "instruct", {})
    if "repetition_penalty" in result:
        result.setdefault("repeat_penalty", result.pop("repetition_penalty"))
    for name, value in settings.items():
        target = "repeat_penalty" if name == "repetition_penalty" else name
        result.setdefault(target, value)
    return result
