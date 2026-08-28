# Runtime adapters

The registry currently contains `llamacpp`, `vllm`, and `freetoken`.

Each adapter owns the complete backend argv. `llamacpp` preserves the existing
`llama-server -m MODEL --alias ALIAS ...` command. `vllm` emits
`vllm serve MODEL ...` and maps only explicitly supported profile fields:
context size, `parallel` (as tensor parallelism unless overridden), GPU memory
utilization (`gpu_memory_utilization`), KV-cache dtype (`kv_cache_dtype`),
endpoint/security flags, tool calling (`tool_call_parser`), and multimodal
media limits (`limit_mm_per_prompt`, `allowed_local_media_path`).

`freetoken` emits `/path/to/python -m freetoken --model-path MODEL ...` for
FreeToken FTW model directories. Its `extra_args` are checked against an
explicit allowlist covering memory/cache sizing, MoE/NVFP4 backends, tokenizer
and context settings, reasoning/tool parsers, and supported feature toggles.
The broker still owns host/port, artifact hashes, systemd limits, readiness,
and lifecycle transitions.

The profile parser rejects runtime-specific fields and unsupported passthrough
flags. Runtime-specific options should be added to the adapter and parser
together; arbitrary passthrough flags are intentionally not an escape hatch.
