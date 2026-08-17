# Runtime adapters

The registry currently contains only `llamacpp` and `vllm`.

Each adapter owns the complete backend argv. `llamacpp` preserves the existing
`llama-server -m MODEL --alias ALIAS ...` command. `vllm` emits
`vllm serve MODEL ...` and maps only explicitly supported profile fields:
context size, `parallel` (as tensor parallelism unless overridden), GPU memory
utilization (`gpu_memory_utilization`), KV-cache dtype (`kv_cache_dtype`),
endpoint/security flags, tool calling (`tool_call_parser`), and multimodal
media limits (`limit_mm_per_prompt`, `allowed_local_media_path`).

The profile parser rejects llama.cpp-only fields on vLLM profiles. Runtime
specific options should be added to the adapter and parser together; arbitrary
passthrough flags are intentionally not an escape hatch.
