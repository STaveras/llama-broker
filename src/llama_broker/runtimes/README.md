# Runtime adapters

The registry contains `llamacpp`, `vllm`, and `strata`.

Each adapter owns the backend command shape. `llamacpp` preserves the existing
`llama-server -m MODEL --alias ALIAS ...` command. `vllm` emits
`vllm serve MODEL ...` and maps only explicitly supported profile fields.
`strata` launches the pinned `strata-server` bridge and a per-profile Strata
launch config; it does not use llama.cpp or vLLM flags.

## Artifact verification

Normal requests, model listing, and profile switches check file presence without
hashing model weights or runtime artifacts. If backend activation fails, the
controller cleans up the target, runs a full artifact verification, and records
its result in the failed state and broker events. `brokerctl verify PROFILE`
remains available for an explicit content check.

## Strata contract

- Strata profiles are text-only and require `parallel: 1`. Vision, MCP,
  speculative MTP, arbitrary `extra_args`, and profile environment overrides
  are rejected until separately qualified.
- `server_binary` is an executable `strata-server` bridge inside a content-
  addressed runtime bundle. Build it from a clean, pinned Strata checkout and
  a locally compiled engine; do not execute Strata's one-click installer or
  download a `releases/latest` binary. Create a CPython 3.14 venv with copies and
  install the checked-in wheel hashes:
  ```sh
  python3 -m venv --copies BUNDLE/venv
  BUNDLE/venv/bin/python -m pip install --require-hashes --only-binary=:all: \
    -r tools/strata-runtime-requirements-py314.lock
  python3 tools/package_strata_bridge.py --bundle-root BUNDLE \
    --python BUNDLE/venv/bin/python
  ```
  This writes the executable at `BUNDLE/bin/strata-server` and prints the hash
  to record in the runtime catalog. The bundle must also contain the matching
  Strata source snapshot, compiled engine, static expert profile, and private
  Python environment installed from the lock above.
- The bridge reads and hashes every Python file in its source inventory before
  import; the inventory must exactly match the Python files in the working
  directory and those hashes must match the clean pinned source checkout.
- The bridge reads the broker API key from its private file and applies
  authentication before dispatching every HTTP request. It rereads the file per
  request so overlapping key rotations remain valid; it never stores the key in
  profile/config JSON, environment variables, or argv.
- The bridge accepts only loopback binds and applies broker-managed host/port
  values. It exposes the Strata backend as HTTP only on loopback; the broker's
  client-facing TLS/API-key boundary remains unchanged. Startup refuses an
  unsafe bind, an unpinned config/engine/source inventory, insufficient host
  headroom or VRAM, excess swap, or a cgroup limit that exceeds policy.
- `strata_artifacts_manifest_path` and `strata_launch_config_path` are
  profile-specific absolute paths with SHA-256 pins. The manifest has safe
  model-root-relative paths and hashes for both GGUF shards and every Strata
  pack file. The launch config pins the engine binary/hash, pack/native/PLE
  paths, tokenizer, context, and backend model name. The broker profile alias
  can be renamed independently; requests are translated to the pinned backend
  name. Unknown config fields and non-loopback host, API-key, MCP, vision, and
  network flags are rejected.
- Strata IQ2_XS profiles have a resident mode (about 48 GiB system RAM) and an
  explicit low-RAM mmap mode. Low-RAM mode requires the pack's `experts.bin` and
  `--mmap-experts`; it estimates the hot-set fit using total system RAM plus GPU
  VRAM, while warning (not silently treating VRAM as host RAM) when much of the
  expert arena is likely to page from SSD. Both modes require at least 6 GiB
  host headroom after the broker stops the old backend, at least 12 GiB total /
  8 GiB free VRAM, swap use at or below 1 GiB, and an effective cgroup cap no
  higher than physical system RAM.

The profile parser rejects runtime-specific fields on other runtime families.
Runtime-specific options are allowlisted in the adapter and parser together;
arbitrary passthrough flags are intentionally not an escape hatch.
