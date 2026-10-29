# llama.cpp batch values

`batch_size` (logical batch, `-b`) and `ubatch_size` (physical micro-batch,
`-ub`) use the same validation for every llama.cpp profile, independent of
runtime build ID, executable hash, model, or source commit:

- Both manifest values must be integers in `1..2147483647` (inclusive).
- Booleans, floats, strings, null, nonpositive values, and overflow are rejected.
- `ubatch_size <= batch_size`; equality is allowed, including `128/128`.

The editor and CLI accept integer text and store integers in the manifest.
Typed simple-editor drafts reject booleans and floats instead of coercing them.
There are no broker tuning ceilings at `512/128` or qualified-build batch
exceptions at `4096/512`. For example, `8192/1024` and `8192/8192` are valid
configuration values, not claims that a particular model/GPU can run them.

## Why the signed 32-bit maximum?

The limiting interface is llama-server's command-line parser, not GPU memory.
In the source for the installed qualified llamAmpere runtime, commit
`e7093a5526aa82e3e271120358dc021ab222060a`:

- `common/arg.cpp:890,953` converts integer options with `std::stoi`;
  `:1804-1815` registers `-b` and `-ub` with `int` handlers.
- `common/common.h:559-560` stores `n_batch` and `n_ubatch` as `int32_t`.
- `include/llama.h:384-385` exposes `llama_context_params.n_batch/n_ubatch`
  as `uint32_t`.

Thus the compatible positive CLI/API range ends at `INT32_MAX`, not
`UINT32_MAX`. The installed BeeLlama source confirms the same types and parser
(`common/common.h:491-492`, `common/arg.cpp:935,998,1883-1895`,
`include/llama.h:435-436`). This is a representational boundary, not an
empirically chosen tuning limit. These checks do not launch a runtime or
allocate a batch of that size.

## Configuration paths and remaining safeguards

`ProfileRegistry` performs the authoritative range/type/order checks. CLI
flag edits and interactive/TUI edits validate the resulting manifest before
publishing; `validate_manifest`, atomic writes, and edit transactions use the
same registry. Inherited profiles are resolved before registry validation.
Setup's `512/128` values are defaults, not limits. Autotune's finite
`128/256/512` batch and `32/64/128` micro-batch candidate grids remain bounded
search policy, not validation limits on user-configured values.

RAM/VRAM admission, reserves, cgroup memory/swap/tasks limits, context and
parallel restrictions, artifact pinning, codec allowlists, extra-argument
security, and qualified-runtime non-batch requirements are unchanged. An
accepted value can still fail admission, model-specific runtime checks, or
allocation. Larger batches can increase RAM/VRAM use or reduce stability;
benchmark deliberately under the existing resource controls.

## Activating a source change

Saving profile values and replacing broker Python code are different actions.
Manifest reload does not reload Python modules already imported by a running
broker. Deploy the reviewed code to the service's actual source/install path,
then arrange a broker restart in a maintenance window if the running process
must use the new validator. This can interrupt client requests and startup
reconciliation can affect the managed backend; inspect active operations and
service policy first. Do not promote a runtime, change a profile, reinstall
units, or reload a model merely to activate this validation change.
