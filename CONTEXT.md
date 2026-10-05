# Llama-Broker domain context

## Profile transition

A profile transition is the broker-owned lifecycle operation that moves the
single active backend from one profile to another, or restarts/stops the current
profile. It includes request draining, operation serialization, manifest and
catalog revision checks, process control, readiness verification, durable state,
and fail-closed failure handling.

## Profile transition deepening

The future profile transition module should optimize for the current one-backend
lifecycle first. Multi-instance residency is a later phase, not a requirement
for the first deepening.

The preferred architecture is a pure orchestration core with injected adapters.
The core exclusively owns the cross-process lifecycle lock for each transition.
Adapters provide manifest persistence, process control, status inspection,
readiness, and live reload behavior under that ownership; none may reacquire or
re-enter the lock owner while it is held.

New callers are explicitly asynchronous. `brokerctl`, the TUI, HTTP callers,
and a future browser view receive an operation identity and observe durable
progress rather than relying on a long blocking request. HTTP operation submission
and status use broker-native admin routes, not OpenAI inference routes. HTTP
callers may cancel their own queued request through an authenticated admin route,
subject to the shared pre-commit cancellation rules. Existing HTTP
`/admin/switch` keeps its current `wait=true` default; `/admin/restart` and
`/admin/stop` gain the same blocking default for backward compatibility. Setting
`wait=false` opts into async and returns a durable operation ID.
New callers use `wait=false` and operation status. The CLI exposes explicit
`brokerctl operation list|status|wait|cancel` commands and an interactive
`brokerctl operation tui`; the curses TUI also refreshes its profile choices
when the catalog changes.

Operation state persists across broker restarts. Post-commit target failure is
fail-closed with no automatic restoration; pre-commit validation failure
preserves the proven prior backend. Duplicate queued targets coalesce; distinct
queued targets remain strict FIFO. Explicit operations and transient restores are
never preempted. The one exception is a request-driven load with no active leases
or queued transition: when `LLAMA_BROKER_PREEMPT_IDLE_LOADS` is enabled, a request
for another profile may cancel that idle load and reserve the next transition
slot. The preempted request waits behind the newer one and retries without
preempting another load. Preemption still waits for the prior backend to exit; no GPU
memory telemetry is used to start overlapping backends. Each caller receives a
caller-specific `request_id` and the shared `execution_id` for a coalesced
target; `request_id` addresses status and cancellation, and status exposes the
linked execution. A caller may stop waiting or cancel its own queued request
without cancelling shared work another caller still wants.
Cancellation is allowed while queued and, at most, before destructive work
begins. Once draining commits to stopping or starting a backend, the transition
finishes safely and publishes a terminal result. After a broker restart, safe
queued work may be replayed; interrupted in-flight work is failed and the
actual backend state is reconciled before new work is accepted. Ordinary load
of the active profile is idempotent only when no earlier queued operation will
change the active profile; if A is active, B is queued, then A is requested, the
second A remains queued behind B. Explicit restart is a forced transition even
when the requested profile is already active. A transient profile's automatic
restoration is conditional: if no newer explicit lifecycle operation is queued,
restore the prior profile; if a newer load, restart, or unload/stop is queued,
skip the automatic restore and run that operation at its normal queue priority.
Once restoration has begun draining or stopping a backend, it finishes before
later queued work runs. Only one transient context is supported initially; a
future restore stack may support nested transient contexts. A queued execution
uses the immutable profile/catalog snapshot captured when it was requested. It
must recheck that snapshot's configuration, artifact availability, and safety facts
before destructive work, but it never silently re-resolves to newer profile data.
Artifact content hashing runs after failed backend activation or an explicit
`brokerctl verify`; ordinary requests, model listing, and switches do not hash
model weights or runtime artifacts. Verification uses
deterministic in-memory adapters for orchestration-core tests, with separate
contract tests for persistence and process-control adapters; tests must never
operate on the live broker or model. Focused tests cover ordinary
load/switch/cancellation, strict FIFO, duplicate-target coalescing, the fixed
three-pending cap, priority restart/stop, transient restoration, immutable
snapshot rejection, and corrupt-state recovery. Exhaustive crash/fault injection
remains a separate test-hardening phase.
Durable request and execution records extend the existing `state.json` with a
versioned `operations` namespace, while profiles/catalog remain the configuration
authorities. A shared durable-state adapter atomically preserves the existing
top-level backend fields and a
namespaced `operations` section under one serialization lock; independent
whole-file writers must not overwrite each other. A request is acknowledged
only after its request/execution records are atomically persisted; persistence
failure rejects acceptance. On startup,
unreadable state or an unsupported state schema is preserved as a backup; the
broker reconciles actual services and rebuilds minimal state automatically
rather than starting from an assumed inactive state. During this recovery,
pending operations are failed rather than replayed and must be resubmitted;
interrupted work is reconciled against actual services. Request identities are
broker-generated; submissions do not require caller-supplied idempotency keys or
add nonstandard fields to OpenAI-compatible inference payloads. Equivalent queued
targets still coalesce. Retain the 32 most recent completed shared executions,
including every attached request outcome; evict each execution and its requests
together. Evicted request IDs return a generic not-found/expired result; retain no
tombstones beyond the history. Allow at most three
queued executions awaiting start; the currently executing transition is not
counted, and coalesced requests share one queued execution. When full, reject new
distinct normal submissions rather than discard accepted work. Explicit restart
and unload/stop bypass queued loads and run immediately after any
non-interruptible in-flight transition; they do not consume normal queue slots.
Unload/stop outranks restart and cancels any restart that has not begun. Restart
still outranks ordinary queued loads. Keep existing
`brokerctl load`, `start`, and `restart` command names, but have them return
request/execution identities; waiting is explicit through a separate operation
wait/status command. Status exposes `queued → preflight → committed/running →
terminal`, making the irreversible cancellation point visible. Cancellation is
allowed only before commit; afterward it only stops the caller from waiting. A
coalesced execution continues while at least one attached request remains; if
all attached requests cancel before draining starts, the shared execution is
removed from the queue. Profile and catalog reloads are ordered operations in
the same FIFO queue, so configuration changes cannot race a queued snapshot or
an in-flight backend lifecycle operation. Explicit restart is a priority
execution: it may move ahead of queued loads, but it never interrupts an
in-flight drain or process start; its force semantics still bypass active-load
idempotence only. Duplicate explicit restart requests coalesce until the entire
transition queue is empty. Cancellation remains possible through
preflight—snapshot and revision validation, lock acquisition, and quiescence
checks—with an irreversible commit marker immediately before stopping the old
backend. After that marker, cancellation means only that a caller stops waiting.
Explicit unload/stop is also a priority execution after any in-flight transition
and before queued loads. If a target execution fails, later queued work continues
in order; the failed request/execution remains terminal and visible, and each
later target must independently pass snapshot, artifact, safety, and readiness
checks. New inference leases are rejected once any transition is queued;
existing leases drain under the bounded drain policy before destructive work
begins.

These decisions were confirmed during the profile transition grilling
conversation. Source transcript: `@session:default/20260906_184533_16627d`.
