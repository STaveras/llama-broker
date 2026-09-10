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
Adapters provide filesystem locking, manifest persistence, process control,
status inspection, readiness, and live reload behavior. No adapter may re-enter
the lock owner while that lock is held.

New callers are explicitly asynchronous. `brokerctl`, the TUI, HTTP callers,
and a future browser view receive an operation identity and observe durable
progress rather than relying on a long blocking request. The existing HTTP
`wait=true` mode remains as an explicit blocking compatibility path; new
callers use `wait=false` and operation status. The command-line experience may
add an explicit wait operation later, but transition execution remains
asynchronous by default.

Operation state persists across broker restarts. Target failure is fail-closed
with no automatic restoration. Duplicate queued targets coalesce; distinct
queued targets remain strict FIFO, and an in-flight transition is never
superseded. Each caller receives a request identity linked to the one shared
execution for a coalesced target. A caller may stop waiting or cancel its own
queued request without cancelling shared work another caller still wants.
Cancellation is allowed while queued and, at most, before destructive work
begins. Once draining commits to stopping or starting a backend, the transition
finishes safely and publishes a terminal result. After a broker restart, safe
queued work may be replayed; interrupted in-flight work is failed and the
actual backend state is reconciled before new work is accepted. Ordinary load
of the active profile is idempotent and completes without a backend restart;
explicit restart is a forced transition even when the requested profile is
already active. A transient profile's automatic restoration is conditional: if
no newer explicit target is queued, restore the prior profile; if a newer target
is queued, skip the automatic restore and transition directly to that target.
Once restoration has begun draining or stopping a backend, it finishes before
later queued work runs. Only one transient context is supported initially; a
future restore stack may support nested transient contexts. A queued execution
uses the immutable profile/catalog snapshot captured when it was requested. It
must recheck that snapshot's pinned artifacts and safety facts before destructive
work, but it never silently re-resolves to newer profile data. Durable request
and execution records extend the existing versioned `state.json`, while
profiles/catalog remain the configuration authorities. Retain the most recent
32 completed transition operations. Keep existing `brokerctl load`, `start`, and
`restart` command names, but have them return request/execution identities;
waiting is explicit through a separate operation wait/status command. A
coalesced execution continues while at least one attached request remains; if
all attached requests cancel before draining starts, the shared execution is
removed from the queue. Profile and catalog reloads are ordered operations in
the same FIFO queue, so configuration changes cannot race a queued snapshot or
an in-flight backend lifecycle operation. Explicit restart is a priority
execution: it may move ahead of queued loads, but it never interrupts an
in-flight drain or process start; its force semantics still bypass active-load
idempotence only. Duplicate restart requests coalesce while an equivalent
restart remains queued or in flight; a later request after that execution has
completed is a new forced execution. Cancellation remains possible through
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
conversation.
