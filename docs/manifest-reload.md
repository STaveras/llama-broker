# Live profile reload

The broker watches the profile manifest and queues a reload when its bytes change.
It keeps serving the last successfully loaded configuration until the queued reload
is applied. A failed reload is retried on a later watcher tick; it does not require
an administrator to wait for, or poll, the operation.

The watcher tracks two different facts:

- **Applied identity:** the SHA-256 of the exact manifest bytes used to build the
  loaded registry. It is captured during the same read that supplies the parser,
  not by rereading the file after parsing or when the watcher starts.
- **Pending operation:** a real request/execution identity in the existing reload
  queue. Submission alone does not mark any bytes as applied.

Successful queued application publishes the loaded registry and its applied identity
atomically under the same admission lock used to register watcher submissions,
after controller lifecycle work succeeds and its locks have exited. Failed or
rolled-back queued application leaves the identity unchanged. A queue operation can time
out and report failure while its quarantined transition is still running. If that
transition later applies successfully, the identity advances at the actual registry
swap even though the queue result remains failed. Restoring earlier exact manifest
bytes is then detected and reloaded rather than mistaken for the loaded revision.
Queued cancellation and expired operation-history records cannot leave the watcher
permanently stuck. Cancelling one caller does not cancel shared reload work still
wanted by another caller.

While one watcher reload is queued or running, further edits are coalesced. The
existing snapshot validation rejects a pending revision if it becomes stale before
commit; the next tick then submits the latest manifest. Edits after commit are not
incorrectly certified as applied: the completed reload retains the identity of the
bytes it actually parsed, and the next tick detects the newer disk content.

Operation status/wait commands only observe results. Waiting on an older successful
reload cannot advance the baseline of a newer, still-pending revision.

On each non-shutdown watcher tick, the same admission lock protects a cheap
reconciliation of the baseline from the identity already attached to the loaded
registry. This happens before checking pending execution state, so alternate
registry swaps (including `reload_manifest_for_autotune`) are reflected even while
new submissions remain coalesced. It does not reread disk to certify loaded bytes,
change the alternate path's rollback policy, or equate queue failure with rollback.

Unchanged ticks hash only the small manifest. They do not parse profiles or rehash
model/runtime artifacts. This does not add detection of external catalog-only edits
when the manifest itself is unchanged. The existing reload operation still validates
and resolves the referenced catalog when it runs.

In-memory registries without a source-byte identity conservatively trigger a reload.
Legacy migration similarly leaves the identity unknown (`None`), forcing a reload
rather than claiming that newly rewritten bytes were already parsed.

## Regression tests

`tests/test_manifest_watcher_regressions.py` drives actual reload workers and queue
operations with temporary manifests and an injected controller. It covers startup
edits, same-read identity capture, successful completion without waiters, failed
unit-install rollback and retry, timeout followed by late application and exact-byte
restoration/reload, alternate autotune application followed by exact-byte restoration
and queued recovery, cancellation/coalescing, old waiters, edits during execution,
operation-history eviction, and unchanged-tick cost. It does not operate
on the installed broker, systemd, or real model runtimes.
