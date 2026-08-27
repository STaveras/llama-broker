# Hermes Profile Residue Cleanup Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe Hermes-only command that inventories and removes known residue left by deleted or manually removed Hermes bot profiles without touching shared state or llama-broker model definitions.

**Architecture:** Add a focused `hermes_cli.profile_cleanup` module that scans the default Hermes root for verifiable orphan artifacts, returns a structured report, and applies only explicitly approved cleanup actions. Extend `hermes profile cleanup` with report-first semantics, and harden normal profile deletion so it removes every Hermes-owned alias for the profile. The implementation must not make llama-broker aware of Hermes profiles.

**Tech Stack:** Python 3.11+, `pathlib`, existing Hermes profile/service helpers, argparse, pytest, temporary `HERMES_HOME` fixtures.

---

## Scope and invariants

The command operates on the Hermes installation root returned by `_get_default_hermes_home()` / `get_default_hermes_root()`, not on whichever profile is currently active.

It may inspect and remove only artifacts that are provably Hermes-owned and provably point at a profile that is no longer installed:

- Hermes-generated shell wrappers under `~/.local/bin` whose body contains the generated `hermes -p <profile>` target and whose target profile is absent.
- Profile-scoped gateway service artifacts whose embedded `HERMES_HOME` resolves to a missing named Hermes profile.
- Profile-scoped s6 gateway slots whose profile target is absent, when running in the s6/container service environment.
- A stale root `active_profile` file that names a missing profile.
- Empty invalid entries under `<hermes-root>/profiles/`, such as a reserved `default` directory.

The command must not automatically delete or rewrite:

- The shared root `state.db`, `sessions/`, `cron/`, caches, skills, plugins, auth, or logs.
- Sessions with missing/legacy `profile_name` values; current root state contains many legacy rows with no reliable profile ownership.
- Arbitrary scripts in `~/.local/bin` that do not match the Hermes wrapper format.
- Non-empty invalid profile directories. They are reported as protected/manual-review items unless a later, separately approved purge feature is added.
- Any llama-broker configuration, model profile, backend unit, model file, or broker state.

The live current-state scan found:

- Installed Hermes profiles: `default`, `gemma4`, `orbis`, `qwen3-5`.
- `~/.hermes/profiles/default/` exists even though `default` is reserved and ignored by profile enumeration; it contains one file and must therefore be reported, not silently deleted.
- Existing Hermes wrappers currently point only to installed profiles.
- No current named Hermes gateway unit was found.
- The shared root database contains legacy sessions without profile ownership and must remain untouched.

## Command UX

Use a new subcommand:

```bash
# Safe inventory; no mutation by default.
hermes profile cleanup

# Explicitly request cleanup; prompt before changing anything.
hermes profile cleanup --apply

# Non-interactive cleanup for automation.
hermes profile cleanup --apply --yes

# Machine-readable inventory/report.
hermes profile cleanup --json
```

`--apply` is required for mutation. `--yes` only bypasses the confirmation prompt and is invalid or harmless without `--apply` according to the existing CLI convention. The command should be idempotent: a second scan after a successful apply reports no previously removed candidates.

The report should distinguish:

- `removable`: known Hermes-owned orphan artifacts that the command can remove.
- `protected`: suspicious or invalid artifacts requiring manual review.
- `healthy`: discovered profile-owned artifacts that correctly point to installed profiles.

Each candidate should expose a kind, profile target if known, absolute path, reason, and planned action. JSON output must contain paths and metadata only; never include secrets or file contents.

---

### Task 1: Add the profile-residue inventory module

**Objective:** Create a focused scanner and report model without adding another model tool or broker integration.

**Files:**
- Create: `hermes_cli/profile_cleanup.py`
- Test: `tests/hermes_cli/test_profile_cleanup.py`

**Step 1: Write failing tests for the report model and root resolution**

Create a temporary `HERMES_HOME` fixture with a default root and named profiles. Assert that scanning from a named-profile process still uses the shared root and that the installed set is `{default} ∪ valid named profile directories`.

Expected initial result: import/function failures because the module does not exist.

**Step 2: Implement the minimal inventory types**

Use a frozen dataclass such as:

```python
@dataclass(frozen=True)
class ProfileResidue:
    kind: str
    profile: str | None
    path: Path
    reason: str
    action: str
    safe: bool = True
```

Add a report container with `installed_profiles`, `removable`, `protected`, and `healthy` collections. Keep path resolution and profile-name validation centralized through existing `hermes_cli.profiles` helpers.

**Step 3: Implement safe filesystem scanners**

Add scanners for:

1. `profiles/` entries:
   - Ignore valid installed profile directories.
   - Recognize reserved/malformed entries.
   - Mark non-empty invalid directories protected.
   - Mark empty invalid directories removable only when their name/path is unambiguously under the profiles root.
2. Wrapper aliases:
   - Scan only regular files in the wrapper directory.
   - Reuse the existing wrapper-size limit and generated-wrapper signature.
   - Return every alias that targets an absent profile, including custom aliases; do not rely only on `build_alias_map()`, which intentionally collapses multiple aliases to one display alias.
   - Ignore arbitrary binaries/scripts.
3. Gateway services:
   - Linux: inspect only the user unit directory and verify the unit’s `HERMES_HOME` points to `<root>/profiles/<missing-name>` before classifying it.
   - macOS: inspect only Hermes launch-agent artifacts and verify the profile home in the plist/launch arguments.
   - Windows: use the existing `gateway_windows` task/script path helpers rather than inventing a second naming scheme.
   - s6: inspect only the known `gateway-<profile>` service-slot namespace and verify the target profile is absent.
4. Sticky active profile:
   - If `active_profile` names a missing valid profile, report it as a safe reset-to-default action.

**Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/hermes_cli/test_profile_cleanup.py -o addopts= -q
```

Expected: all inventory tests pass, with no live filesystem changes outside the temporary fixture.

**Step 5: Commit only if implementation is authorized**

```bash
git add hermes_cli/profile_cleanup.py tests/hermes_cli/test_profile_cleanup.py
git commit -m "feat: inventory orphaned Hermes profile residue"
```

Do not commit unrelated pre-existing working-tree changes.

---

### Task 2: Add explicit cleanup application and idempotence

**Objective:** Apply only the scanner’s `removable` candidates after an explicit request, while preserving protected data.

**Files:**
- Modify: `hermes_cli/profile_cleanup.py`
- Test: `tests/hermes_cli/test_profile_cleanup.py`

**Step 1: Write failing apply tests**

Cover:

- Default invocation/report mode does not unlink, stop services, reset files, or mutate the fixture.
- `--apply` removes a recognized stale wrapper and leaves an arbitrary similarly named script intact.
- `--apply` removes a verified stale service artifact through the existing service-manager cleanup path.
- `--apply` resets a stale `active_profile` to default.
- Non-empty invalid profile directories remain intact and are reported as protected.
- A second scan after apply has no duplicate removable candidate.
- Partial failures are reported per artifact and do not cause unrelated candidates to be deleted silently.

**Step 2: Implement the apply phase**

Apply actions in this order:

1. Stop/disable verified stale gateway service artifacts using existing Hermes service helpers; never stop a service based only on a filename.
2. Remove verified Hermes wrapper aliases via the existing wrapper safety checks.
3. Unregister verified s6 slots when applicable.
4. Reset a stale `active_profile` file using `set_active_profile("default")`.
5. Remove only empty invalid profile-root entries.

Return a result containing removed paths and failures. Use the existing systemd/launchd/Windows abstractions where available; do not shell out to broker tooling.

**Step 3: Add report formatting**

Implement stable human output and `--json`. Human output must show the candidate count, exact paths, reason, and whether the item is removable or protected. JSON must be deterministic for tests.

**Step 4: Run focused tests**

```bash
python -m pytest tests/hermes_cli/test_profile_cleanup.py -o addopts= -q
```

Expected: report mode is non-mutating; apply mode is idempotent and leaves protected data untouched.

---

### Task 3: Wire `hermes profile cleanup` into the CLI

**Objective:** Expose the scanner through the existing profile command family without adding a new top-level command.

**Files:**
- Modify: `hermes_cli/subcommands/profile.py`
- Modify: `hermes_cli/main.py`
- Test: `tests/hermes_cli/test_profile_cleanup.py`

**Step 1: Add the parser**

Add a `cleanup` profile action with:

```text
--apply       Apply removable cleanup candidates; otherwise report only.
--yes, -y     Skip the apply confirmation prompt.
--json        Emit the report as JSON.
```

The help text must explicitly say that shared session state and model/broker configuration are not removed.

**Step 2: Add the handler branch**

In `cmd_profile`, call the new module using the shared default Hermes root. Keep exit behavior useful for scripts:

- `0` when the scan succeeds, including when protected items are found.
- non-zero only for invalid arguments or an operational failure that prevented a complete scan/apply.

A protected item is a finding, not a command failure.

**Step 3: Add CLI integration tests**

Invoke the real parser/handler with a temporary `HERMES_HOME`. Verify:

- `profile cleanup` is report-only.
- `profile cleanup --json` is parseable and contains no secret/file-content fields.
- `profile cleanup --apply --yes` removes only the verified stale fixture artifacts.
- `profile cleanup` from a named profile still scans the shared root.

**Step 4: Verify help and tests**

```bash
hermes profile cleanup --help
python -m pytest tests/hermes_cli/test_profile_cleanup.py tests/hermes_cli/test_profiles.py -o addopts= -q
```

Expected: help lists the new options and the targeted tests pass.

---

### Task 4: Harden ordinary profile deletion against future residue

**Objective:** Prevent `hermes profile delete` from leaving custom aliases behind, while preserving its existing service/process safety behavior.

**Files:**
- Modify: `hermes_cli/profiles.py`
- Test: `tests/hermes_cli/test_profiles.py`

**Step 1: Write the regression test**

Create one profile with both its canonical wrapper and a custom alias. Delete the profile with `yes=True`. Assert that both wrappers are gone, the profile directory is gone, and an unrelated executable in `~/.local/bin` remains.

**Step 2: Add an all-alias helper**

Add a helper that returns every recognized Hermes wrapper targeting a given canonical profile. Reuse the existing validation and bounded wrapper read behavior. Do not change `build_alias_map()`’s display semantics.

**Step 3: Update `delete_profile()`**

Remove every wrapper returned by the helper rather than checking only `<wrapper-dir>/<profile-name>`. Keep service shutdown, backend termination, profile-directory removal, and active-profile reset behavior unchanged.

**Step 4: Run deletion regressions**

```bash
python -m pytest tests/hermes_cli/test_profiles.py -o addopts= -q
```

Expected: existing deletion tests and the custom-alias regression pass.

---

### Task 5: Document the command and safety boundary

**Objective:** Make the distinction between Hermes profile cleanup and broker model management explicit.

**Files:**
- Modify: `website/docs/reference/profile-commands.md`
- Modify: `website/docs/user-guide/profiles.md`
- Test: documentation/link checks if present in the repository

**Step 1: Document the command**

Add examples for report-only, apply, non-interactive, and JSON modes. Explain the candidate classes and the protected classes.

**Step 2: Add the architecture note**

State plainly:

- Hermes profiles own bot configuration, sessions, skills, cron, and gateway state.
- llama-broker owns base model definitions and backend routing only.
- `hermes profile cleanup` does not inspect or modify broker model profiles.

**Step 3: Document the known manual-review case**

Explain that non-empty invalid directories such as a reserved `profiles/default` entry are reported but not deleted automatically. Users can inspect or remove those separately after reviewing their contents.

**Step 4: Verify documentation references**

Run the repository’s documentation check, or at minimum verify the referenced command output:

```bash
hermes profile cleanup --help
```

---

### Task 6: End-to-end verification on the live installation

**Objective:** Prove the feature against the real Hermes installation without mutating it by default.

**Files:**
- No source changes; verification only.

**Step 1: Capture current state**

```bash
hermes profile list
hermes profile cleanup --json
```

Expected current findings include the non-empty invalid `~/.hermes/profiles/default/` entry as protected. Existing valid aliases and services should be healthy.

**Step 2: Confirm report-only behavior**

Hash or stat the reported paths before and after a second report-only run. No files, services, or profile configs may change.

**Step 3: Review before applying**

Do not apply cleanup to the live installation automatically. Review each removable candidate and protected item first. In particular, do not remove the current non-empty `profiles/default` directory through this command unless a separate explicit purge design is approved.

**Step 4: Run regression tests**

```bash
python -m pytest tests/hermes_cli/test_profile_cleanup.py tests/hermes_cli/test_profiles.py tests/hermes_cli/test_subcommands_profile_gateway.py -o addopts= -q
```

If the implementation touches shared profile routing or service lifecycle, run the relevant gateway/profile integration tests as well.

---

## Review checklist

- [ ] No llama-broker changes or broker awareness of Hermes profiles.
- [ ] Bare cleanup command is non-destructive.
- [ ] `--apply` is required for mutation; `--yes` is explicit for automation.
- [ ] Only Hermes-owned wrappers/services with verified missing targets are removable.
- [ ] Arbitrary scripts and non-empty invalid directories are protected.
- [ ] Shared root sessions/state/caches/auth/skills are never swept by profile cleanup.
- [ ] Custom aliases are removed during ordinary profile deletion.
- [ ] Temporary-HERMES_HOME tests exercise real path resolution.
- [ ] Live verification begins with JSON/report-only mode.
- [ ] Existing unrelated working-tree changes remain untouched.
