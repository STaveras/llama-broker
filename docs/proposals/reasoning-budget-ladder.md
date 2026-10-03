# Proposal: Reasoning budget ladder for effort levels

Status: agreed; implemented in `apply_reasoning_policy`, uncommitted in the `development` working tree.

## Problem

The experimental ladder from commit `20b2376` crossed over: `minimal = batch_size`
(256) while `low = budget_limit // ubatch_size` (128), so the lowest effort
reserved *more* thinking budget than the next one up. The medium/high/extra-high
runge were `low * 2^(index+1)`, an exponential curve anchored to the crossed
`low` value and to nothing else.

## Decision

Derive every level from the **per-slot budget** (`request_max_output_tokens`
when present, else the profile `max_output_tokens`) in **ubatch-sized chunks**:

| Level | Formula |
|---|---|
| `minimal` | `min(batch_size, budget_limit)` — 1-batch anchor |
| `low` (n=1) | `budget_limit // ubatch_size` |
| `medium` (n=2) | `2 * budget_limit // ubatch_size` |
| `high` (n=3) | `3 * budget_limit // ubatch_size` |
| `extra high` (n=4) | `4 * budget_limit // ubatch_size` |
| `max` | `budget_limit` (unchanged) |

Implementation: `budget = min(budget_limit, index * budget_limit // ubatch_size)`
where `index` is the level's position in
`("minimal", "low", "medium", "high", "extra high", "max")`.

Reference values (limit=8192, batch=256, ubatch=64):

| Level | Budget |
|---|---|
| `minimal` | 256 |
| `low` | 128 |
| `medium` | 256 |
| `high` | 384 |
| `extra high` | 512 |
| `max` | 8192 |

## Properties

- **Monotonic**: `low < medium < high < extra high < max` for any positive
  `ubatch_size`. `minimal` is a documented 1-batch anchor, not a ladder rung;
  `medium == minimal` at the reference config is intended (2 ubatches of 2 =
  1 batch).
- **Strictly below `max`**: level n reserves n chunks of `limit/ubatch`; since
  `n < ubatch_size` for sane ubatches, no level ever reaches the slot limit —
  no clamping degeneracy at small slots.
- **Ubatch-aligned**: every level is a whole number of ubatches of the slot budget.
- **Explicit budget wins**: client-supplied `thinking_budget_tokens` / `reasoning_budget_tokens` still override the ladder.
- **Zero-budget safety unchanged**: a client budget of 0 still disables thinking and pins the budget at 0.

## Rejected alternatives

- **Exponential runge (`low * 2^(index+1)`): kept the crossover bug and reached the slot limit at `extra high` on small slots (e.g. at 2048/128 the top level hit 8x batch = the whole limit).
- **Batch-anchored** (`n * batch * 2`): batch-absolute, 4x the limit-relative values, and self-clamped into `max` at small slots. Rejected in favor of staying strictly under the slot limit at every config.

## Tests

- Pinned ladder tables updated (8192 and 32768 reference configs).
- New monotonicity test across six slot configs: low < medium < high < extra high < max, all levels strictly below the limit, and minimal pinned to the 1-batch anchor.

## Files

| File | Change |
|---|---|
| `src/llama_broker/daemon.py` | `apply_reasoning_policy` ladder block replaced with the ubatch-chunk formula.
| `tests/test_broker_http.py` | pinned values updated; monotonicity test added.
| this file | design note |
