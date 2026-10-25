# Reasoning budget ladder for effort levels

Status: implemented, tested, and active in the restarted broker service.

## Requirements

- Six ordered enabled tiers: minimal, low, medium, high, extra high, max.
- Derive their spacing from the effective token allowance for this response;
  max equals that allowance, including when the client omits its limit.
- Support 1:1 and nondivisible batch:ubatch settings without assuming that
  batch size must be a multiple of ubatch size.
- Keep the lower tiers useful at ordinary turn limits, and compress the range
  for short responses. No table of fixed percentages per named tier.
- Preserve explicit thinking budgets, explicit thinking disablement, effort
  aliases, and the existing behavior of other runtimes.

## Decision

Use a geometric progression from a minimum allowance to the effective output
limit. Round upwards to whole ubatches when six ubatches fit in the turn.
Use individual tokens for shorter turns so a large physical batch does not
collapse the whole scale.

Let `L` be the validated output limit actually forwarded to the backend, `B`
the logical batch size, `U` the physical ubatch size, and `N=6` enabled tiers.

```python
quantum = U if L >= N * U else 1
nominal_minimum = max(3072, B, N * U)
compressed_minimum = min(nominal_minimum, max(1, L // N))
minimum = ceil(compressed_minimum / quantum) * quantum

# i = 0 .. N - 1
budget = min(L, round_up(minimum * (L / minimum)**(i / (N - 1)), quantum))
```

The implementation compares integer powers and searches quantum counts. It
finds the exact upwards-rounded budget without floating-point root errors.
The max tier is assigned `L` directly, even if `L` is not ubatch-aligned.
Adjacent tiers can coincide when integer or ubatch rounding leaves little room.

The nominal minimum provides at least one logical batch and six physical
ubatches, subject to the short-turn compression. In ubatch units this is
`max(ceil(3072/U), ceil(B/U), N)`. This expresses the batch:ubatch relationship
without assuming an integer ratio. It is a policy choice, not a claim that
larger batches make a model more capable. llama.cpp defines these as the
[logical and physical maximum decode batch sizes](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h).

3072 is the calibration starting point: it exceeds the largest unforced
thinking count in the 9B pilots (2561) and the entire largest response in the
reviewed 27B session (2608). A ceiling permits that work; it does not require
the model to consume the allowance. For short turns, `L // N` compresses the
minimum so the effort levels can still span the available range. This is
derived from the tier count rather than six separately selected percentages.

## Resulting budgets

At a 16384-token turn limit:

| Profile / batch / ubatch | Minimal | Low | Medium | High | Extra high | Max |
|---|---:|---:|---:|---:|---:|---:|
| Stock 9B / 128 / 128 | 2816 | 4096 | 5760 | 8192 | 11648 | 16384 |
| Antidoom 9B / 256 / 64 | 2752 | 3968 | 5632 | 8064 | 11520 | 16384 |
| Qwen 27B / 4096 / 512 | 3072 | 4608 | 6144 | 8704 | 11776 | 16384 |

Other limits, with batch 256 and ubatch 64:

| Turn limit | Minimal | Low | Medium | High | Extra high | Max |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 192 | 256 | 320 | 448 | 512 |
| 8192 | 1408 | 2048 | 2880 | 4096 | 5824 | 8192 |
| 32768 | 3072 | 4992 | 7936 | 12736 | 20416 | 32768 |
| 98304 | 3072 | 6144 | 12288 | 24576 | 49152 | 98304 |

## Calibration evidence and limits

Pi session `01a15a9c-2fcf` used Qwen3.8 27B with ubatch 512. All eight
requests received a 4096-token thinking allowance. Its largest entire output
was 2608 tokens; no thinking response could have reached that allowance.
The long session mainly reflected broad filesystem searches.

The Antidoom and stock pilots each attempted the same 36 trials at caps of
0, 512, 1024, 2048, 4096, and 8192, with output limited to 16384. Prompt reuse
was disabled, the slot fixed, and the model reloaded between distinct fixtures.
The valid Antidoom pilot peaked at 1921 thinking tokens per response. Stock
peaked at 2561 in its higher-cap trials. Stock solved one assignment only at
caps of 2048 and above. Neither model improved consistently as caps increased.

A stock assignment at cap 1024 continued for another 6542 visible tokens and
still answered incorrectly. At cap 512 it timed out after 240 seconds; a
supplemental retry exhausted all 16384 output tokens after 261.65 seconds.
A thinking-channel allowance does not bound reasoning-like work in visible
output. The separate output limit remains essential.

Raw pilots and matched reports are local artifacts under
`generated/benchmarks/thinking-caps-20261020/` and
`generated/benchmarks/thinking-caps-stock-20261020/`. The tool prompt's
`active_file` field is ambiguous, and strict JSON compliance is scored
separately from correct values. These small deterministic fixtures do not
establish an optimal budget or a general model ranking.

## Effective turn limit

For chat/text completions the broker now validates or inserts the output
limit before applying the reasoning policy. Previously an omitted client
limit could produce thinking based on the slot ceiling while the forwarded
response used a smaller default. `max_tokens` and `max_completion_tokens`
are both covered. Oversized or invalid output limits still fail closed.
Explicit thinking budgets retain their existing override behavior.

The Pi configuration provider now emits `maxTokens`, the field read by the
installed Pi 1.3.0 model composer, in place of the ignored `maxOutputTokens`.
The live config was migrated for ten broker models, preserving their numeric
ceilings and all other fields/providers. Its original file has a private
backup; the migration record is in
`generated/benchmarks/automatic-reasoning-scale-20261020/pi-migration.json`.
Pi's own installed composer parsed all ten ceilings correctly. Existing Pi
processes must reload their model configuration to pick up that field change;
the broker policy is already applied to each request's actual forwarded limit.

## Verification

- Pinned full ladders cover actual batch/ubatch configurations, short turns,
  and an exact doubling progression.
- Tests cover ordered positive budgets, exact max, growth with the turn limit,
  batch-dependent minimums, nondivisible ratios, and explicit overrides.
- An exhaustive sweep of the production function checked 163840 ladders /
  983040 budgets: every limit 1..32768 for five batch configurations, testing
  ordering, cap bounds, and nondecreasing budgets as the turn limit grows.
- HTTP tests verify both output-limit spellings and defaulted limits reach
  the policy before forwarding.
- 170 HTTP/profile/shutdown and 20 configuration tests passed.
- 60 live automatic-tier requests across stock 9B, Antidoom, and 27B matched
  the calculated budgets in broker policy events; all returned READY.
- 12 isolated fixtures replayed through automatic minimal without reaching a
  thinking cap. All final answers and generation totals matched the earlier
  4096-cap fixtures (stock 2/6 correct, Antidoom 1/6 correct).
- The installed Pi composer accepted all ten corrected model ceilings, and
  its request generator passed six synthetic-fetch checks with varying
  context lengths. Existing Pi sessions need a model-config reload.
- Runtime, matrix, capability, Pi, and exhaustive evidence is saved under
  `generated/benchmarks/automatic-reasoning-scale-20261020/`; the original
  active model was restored ready.
