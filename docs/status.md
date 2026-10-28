# Understanding `brokerctl status`

`brokerctl status` shows the primary model and adds information when another
capability is available or something is happening. It does not load models,
perform inference, or change the broker's operating state.

## System 1

When System 1 is disabled or absent, there is no extra section. When its configured
backend passes the existing health check, status adds its model and routing state.
A configured backend that fails the health check is shown as **unavailable** rather
than disappearing. Missing readiness is **unknown**, not assumed healthy or stopped.
Availability and automatic routing are separate: the resident model can serve direct
decision requests with automatic routing disabled.

The following is an illustrative layout, not a reading from a running broker:

```text
System 1:  available — resident-d1 (routing enabled)
S1 usage:  12 direct; 40 routes (since broker start)
S1 active: 1 direct; 0 routes
S1 errors: 1 direct; 2 routes; 3 fallbacks
S1 timing: last 8.5 ms; mean 11.2 ms (routing)
S1 targets: main=28, small=12
```

Idle models show just availability and usage totals. Active counts, errors/fallbacks,
latency and selected targets appear when there is something to report. The target
summary shows the three busiest targets, with a count of additional targets. Use
`brokerctl status --json` for the complete retained target breakdown.

### What the numbers mean

- **Direct:** validated requests forwarded to System 1 through the raw System One
  endpoint or the Decisions API. They are separate from routing selection and
  ordinary primary-model inference. Health probes do not count as model use.
- **Routes:** routing-selection attempts, including single-candidate shortcuts.
  These are selections, not a promise that downstream inference completed.
- **Active:** attempts still in progress. These are not estimates of occupied GPU
  slots, and System 1 counts are independent of primary-model request leases.
- **Errors:** direct backend failures or routing selection errors. Routing can fail
  over successfully after a backend error, so a routing error and a fallback can
  count the same attempt. A deliberate low-confidence fallback is not itself a
  backend error.
- **Fallbacks:** attempts that actually selected a fallback target. An attempt with
  no usable target is a failure, not a successful fallback.
- **Timing:** last and mean routing-selection duration in milliseconds. This is not
  token throughput, a percentile, or the time spent generating the final answer.
- **Targets:** selected profiles, including shortcuts and fallback selections.

Counters are in memory for the current System 1 service lifetime, normally since
broker startup; they reset on restart. No prompts, response bodies, raw error text,
request history, or credentials are retained in these statistics. Target-label
cardinality is bounded: the first 128 distinct labels are retained, and subsequent
unseen labels accumulate in the reserved `[other]` bucket (at most 129 buckets).
Status does not ask the model additional questions to obtain stats.

## Broker activity

Additional lines explain in-flight primary-model requests, ongoing transitions,
operation preflight, and nonempty queues:

```text
Activity:  2 in-flight requests
Changing:  switching to large
Queue:     2 waiting operations
```

Operation counts refer to executions, not callers: coalesced requests for the same
operation do not multiply the queue count. Preflight is identified as preflight;
it is not presented as a model that has already loaded. Live activity is a snapshot
and can change while the command prints.

## Older or unreachable brokers

The existing primary-model fallback remains available when the live broker cannot
be reached. No live activity or System 1 counters are invented from persisted
primary-model state. Older brokers may provide System 1 readiness but no counters;
status can show availability without claiming zero usage. Missing metric values
are reported as unknown. Older transition counters are labeled **pending
transitions**, not asserted to be a count of waiting queue executions.

`--json` stays machine-readable and retains the additive live status payload under
`live`; `--verbose` includes that same payload after the human-readable summary.
The existing command exit-code policy is unchanged.
