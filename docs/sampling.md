# Sampling defaults

Different jobs can call for different sampling settings, even with the same
model. You can save two sets of defaults in a llama.cpp profile: one for
**Instruct** requests and one for **Thinking** requests.

These are fallbacks, not restrictions. If your client sends a temperature,
the broker leaves it alone. If the client leaves out top p, the broker fills
that field from the selected profile and mode. Each setting is handled
separately.

## Edit the defaults

Open the profile editor:

```bash
brokerctl profile edit <profile-id>
```

The basic view includes **Sampling defaults**, followed by the individual
Instruct and Thinking settings. Use the arrow keys to choose a field and
Enter to edit it. Press **s** to review and save, or **q** to leave without
saving your draft.

- Choose **Qwen 27B** under Sampling defaults to fill both modes with the
  preset below. It only changes the selected profile.
- Edit any of the twelve values to make your own setup.
- Clear a value to leave that parameter to the runtime when the client omits it.
- Choose **Runtime defaults** to remove both sets of broker defaults.

No preset is applied automatically based on a model's name or filename.
Profiles without sampling defaults keep their existing behavior.

## Qwen 27B preset

| Setting | Instruct | Thinking |
| --- | --- | --- |
| Temperature | 0.7 | 1.0 |
| Top p | 0.80 | 0.95 |
| Top k | 20 | 20 |
| Min p | 0.0 | 0.0 |
| Presence penalty | 1.5 | 0.0 |
| Repetition penalty | 1.0 | 1.0 |

## Which mode gets used?

The broker first resolves the request's reasoning settings, then chooses the
sampling defaults. With no reasoning requested, it uses Instruct. An enabled
thinking request uses Thinking. An explicit thinking override takes precedence
as it does for reasoning itself; a zero thinking budget disables thinking.

For example, with the preset saved:

- A request with no reasoning or sampling settings gets the Instruct values.
- A request with `"reasoning_effort": "high"` and no sampling settings gets the
  Thinking values.
- A thinking request that also sends `"temperature": 0.3` keeps that temperature
  and gets the remaining Thinking defaults.
- Explicit zero values stay zero. A field present with `null` is also left to
  the backend to validate, rather than silently replaced.

Some clients send sampling values even when you have not changed their UI
settings. Those still count as explicit request values.

Saving profile defaults refreshes the broker's catalog without reloading the
model. Requests already using a profile snapshot keep their original settings;
new requests use the updated defaults. This changes neither model files nor
runtime startup arguments.

## Configuration and API fields

The profile stores the settings as:

```json
"sampling_defaults": {
  "instruct": {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0
  },
  "thinking": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0
  }
}
```

The broker adds missing values to the top level of the outgoing JSON body for
`/v1/chat/completions` and `/v1/completions`.

Temperature, top p, and presence penalty use their usual OpenAI-compatible
field names. Top k, min p, and repetition penalty are runtime extensions,
not universal OpenAI parameters. For llama.cpp, the broker sends
`repeat_penalty`, translating the profile's `repetition_penalty` name. An
explicit client `repeat_penalty` wins over the default. A client using
`repetition_penalty` is translated too when profile defaults are configured;
if both spellings are present, `repeat_penalty` wins.

This feature currently supports llama.cpp profiles only. Other runtimes do
not accept the profile setting. Changing a profile to another runtime in the
editor removes the now-inapplicable sampling defaults from the draft.
