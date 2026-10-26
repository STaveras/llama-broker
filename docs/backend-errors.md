# Backend errors in streaming responses

For `stream: true` chat and text completions, the broker sends HTTP 200 and SSE
heartbeats while waiting for model activation and generation. Those headers
cannot be changed if the backend subsequently rejects the request. Clients
must inspect terminal error events rather than treating HTTP 200 as success.

Backend HTTP 4xx/5xx rejections now retain their actionable JSON error fields
inside an `event: error` frame. For example, an oversized llama.cpp request
produces an error like this (identifiers are illustrative):

```text
event: error
data: {"error":{"message":"request (147362 tokens) exceeds the available context size (131072 tokens), try increasing it","type":"exceed_context_size","code":"context_length_exceeded","status_code":400,"upstream_code":400,"prompt_tokens":147362,"context_length":131072,"request_id":"example-request","profile":"example-profile"},"request_id":"example-request","status":400}
```

The connection ends after the error. The broker does not append `[DONE]`,
retry generation, or mark this as a successful completion. A rejected request
releases its profile lease normally; it does not invalidate an otherwise
healthy backend or count as a backend crash.

## Recovery fields

- `message`: the backend's explanation, limited to 4,096 characters.
- `type`, `code`, `param`: the backend's scalar request-error fields when supplied.
- `status_code`: the actual upstream HTTP status, repeated **inside** `error`
  because streaming SDKs may retain only that object when raising an exception.
- `request_id`, `profile`: broker correlation identifiers, also inside `error`.
- `prompt_tokens`, `context_length`: token measurements when the backend supplies
  them, or when they can be extracted from llama.cpp's explicit overflow message.
  Native `n_prompt_tokens`, `n_ctx`, and `n_ctx_slot` fields are retained if present.

Known context-overflow codes/types and llama.cpp's explicit request-versus-context
message are normalized to `code: context_length_exceeded`. If normalization
replaces a different upstream code, the original is retained as `upstream_code`.
Other validation errors keep their own codes; they are not relabeled as overflow.

This lets Hermes classify an oversized prompt as context overflow and enter its
existing compression-and-retry path, rather than retrying a generic proxy failure
and immediately escalating through its provider fallback chain. Compression can
still fail or be unavailable; this change makes the underlying cause available
for the client's existing recovery policy, not a guarantee that every conversation
can be compressed successfully.

## Bounds and unchanged behavior

The broker reads at most 65,537 bytes to detect an error body exceeding its
65,536-byte limit. Reads remain under the existing backend deadline and heartbeat
behavior. Malformed, oversized, or non-JSON error bodies receive a safe
`broker_upstream_error` without reflecting raw HTML or debug payloads. Only the
listed recovery fields are copied; arbitrary backend objects are discarded.

Non-streaming error bodies retain their existing HTTP status/body passthrough.
Transport failures, deadlines, malformed success streams, and backend-crash
handling continue to use their existing broker errors.

## Hermes compatibility probe

With an installed Hermes checkout and its Python dependencies available:

```sh
python tools/check_hermes_error_compat.py --hermes-root /path/to/hermes-agent
```

The probe uses disposable loopback HTTP servers, real broker handlers, the real
OpenAI SDK, and Hermes' actual classifier and overflow-recovery code. The backend
and compression hook are scripted fixtures: no production model, provider
credentials, service transitions, or real summarization is involved. It verifies
HTTP 400 and 500 overflow rejections invoke compression, request a compacted retry,
and complete that fixture retry without requesting provider fallback.
