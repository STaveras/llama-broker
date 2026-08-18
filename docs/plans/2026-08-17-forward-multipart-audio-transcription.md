# Multipart Audio Transcription Forwarding Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Complete the explicit README TODO by forwarding authenticated, bounded `multipart/form-data` requests to `/v1/audio/transcriptions` without JSON parsing, media logging, or changes to existing broker routing behavior.

**Architecture:** Keep `BrokerHandler` as the authentication, request-boundary, profile-lease, and proxy authority. Split raw body framing/size validation from JSON decoding, then add a narrow raw-body route for audio transcription. The route will select the profile currently loaded by the coordinator, preserve the multipart bytes and `Content-Type` unchanged, and reuse the existing backend TLS, credential forwarding, response handling, request IDs, and fail-closed errors.

**Tech Stack:** Python 3.11 standard library, `http.server`, `http.client`, existing `BrokerApplication`/`ProfileCoordinator`, `unittest`, and the in-process HTTP backend fixture.

---

## Scope and contract

The README's explicit TODO is the source requirement:

- `POST /v1/audio/transcriptions` accepts authenticated `multipart/form-data`.
- The raw upload body and original `Content-Type` (including boundary) reach the backend unchanged.
- Authentication occurs before upload forwarding.
- Body size remains bounded by `Content-Length` and `max_body_bytes`.
- Request IDs and fail-closed errors remain intact.
- Uploaded media is never logged or persisted by the broker.
- JSON `input_audio` through `/v1/chat/completions` remains unchanged.

### Deliberate boundary for this TODO

The broker will not parse multipart fields or inspect the uploaded media. It will select the same currently loaded profile used by `route_payload()` when a JSON request omits `model`, then forward the multipart form—including any `model` form field—unchanged to the backend. Introducing model selection from multipart form fields is a separate routing design and must not be smuggled into this forwarding fix.

Chunked uploads remain unsupported because the broker's existing bounded request contract requires `Content-Length`. This plan does not add disk spooling, streaming upload support, a multipart parser, or new dependencies.

## Current implementation seams

- `src/llama_broker/daemon.py:1730` — `_read_payload()` currently combines Content-Length validation, body reading, and JSON decoding. Extract raw-body validation so JSON and multipart routes share the same bounds.
- `src/llama_broker/daemon.py:1534` — `BrokerHandler.do_POST()` authenticates `/v1/*`, then unconditionally calls `_read_payload()`.
- `src/llama_broker/daemon.py:1751` — `_request_headers()` already removes hop-by-hop request headers, preserves `Content-Type`, and regenerates `Content-Length` from the forwarded body.
- `src/llama_broker/daemon.py:1761` — `_proxy()` forwards a body through the existing backend TLS/credential connection and preserves backend response headers while adding broker request metadata.
- `tests/test_broker_http.py:68` — the fake backend currently assumes every POST body is JSON; extend it with a raw-body capture branch for the audio route.
- `tests/test_broker_http.py:260` — existing `_post()` helper covers JSON requests; add a raw request helper or a focused multipart helper without weakening the existing JSON tests.
- `README.md:136` — mark the explicit TODO complete after implementation and verification.

---

## Implementation tasks

### Task 1: Extract bounded raw-body reading

**Objective:** Make Content-Length and maximum-body validation reusable without changing current JSON error behavior.

**Files:**
- Modify: `src/llama_broker/daemon.py` near `_read_payload()`.
- Test: `tests/test_broker_http.py` existing body-limit/error tests.

**Steps:**

1. Add a private raw-body helper that:
   - requires `Content-Length`;
   - parses a non-negative integer;
   - rejects lengths greater than `application.max_body_bytes`;
   - reads exactly that many bytes;
   - rejects a short read as an invalid/incomplete request;
   - returns bytes without decoding or logging them.
2. Refactor `_read_payload()` to call the helper and retain its current UTF-8/JSON/object validation messages.
3. Run the existing HTTP tests.

**Verification:**

```bash
python3 -m unittest tests.test_broker_http -v
```

Expected: existing tests pass with no behavior change to JSON routes.

### Task 2: Add authenticated multipart route admission

**Objective:** Route only the intended audio endpoint into the raw-body path after authentication.

**Files:**
- Modify: `src/llama_broker/daemon.py:1534-1564`.
- Test: `tests/test_broker_http.py`.

**Steps:**

1. Keep `_require_api_auth()` before all body reads.
2. Recognize only `POST /v1/audio/transcriptions` as a raw multipart route.
3. Require a `Content-Type` beginning with `multipart/form-data`; reject missing or other media types with the established JSON error envelope, a suitable `4xx` status, and `X-Request-ID`.
4. Read the bounded raw body through the shared helper.
5. Resolve the currently loaded profile from the broker coordinator; do not parse the multipart body for a model or media content.
6. Apply the same artifact-availability check used by JSON routes. Do not start a broken/unverifiable profile.
7. Lease the resolved profile through the existing coordinator before proxying, preserving drain-on-switch and transient restore behavior.
8. Leave all other POST paths on the existing JSON path.

**Verification:** Add a test that a multipart request reaches the new route while a JSON request still follows the existing route.

### Task 3: Reuse proxy behavior for opaque request bodies

**Objective:** Forward raw multipart bytes and headers while retaining existing backend connection and response semantics.

**Files:**
- Modify: `src/llama_broker/daemon.py` around `_proxy()` and its call sites.
- Test: `tests/test_broker_http.py`.

**Steps:**

1. Separate the proxy's response-streaming decision from JSON payload inspection. A raw audio request should not require a decoded payload merely to decide whether the response is streamed.
2. Either:
   - introduce a small `_proxy_raw(body, profile, stream=False)` wrapper that delegates to a shared proxy implementation; or
   - change `_proxy()` to receive an explicit stream flag while keeping the JSON call site behavior unchanged.
3. Ensure `_request_headers()` continues to preserve the exact `Content-Type` header and all non-hop-by-hop headers, including the authenticated backend credential and request ID, while setting the bounded forwarded `Content-Length`.
4. Do not add body contents to `_log()` calls, exception messages, request metadata, or debug output.
5. Forward backend response status, content type, body, and errors through the same path as existing JSON requests.

**Verification:** The fake backend must observe byte-for-byte equality between the client multipart body and the forwarded body, plus exact Content-Type equality.

### Task 4: Extend the fake backend for raw multipart capture

**Objective:** Test binary upload forwarding without weakening existing JSON test coverage.

**Files:**
- Modify: `tests/test_broker_http.py:68-105`.
- Modify: `tests/test_broker_http.py:220-226` to add raw-body/header capture collections.

**Steps:**

1. In `BackendHandler.do_POST()`, branch on `/v1/audio/transcriptions` before JSON decoding.
2. Read the declared body length as bytes.
3. Store the raw body and request headers on the test server.
4. Return a small JSON transcription response such as `{ "text": "fixture transcription" }`.
5. Keep the existing JSON decoding and streaming behavior for all other POST paths.

**Verification:** Existing JSON tests continue to pass, and the backend fixture can prove that a body containing non-JSON/binary bytes is forwarded intact.

### Task 5: Add the successful multipart regression test

**Objective:** Prove the complete authenticated forwarding contract.

**Files:**
- Modify: `tests/test_broker_http.py`.

**Test shape:**

1. Construct a deterministic multipart body with a fixed boundary, text form fields, and a file part containing bytes that are not valid JSON (including at least one binary byte).
2. Send it to `/v1/audio/transcriptions` with:
   - `Authorization: Bearer native-key`;
   - the exact `Content-Type` including boundary;
   - a caller-supplied `X-Request-ID`.
3. Assert HTTP `200` and the backend transcription JSON response.
4. Assert the fake backend received the exact original bytes.
5. Assert the backend received the exact Content-Type, Authorization, and request ID.
6. Assert the broker response carries the supplied request ID.
7. Assert the coordinator used the currently loaded profile and existing lease path.

**Verification:**

```bash
python3 -m unittest tests.test_broker_http.BrokerHttpTests.test_multipart_audio_transcription_is_forwarded -v
```

Expected: PASS, with no JSON decode error.

### Task 6: Add fail-closed and media-redaction regressions

**Objective:** Prove malformed or unauthorized uploads cannot reach the backend and uploaded bytes never enter logs.

**Files:**
- Modify: `tests/test_broker_http.py`.

**Tests:**

1. Missing Authorization: assert `401`, Bearer challenge/request ID, and no backend capture.
2. Wrong Authorization: assert `401` and no backend capture.
3. Missing or non-multipart Content-Type: assert the selected `4xx` error envelope, request ID, and no backend capture.
4. Content-Length above the configured test limit: assert bounded failure, request ID, and no backend capture.
5. Short body relative to Content-Length: assert fail-closed failure and no backend capture.
6. Patch/capture the broker logging function while sending a unique binary marker; assert the marker is absent from every broker log message.

Do not assert implementation-specific private helper calls. Assert only externally visible status, headers, response shape, backend observations, and log absence.

### Task 7: Update the README contract

**Objective:** Make the completed route discoverable and remove the stale TODO.

**Files:**
- Modify: `README.md:125-162`.

**Steps:**

1. Add `POST /v1/audio/transcriptions` to the stable API table with its authenticated multipart contract.
2. Change the TODO checkbox to `[x]` and replace the problem statement with a concise completed-behavior note, or remove the TODO section if no other explicit TODO remains.
3. State that the broker preserves multipart bytes and Content-Type, requires bounded Content-Length, and does not log uploaded media.
4. Preserve the existing note that JSON `input_audio` through chat completions remains supported.
5. Do not add real paths, credentials, active model aliases, or machine-specific examples.

### Task 8: Run the complete verification gate

**Objective:** Prove the change is offline-safe and does not regress the broker HTTP contract.

**Commands:**

```bash
python3 -m unittest tests.test_broker_http -v
python3 tests/run_offline.py
python3 -m py_compile src/llama_broker/*.py tests/test_broker_http.py
python3 -m compileall -q src tests
git diff --check
```

Expected:

- Focused broker HTTP tests pass.
- Full offline suite passes.
- Python compilation succeeds.
- `git diff --check` emits no findings.
- No model, GPU, systemd, credential, or live-service operation occurs.

After implementation, inspect the final diff for accidental body logging, multipart normalization, changes to JSON route behavior, or changes outside the scoped source/test/README files.

---

## Acceptance checklist

- [ ] Authenticated `POST /v1/audio/transcriptions` accepts bounded multipart bodies.
- [ ] Authentication happens before body forwarding.
- [ ] Exact multipart bytes reach the backend unchanged.
- [ ] Original multipart Content-Type and boundary are preserved.
- [ ] Backend credentials and request IDs retain existing behavior.
- [ ] Default-profile lease and artifact checks remain fail-closed.
- [ ] Invalid media type, missing/invalid auth, oversized body, and short body fail closed.
- [ ] Uploaded media never appears in broker logs or persisted state.
- [ ] Existing JSON endpoints and JSON `input_audio` behavior remain unchanged.
- [ ] Focused tests, offline suite, compile checks, and diff checks pass.
- [ ] README no longer presents this behavior as incomplete.

## Risks and non-goals

- No multipart parser is added; multipart form fields are opaque to the broker.
- No model routing based on a multipart `model` field is introduced.
- No chunked upload support is added.
- No disk-backed upload buffering is added.
- No changes are made to live service units, credentials, profiles, models, runtimes, or Git remotes.
- A future model-routing requirement for audio profiles should be a separate design and test task rather than hidden in this forwarding fix.
