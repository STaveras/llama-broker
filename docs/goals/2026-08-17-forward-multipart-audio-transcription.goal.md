# `/goal` Prompt: Forward Multipart Audio Transcription Requests

Copy the following prompt into `/goal` from the repository root:

```text
Implement the explicit unchecked TODO in README.md: forward authenticated multipart audio transcription requests through Llama-Broker.

Repository: /home/stan/workspace/llama-broker

Read first:
- README.md, especially the TODO section and API/security contract.
- src/llama_broker/daemon.py, especially BrokerHandler.do_POST, _read_payload, _request_headers, and _proxy.
- tests/test_broker_http.py and the existing broker HTTP regression tests.

Required behavior:

1. Add support for POST /v1/audio/transcriptions.
2. Authenticate with the existing Bearer-key path before reading or forwarding the upload.
3. Accept only bounded multipart/form-data requests. Preserve the existing Content-Length requirement and max_body_bytes limit; reject missing, invalid, oversized, or incomplete bodies with the existing fail-closed JSON error style and an X-Request-ID.
4. Do not parse, rewrite, normalize, log, or persist the multipart body or uploaded media. Forward the exact body bytes and the original Content-Type, including its boundary, through the existing authenticated backend connection.
5. Preserve the existing request-header filtering, backend TLS/credential forwarding, request-ID behavior, response-header behavior, and backend error handling.
6. Use the profile currently loaded by the broker coordinator for this raw-body route, matching the existing missing-model JSON behavior. Do not invent a new multipart model-routing convention or inspect the multipart file contents as part of this TODO. Leave the form fields untouched for the backend.
7. Keep all existing JSON routes and transformations unchanged, including chat/completions input_audio support, output limits, profile leases, drain-on-switch behavior, and loopback-only administrative routes.
8. Add offline regression coverage for:
   - successful multipart forwarding with binary/non-JSON media bytes;
   - exact body and Content-Type preservation at the backend;
   - Bearer authentication and request-ID preservation;
   - missing/invalid credentials failing before backend access;
   - unsupported Content-Type and oversized bodies failing closed;
   - uploaded media not appearing in broker log messages.
9. Update README.md to mark this TODO complete and document the route without exposing secrets or machine-specific details.

Implementation guidance:
- Refactor body-size/Content-Length validation into a raw-body helper that _read_payload can reuse, rather than duplicating validation.
- Keep JSON decoding isolated to JSON routes.
- Prefer a small proxy-signature refactor or a dedicated raw proxy path over passing fake JSON payloads merely to control streaming behavior.
- Use the existing tests/test_broker_http.py fake backend and test fixture seams.
- Do not add dependencies, multipart parsers, disk spooling, or new configuration unless the existing code proves they are necessary.

Verification:
- python3 -m unittest tests.test_broker_http -v
- python3 tests/run_offline.py
- python3 -m py_compile src/llama_broker/*.py tests/test_broker_http.py
- git diff --check

Operational constraints:
- Do not stop, restart, reload, switch, or reconfigure the live broker.
- Do not modify active profiles, credentials, model files, runtime artifacts, or service units.
- Do not install OS-level dependencies.
- Do not push, delete, rename, or change any GitHub repository.
- Report the exact tests and verification results; do not claim completion from inspection alone.
```

## Definition of done

The goal is complete only when the implementation, focused regression tests, full offline suite, compile check, README update, and diff check all pass. A live-service test is neither required nor permitted for this TODO.
