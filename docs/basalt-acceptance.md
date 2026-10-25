# Basalt runtime acceptance contract

Basalt is an optional runtime adapter with an authenticated TLS bridge and a
supervised native listener. This document specifies acceptance requirements;
it does not attest to a deployment, hardware result or runtime qualification.

The operator supplies the native executable, model, native config, runtime
catalog and profile outside the repository. Pin artifact and config hashes in
that private catalog/profile. Use a native `.basalt` model, set `server.model`
to its absolute path and optionally set `server.alias` to its native model name.
Register `basalt_config_path` and `basalt_config_sha256` in the profile. Choose
resource limits from measurements on the target hardware; repository tests use
small temporary CPU fixtures and cannot determine those limits.

Before deployment, verify TLS with the intended trust roots, backend API-key
rotation, credential isolation, request/response framing, streaming and client
cancellation. Verify bounded deadlines and process cleanup on startup failure,
signals and native/bridge exits. The bridge's request deadline is configurable
within the supported bound; it includes stalled reads and writes.

Validate profile admission, native config/model identity, aliases, MemoryHigh
and hard memory limits. Qualify text, tools and schema responses through the
broker on the actual native runtime. Context capacity and throughput require
separate operator qualification; a configured limit is not a measured result.

The rollback hook dispatches an independent operator-supplied recovery unit.
Its success records dispatch only, not verified restoration. Install and gate
that unit first, restore only a previously verified private configuration and
verify readiness independently. Never restart an owning unit recursively from
its own stop hook. Example units are templates and must be configured locally.
