# Release checklist

Llama Broker 0.1.0 is released from a fresh sanitized orphan snapshot, not from
this private development history.

Before publication:

1. Create a private backup of the exact pre-sanitization checkout.
2. Build the source distribution and wheel with `python -m build`.
3. Run `python tests/run_offline.py` and the clean-install smoke test.
4. Run `python tools/check_public_tree.py` on the sanitized snapshot.
5. Create an immutable Git tag `v0.1.0` on the public snapshot.
6. Publish the GitHub release with source/wheel `SHA256SUMS` and
   `release-manifest.json`.
7. Publish to PyPI through the release workflow's trusted publisher.

The release manifest records the package version, tag, commit, supported Python
versions, and SHA-256 hashes. Signatures and attestations are intentionally not
release gates for 0.1.0.
