# Release policy

`pyproject.toml` is the package-version source. Historical source checkpoints and
their side-branch descendants carry the latest assigned first-parent version.
This labels the snapshot; it does not qualify unmerged changes for publication.
Future development may retain the last stable version or use `X.Y.Z.devN`.
Only an exact `vX.Y.Z` tag matching stable metadata can be a package release.

All tags in `retrospective-tags.json` describe reconstructed source history.
They are reserved for GitHub notes without package uploads or package assets.
The preparation report lists their final targets. None is a verified published
package. Their source versions have been aligned during sanitization.

Historical release workflows are inert. The complete workflow is present in the
newest main/development snapshots and descendants, added during preparation.
This is a deliberate maintenance change, not a claim about historical features.
There is no tag-push publication trigger and no automatic GitHub release job.

To qualify a future package release, bump `project.version`, update the root
package entry in `uv.lock` if present, run CI and the full offline suite, create
the matching local tag at main, build a wheel and sdist, and verify both:

```sh
python tools/check_release.py --development
python -m build
python tools/check_release.py --tag vX.Y.Z --require-head-tag --dist dist
```

CI must reject tag/version mismatches and development versions on release tags.
CI and release qualification use pytest so function-based and unittest tests both
run; only the host-specific `test_local_production_manifest.py` is excluded.
`python tests/run_offline.py` remains a convenient standard-library unittest run.
The release workflow is manually dispatched from main with an explicit tag;
`publish` defaults to false. A publishing run additionally requires a new,
non-retrospective version, `publish=true`, `PYPI_PUBLISH_ENABLED=true`, a configured
`pypi` environment with required reviewers, and a matching PyPI trusted publisher.
Only the publishing job has `id-token: write`. Environment reviewers and the
trusted publisher must be configured separately before enabling publication.
No hosted settings were configured by the local preparation.

## Next-release sequence

The consolidated release uses `0.8.0` in both `pyproject.toml` and the root
`uv.lock` entry. Earlier preparation used `0.8.0.dev0`; the older development
snapshot carried `0.6.16`, while main carried the retrospective, notes-only
`0.7.1`. Release `v0.8.0` requires the matching tag at the qualified main commit.
The sequence below documents the qualification and integration procedure; it
is not a claim that PyPI publication or hardware qualification has occurred.

1. Review the isolated consolidation candidate and its branch/history evidence.
   After explicit integration approval, integrate it into `development` and
   run hosted CI on Python 3.11, 3.12, 3.13 and 3.14. Reconcile any intervening
   development changes and repeat the isolated suite and history scan.
2. Keep unqualified FreeToken, reasoning, GPU deployment and system1 work
   separate. Runtime bundle staging remains experimental; CPU/inert CUDA
   compilation does not establish driver or inference qualification.
3. In a release preparation change, set both version records to `0.8.0`, update
   the unreleased notes, build fresh wheel/sdist artifacts and validate their
   metadata and public contents. Review the exact new history relative to main.
4. After approval, integrate the tested development/release preparation result
   into `main`. Build and validate again from that exact commit. A future
   matching `v0.8.0` tag must point to that main commit; tag creation and manual
   release/publish dispatch are separate operator decisions.

A local development artifact can be checked without creating a tag:

```sh
uv run --extra dev python -m build
uv run python tools/check_release.py --development --dist dist
```

Source-tree scanning does not inspect archives or deleted historical blobs;
review both distributions and all newly reachable history separately. Do not
fix private history by merging and deleting a sensitive file in a later commit.

# Local configuration

Use `brokerctl setup` and the bundled `profiles.example.json` to create local
profiles. `docs/examples/local.env.example` documents portable path overrides;
source it only after adapting paths. Never copy production snapshots into Git.
The runtime catalog and credentials are generated locally. Public examples have
no real credentials or artifact hashes that claim to validate private files.

Migration no longer includes the maintainer's deployment-specific path tables.
It infers ordinary targets from metadata and supports explicit per-run model
overrides through the existing API. Review migration plans and provenance before
applying them on a real installation. Candidate benchmark helper tools remain
experimental and must be configured explicitly before any live execution.
