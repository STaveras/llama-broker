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
