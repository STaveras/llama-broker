"""Read-only bundle listing contracts; every artifact here is a tiny fixture."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_broker import cli
from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from tests.fixtures import materialize_fixture


def write(root, name, content=b"weights"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
    return path


def hf(root, *, sharded=True):
    write(root, "config.json", {"model_type": "llama"})
    write(root, "tokenizer.json", {"model": {}})
    write(root, "tokenizer_config.json", {"tokenizer_class": "AutoTokenizer"})
    write(root, "README.md", b"fixture model")
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] if sharded else ["model.safetensors"]
    for name in names:
        write(root, name)
    if sharded:
        write(root, "model.safetensors.index.json", {"weight_map": {"a": names[0], "b": names[1], "c": names[0]}})
    return names


def discover(*paths):
    # Import inside the test so RED reports missing behavior, not a collection error.
    from llama_broker.model_bundles import discover_model_bundles

    return discover_model_bundles(paths)


@pytest.mark.parametrize("sharded", [True, False])
def test_hf_weights_and_metadata_are_one_node(tmp_path, sharded):
    root = tmp_path / "model"
    weights = hf(root, sharded=sharded)
    bundles = discover(tmp_path)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["bundle_root"] == str(root)
    assert bundle["path"] == str(root / weights[0])  # A file, never an inferred directory operation target.
    assert bundle["format"] == "hf-safetensors"
    assert bundle["status"] == "complete"
    assert bundle["missing_files"] == []
    names = [item["relative_path"] for item in bundle["files"]]
    assert names == sorted(p.name for p in root.iterdir())
    assert [item["relative_path"] for item in bundle["files"] if item["role"] == "weight"] == weights
    assert bundle["size"] == sum(p.stat().st_size for p in root.iterdir())
    assert bundle["size_policy"] == "unique-inode-per-bundle"


def test_siblings_nested_models_and_quantizations_stay_separate(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b", tmp_path / "a" / "quantized", tmp_path / "a" / "nested" / "other"]
    for root in roots:
        hf(root, sharded=False)
    # Same-directory alternatives must not silently merge either.
    write(roots[0], "model.fp16.safetensors")
    write(roots[0], "model.q4.safetensors")
    bundles = discover(tmp_path)
    assert len(bundles) == 6
    assert {b["bundle_root"] for b in bundles} == {str(root) for root in roots}
    for bundle in bundles:
        assert all("/" not in item["relative_path"] for item in bundle["files"])
        assert len([item for item in bundle["files"] if item["role"] == "weight"]) == 1


def test_indexes_with_subdirectories_do_not_swallow_another_model(tmp_path):
    root = tmp_path / "outer"
    hf(root / "quantized", sharded=False)
    write(root, "weights/part.safetensors")
    write(root, "model.safetensors.index.json", {"weight_map": {"a": "weights/part.safetensors", "b": "quantized/model.safetensors"}})
    bundles = discover(tmp_path)
    outer = next(b for b in bundles if b["bundle_root"] == str(root))
    assert outer["status"] == "incomplete"
    assert "weights/part.safetensors" in [f["relative_path"] for f in outer["files"]]
    assert "quantized/model.safetensors" not in [f["relative_path"] for f in outer["files"]]
    assert outer["issues"]
    assert len(bundles) == 2


def test_gguf_files_and_split_sets_are_independent_of_hf(tmp_path):
    hf(tmp_path / "hf")
    for name in ("alpha.Q4.gguf", "alpha.Q8.gguf", "beta-00001-of-00002.gguf", "beta-00002-of-00002.gguf"):
        write(tmp_path, name)
    bundles = discover(tmp_path)
    ggufs = [b for b in bundles if b["format"] == "gguf"]
    assert len(ggufs) == 3
    assert sorted(len(b["files"]) for b in ggufs) == [1, 1, 2]
    assert all(b["status"] == "complete" for b in bundles)


def test_dedup_overlaps_directory_aliases_and_member_symlinks(tmp_path):
    root = tmp_path / "model"
    hf(root)
    write(root, "tokenizer.model", b"shared")
    (root / "vocab.json").symlink_to("tokenizer.model")
    os.link(root / "tokenizer.model", root / "special_tokens_map.json")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    (root / "loop").symlink_to(tmp_path, target_is_directory=True)
    bundles = discover(tmp_path, root, alias)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert {"vocab.json", "special_tokens_map.json", "tokenizer.model"} <= {f["relative_path"] for f in bundle["files"]}
    assert bundle["size"] == sum(p.stat().st_size for p in root.iterdir() if p.is_file()) - 2 * len(b"shared")
    assert bundles == discover(alias, root, tmp_path)


def test_parent_symlink_cycle_cannot_scan_unrelated_siblings(tmp_path):
    root = tmp_path / "model"
    hf(root, sharded=False)
    hf(tmp_path / "unrelated", sharded=False)
    (root / "loop").symlink_to(tmp_path, target_is_directory=True)
    bundle, = discover(root)
    assert bundle["bundle_root"] == str(root)


def test_hf_cache_keeps_snapshot_names_instead_of_blob_targets(tmp_path):
    repo = tmp_path / "models--org--demo"
    blob = write(repo, "blobs/abcdef", b"weight blob")
    root = repo / "snapshots" / "revision"
    write(root, "config.json", {"architectures": ["DemoModel"]})
    (root / "model.safetensors").symlink_to(blob)
    write(repo, "blobs/unfinished.incomplete", b"partial")
    bundle, = discover(tmp_path)
    assert bundle["bundle_root"] == str(root)
    assert bundle["path"] == str(root / "model.safetensors")
    assert next(f["path"] for f in bundle["files"] if f["role"] == "weight") == str(root / "model.safetensors")


def test_two_snapshots_sharing_blobs_are_distinct_semantic_models(tmp_path):
    blob = write(tmp_path, "blobs/weight", b"shared weights")
    for revision in ("rev-a", "rev-b"):
        root = tmp_path / "snapshots" / revision
        write(root, "config.json", {"model_type": "llama"})
        (root / "model.safetensors").symlink_to(blob)
    bundles = discover(tmp_path)
    assert len(bundles) == 2
    assert {b["bundle_root"] for b in bundles} == {str(tmp_path / "snapshots" / r) for r in ("rev-a", "rev-b")}
    assert bundles[0]["size"] == bundles[1]["size"]


def test_multiple_indexes_in_one_directory_stay_separate(tmp_path):
    write(tmp_path, "config.json", {"model_type": "llama"})
    for quant in ("fp16", "q4"):
        names = [f"{quant}-00001-of-00002.safetensors", f"{quant}-00002-of-00002.safetensors"]
        for name in names:
            write(tmp_path, name)
        write(tmp_path, f"model.{quant}.safetensors.index.json", {"weight_map": dict(enumerate(names))})
    bundles = discover(tmp_path)
    assert len(bundles) == 2
    assert [len([f for f in b["files"] if f["role"] == "index"]) for b in bundles] == [1, 1]
    assert all({f["relative_path"].split("-")[0] for f in b["files"] if f["role"] == "weight"} in ({"fp16"}, {"q4"}) for b in bundles)


@pytest.mark.parametrize("format", ["gguf", "safetensors"])
def test_missing_shard_without_index_is_incomplete(tmp_path, format):
    write(tmp_path, "config.json", {"model_type": "llama"})
    write(tmp_path, f"model-00002-of-00003.{format}")
    bundle, = discover(tmp_path)
    assert bundle["status"] == "incomplete"
    assert bundle["missing_files"] == [f"model-00001-of-00003.{format}", f"model-00003-of-00003.{format}"]


def test_zero_byte_indexed_weights_are_unfinished(tmp_path):
    names = hf(tmp_path)
    write(tmp_path, names[0], b"")
    bundle, = discover(tmp_path)
    assert bundle["status"] == "incomplete"
    assert bundle["missing_files"] == [names[0]]


def test_incomplete_index_missing_broken_and_downloading_weights(tmp_path):
    root = tmp_path / "model"
    names = hf(root)
    (root / names[1]).unlink()
    partial = write(root, names[1] + ".incomplete")
    (root / names[1]).symlink_to(partial)
    write(root, ".cache/download/extra.safetensors")
    bundle, = discover(tmp_path)
    assert bundle["status"] == "incomplete"
    assert bundle["missing_files"] == [names[1]]
    assert names[1] not in [f["relative_path"] for f in bundle["files"]]
    (root / names[1]).unlink()
    (root / names[1]).symlink_to(root / "absent")
    assert discover(tmp_path)[0]["missing_files"] == [names[1]]


def test_explicit_and_aliased_download_directories_are_excluded(tmp_path):
    root = tmp_path / "download.incomplete"
    hf(root)
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    assert discover(root) == []
    assert discover(tmp_path) == []


def test_indexed_symlink_into_partial_download_directory_is_missing(tmp_path):
    root = tmp_path / "model"
    names = hf(root)
    (root / names[1]).unlink()
    target = write(tmp_path, "download.incomplete/weights", b"unfinished")
    (root / names[1]).symlink_to(target)
    bundle, = discover(root)
    assert bundle["status"] == "incomplete"
    assert bundle["missing_files"] == [names[1]]


@pytest.mark.parametrize("index", [b"{broken", {}, {"weight_map": {}}, {"weight_map": {"a": "../escape.safetensors"}}, {"weight_map": {"a": "/outside.safetensors"}}, {"weight_map": {"a": 42}}, {"weight_map": {"a": "config.json"}}, {"weight_map": {"a": "null\u0000.safetensors"}}])
def test_invalid_index_is_explicitly_incomplete(tmp_path, index):
    write(tmp_path, "model.safetensors.index.json", index)
    bundle, = discover(tmp_path)
    assert bundle["status"] == "incomplete"
    assert bundle["issues"]
    assert all(f["role"] != "weight" for f in bundle["files"])


def test_unrelated_and_download_directories_are_excluded(tmp_path):
    write(tmp_path, "unrelated/config.json", {"theme": "dark"})
    write(tmp_path, "unrelated/random.safetensors")
    write(tmp_path, "notes/tokenizer.json", {})
    for name in (".cache/download", "blobs", ".hidden", "download.incomplete"):
        hf(tmp_path / name)
    assert discover(tmp_path) == []


@pytest.mark.parametrize("config", [{"model_type": 42}, {"architectures": "dark-theme"}])
def test_invalid_config_is_not_model_evidence(tmp_path, config):
    write(tmp_path, "config.json", config)
    write(tmp_path, "random.safetensors")
    assert discover(tmp_path) == []


def test_index_cannot_use_symlinked_directory_from_a_sibling_model(tmp_path):
    other = tmp_path / "other"
    write(other, "config.json", {"model_type": "llama"})
    write(other, "weights/model.safetensors")
    root = tmp_path / "model"
    write(root, "model.safetensors.index.json", {"weight_map": {"a": "alias/weights/model.safetensors"}})
    (root / "alias").symlink_to(other, target_is_directory=True)
    bundle = next(b for b in discover(tmp_path) if b["bundle_root"] == str(root))
    assert bundle["status"] == "incomplete"
    assert bundle["issues"]
    assert all(f["role"] != "weight" for f in bundle["files"])


def test_cli_json_lists_artifacts_and_keeps_profiles_separate(tmp_path):
    profiles = materialize_fixture(tmp_path / "fixture")
    root = tmp_path / "local"
    hf(root)
    before = profiles.read_bytes()
    output = io.StringIO()
    with redirect_stdout(output), patch.object(cli, "_broker_admin", side_effect=AssertionError("no service control")):
        assert cli.main(["--profiles", str(profiles), "model", "list", "--scan-path", str(root), "--json"]) == 0
    payload = json.loads(output.getvalue())
    assert payload["schema_version"] == 2
    assert payload["object"] == "model_list"
    assert "profiles" not in payload
    bundle, = payload["data"]
    assert Path(bundle["path"]).is_file()
    assert bundle["bundle_root"] == str(root)

    # Exercise the actual load parser/resolver without permitting backend calls.
    for selector in (bundle["path"], bundle["bundle_root"]):
        with (
            patch.object(cli, "create_production_controller"),
            patch.object(cli, "_broker_admin") as admin,
            redirect_stderr(io.StringIO()),
        ):
            assert cli.main(["--profiles", str(profiles), "load", selector]) == 1
            admin.assert_not_called()


def test_cli_default_tree_verbose_and_json_members_agree(tmp_path):
    profiles = materialize_fixture(tmp_path / "fixture")
    root = tmp_path / "local"
    hf(root)
    outputs = []
    for flags in ([], ["--verbose"], ["--json"]):
        output = io.StringIO()
        with patch.object(cli, "MODELS_DIR", root), redirect_stdout(output):
            assert cli.main(["--profiles", str(profiles), "model", "list", "--width", "100", *flags]) == 0
        outputs.append(output.getvalue())
    normal, verbose, machine = outputs
    assert "local [hf-safetensors, complete]" in normal
    assert "├── config.json" in normal
    assert "└── tokenizer_config.json" in normal
    assert str(root) not in normal
    assert str(root) in verbose
    for member in json.loads(machine)["data"][0]["files"]:
        assert member["relative_path"] in normal
    assert all(len(line) <= 100 for line in normal.splitlines())


def test_listing_does_not_change_setup_file_selection_or_hash_bundles(tmp_path):
    from llama_broker.setup_wizard import build_setup_plan, discover_candidates

    root = tmp_path / "model"
    hf(root)
    runtime = write(tmp_path, "llama-server", b"runtime fixture")
    runtime.chmod(0o755)
    with patch("llama_broker.profile_config.sha256_file", side_effect=AssertionError("listing must not hash")):
        bundle, = discover(root)
    report = discover_candidates([runtime, root])
    assert all(candidate.path.is_file() for candidate in report.models)
    assert Path(bundle["path"]) in [candidate.path for candidate in report.models]
    # Local bundle roots are display data, never auto-selected/imported.
    plan = build_setup_plan(report)
    assert plan.selected_model is None
    assert plan.operations == ()
