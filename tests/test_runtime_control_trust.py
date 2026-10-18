import json
import os
import subprocess
from pathlib import Path

import pytest

from llama_broker.runtime_control import (
    RuntimeControlError,
    bundle_build,
    bundle_list,
    bundle_verify,
    project_add,
    bundle_plan,
    save_projects,
)

PYTHON = os.environ.get("PYTHON", "python3")


@pytest.fixture()
def fixture_repo(tmp_path):
    control = tmp_path / "control"
    repo = control / "fixture"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo, check=True)
    (repo / "main.c").write_text("extern int fixture_b(void); int main(void) { return fixture_b() == 7 ? 0 : 1; }\n")
    (repo / "liba.c").write_text("int fixture_a(void) { return 7; }\n")
    (repo / "libb.c").write_text("extern int fixture_a(void); int fixture_b(void) { return fixture_a(); }\n")
    (repo / "netprobe.c").write_text("#include <arpa/inet.h>\n#include <sys/socket.h>\nint main(void) { int s=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(1)}; inet_pton(AF_INET,\"127.0.0.1\",&a.sin_addr); int ok=connect(s,(struct sockaddr*)&a,sizeof(a))==0; return ok ? 1 : 0; }\n")
    (repo / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(fixture C)
if(EXISTS /etc/passwd)
  message(FATAL_ERROR "host secret visible: /etc/passwd")
endif()
add_library(fixture_a SHARED liba.c)
set_target_properties(fixture_a PROPERTIES OUTPUT_NAME fixture_a SOVERSION 1 VERSION 1.0.0)
add_library(fixture_b SHARED libb.c)
target_link_libraries(fixture_b PRIVATE fixture_a)
set_target_properties(fixture_b PROPERTIES OUTPUT_NAME fixture_b SOVERSION 1 VERSION 1.0.0)
add_executable(llama-server main.c)
target_link_libraries(llama-server PRIVATE fixture_b)
add_executable(network-proof netprobe.c)
add_custom_command(TARGET network-proof POST_BUILD COMMAND network-proof)
"""
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "first"], cwd=repo, check=True)
    (repo / "marker.txt").write_text("second\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "second"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    project_add(control, "fixture", "https://fixture.example/fixture.git", default_branch="main")
    return control, repo, commit


def test_pinned_commit_real_bwrap_cpu_build_and_trust(fixture_repo):
    control, repo, commit = fixture_repo
    log = control / "fixture-build.log"
    manifest = bundle_build(control, "fixture", commit=commit, jobs=2, log=lambda line: log.open("a").write(line + "\n"))
    assert manifest["resolved_commit"] == commit
    assert manifest["network"] is False and manifest["cuda"] is False
    assert log.exists() and "Built target llama-server" in log.read_text()
    assert bundle_verify(control, manifest["id"])["status"] == "ok"
    assert {x["path"] for x in manifest["links"]} >= {"lib/libfixture_a.so", "lib/libfixture_b.so"}
    assert "/etc/passwd" not in log.read_text()


def test_sandbox_has_no_network_and_cannot_read_host_secret(fixture_repo):
    import socket
    control, repo, commit = fixture_repo
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0)); listener.listen(1)
    # A successful host connection makes this a real isolation check, rather
    # than a refused port that would also pass outside a network namespace.
    with socket.create_connection(listener.getsockname(), timeout=2): pass
    port = listener.getsockname()[1]
    probe = repo / "netprobe.c"
    probe.write_text(probe.read_text().replace("htons(1)", f"htons({port})"))
    # The production bwrap invocation is unshare-all; this fixture proves the
    # actual build cannot use a host secret or connect to loopback.
    (repo / "CMakeLists.txt").write_text((repo / "CMakeLists.txt").read_text().replace(
        'if(EXISTS /etc/passwd)',
        'execute_process(COMMAND sh -c "test -e /etc/passwd && exit 1 || true")\nif(EXISTS /etc/passwd)'))
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "sandbox"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    log = control / "network.log"
    try:
        manifest = bundle_build(control, "fixture", commit=commit, jobs=1, log=lambda line: log.open("a").write(line + "\n"))
    finally: listener.close()
    assert manifest["network"] is False
    assert log.exists() and "Built target network-proof" in log.read_text()


def test_project_path_and_manifest_traversal_or_anchor_tampering_rejected(fixture_repo):
    control, repo, commit = fixture_repo
    item = json.loads((control / "projects.json").read_text())
    item[0]["path"] = str(control / "../outside")
    (control / "projects.json").write_text(json.dumps(item))
    with pytest.raises(RuntimeControlError):
        bundle_build(control, "fixture", commit=commit)
    project_add(control, "fixture2", "https://fixture.example/two.git", default_branch="main")
    save_projects(control, [dict(x, path=str(control / x["name"])) for x in json.loads((control / "projects.json").read_text())])
    with pytest.raises(RuntimeControlError):
        bundle_list(control / "../control")


def test_manifest_tamper_anchor_is_rejected(fixture_repo):
    control, repo, commit = fixture_repo
    manifest = bundle_build(control, "fixture", commit=commit, jobs=1)
    path = control / "bundles" / manifest["id"] / "manifest.json"
    path.write_text(path.read_text().replace('"network": false', '"network": true'))
    with pytest.raises(RuntimeControlError, match="trust anchor"):
        bundle_list(control)


def test_symlink_chain_is_packaged_without_dereferencing_or_threefold_storage(fixture_repo):
    control, repo, commit = fixture_repo
    manifest = bundle_build(control, "fixture", commit=commit, jobs=1)
    libdir = control / "bundles" / manifest["id"] / "lib"
    regular = [p for p in libdir.iterdir() if p.is_file() and not p.is_symlink()]
    assert len(regular) == 2  # one SONAME target per library; links are not duplicated
    assert all(p.is_symlink() for p in libdir.glob("*.so"))
    assert bundle_verify(control, manifest["id"])["status"] == "ok"


def test_missing_intended_server_never_packages_compiler_probe(fixture_repo):
    control, repo, _ = fixture_repo
    (repo / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(missing C)\n"
        "add_custom_target(llama-server ALL COMMAND ${CMAKE_COMMAND} -E echo no-server)\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "missing server"], cwd=repo, check=True)
    with pytest.raises(RuntimeControlError, match="llama-server"):
        bundle_build(control, "fixture", branch="main")
    assert not (control / "bundles" / "index.json").exists()


def test_plan_and_manifest_pin_actual_default_ref_configuration_and_tools(fixture_repo):
    control, repo, commit = fixture_repo
    plan = bundle_plan(control, "fixture")
    assert plan["resolved_commit"] == commit
    assert plan["ref"] == "main"
    assert plan["toolchain"]["c"]["path"].startswith("/usr/")
    assert plan["toolchain"]["c"]["version"]
    assert plan["toolchain"]["linker"]["version"]
    assert plan["toolchain"]["assembler"]["version"]
    manifest = bundle_build(control, "fixture")
    assert manifest["ref"] == "main" and manifest["branch"] == "main"
    assert manifest["configure_flags"] == plan["configure_flags"]
    assert manifest["toolchain"] == plan["toolchain"]
    assert manifest["id"] == plan["id"]
    assert manifest["architecture"] == plan["architecture"]
    assert manifest["recipe"] == plan["build_recipe"]
    assert manifest["schema_version"] == 2


def test_transitive_packaged_closure_runs_without_build_tree_or_host_search(fixture_repo):
    from llama_broker import runtime_control as rc
    control, repo, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    assert m["dependencies"]["mode"] == "verified-elf-closure-v1"
    assert "libfixture_b.so.1" in m["dependencies"]["objects"]["bin/llama-server"]["needed"]
    assert "libfixture_a.so.1" in m["dependencies"]["objects"]["lib/libfixture_b.so.1.0.0"]["needed"]
    args = [rc.shutil.which("bwrap"), "--unshare-all", "--die-with-parent", "--tmpfs", "/",
            "--proc", "/proc", "--ro-bind", str(control / "bundles" / m["id"]), "/bundle", "--clearenv"]
    for record in m["dependencies"]["host"].values():
        args += ["--ro-bind", record["path"], record["path"]]
    subprocess.run([*args, "/bundle/bin/llama-server"], check=True, capture_output=True)


def reanchor(control, m):
    import hashlib
    path = control / "bundles" / m["id"] / "manifest.json"
    path.write_text(json.dumps(m))
    index = control / "bundles/index.json"
    data = json.loads(index.read_text())
    data["bundles"][m["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    index.write_text(json.dumps(data))


@pytest.mark.parametrize("field,value", [
    ("files", []), ("files", {"/etc/passwd": "0" * 64}),
    ("files", {"../outside": "0" * 64}), ("files", {"lib/../../outside": "0" * 64}),
    ("files", {"bin/llama-server": False}), ("files", {"bin/llama-server": "oops"}),
    ("links", {}), ("links", [None]), ("links", [{"path": "lib/x", "target": "../../outside"}]),
    ("links", [{"path": "/outside", "target": "a"}]),
    ("links", [{"path": "lib/x", "target": "/etc/passwd"}]),
    ("dependencies", []), ("dependencies", {"mode": "readelf+ldconfig", "needed": []}),
    ("schema_version", True), ("schema_version", 1), ("cuda", "false"),
    ("network", True), ("recipe", "arbitrary"), ("resolved_commit", "HEAD"),
    ("architecture", []), ("toolchain", {"c": {"path": "/etc/passwd"}}),
    ("configure_flags", [False]), ("source_url", 42), ("branch", []), ("ref", None),
])
def test_nested_manifest_rejected_before_consumption(fixture_repo, field, value):
    control, _, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    m[field] = value
    reanchor(control, m)
    with pytest.raises(RuntimeControlError):
        bundle_list(control)


def test_declared_regular_file_cannot_read_through_parent_symlink(fixture_repo):
    from llama_broker import runtime_control as rc
    control, _, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    outside = control / "outside"
    outside.mkdir(); (outside / "secret").write_text("never hash this")
    bundle = control / "bundles" / m["id"]
    (bundle / "escape").symlink_to(outside, target_is_directory=True)
    m["files"]["escape/secret"] = "0" * 64
    reanchor(control, m)
    original = rc._sha
    def guarded(path):
        assert path.resolve() != outside / "secret", "arbitrary external read"
        return original(path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rc, "_sha", guarded)
        with pytest.raises(RuntimeControlError):
            bundle_verify(control, m["id"])


def test_index_write_failure_rolls_back_bundle_and_preserves_prior_index(fixture_repo, monkeypatch):
    from llama_broker import runtime_control as rc
    control, repo, commit = fixture_repo
    first = bundle_build(control, "fixture", commit=commit)
    before = (control / "bundles/index.json").read_bytes()
    (repo / "marker.txt").write_text("new commit")
    subprocess.run(["git", "add", "marker.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "new"], cwd=repo, check=True)
    replace = rc.os.replace; write = Path.write_text
    def fail_replace(src, dest):
        if Path(dest).name == "index.json": raise OSError("injected index failure")
        return replace(src, dest)
    def fail_write(path, *args, **kwargs):
        if path.name == "index.json": raise OSError("injected index failure")
        return write(path, *args, **kwargs)
    monkeypatch.setattr(rc.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises((OSError, RuntimeControlError), match="index failure"):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        bundle_build(control, "fixture", commit=head)
    assert (control / "bundles/index.json").read_bytes() == before
    assert {p.name for p in (control / "bundles").iterdir() if p.is_dir()} == {first["id"]}


def _concurrent_builder(control, name, barrier, queue):
    from llama_broker import runtime_control as rc
    import time
    original = rc._dependency_inventory
    read = Path.read_text
    def inventory(*args):
        result = original(*args)
        barrier.wait(timeout=30)
        return result
    def slow_read(path, *args, **kwargs):
        result = read(path, *args, **kwargs)
        if path.name == "index.json": time.sleep(0.2)
        return result
    rc._dependency_inventory = inventory
    Path.read_text = slow_read
    try: queue.put((True, bundle_build(control, name, branch="main")["id"]))
    except Exception as exc: queue.put((False, repr(exc)))


def test_multiprocess_builders_preserve_every_trust_anchor(fixture_repo):
    import multiprocessing
    import shutil
    control, repo, _ = fixture_repo
    bundle_build(control, "fixture", branch="main")
    for n in range(4):
        shutil.copytree(repo, control / f"fixture{n}", ignore=shutil.ignore_patterns(".broker-build"))
        project_add(control, f"fixture{n}", "https://fixture.example/fixture.git", default_branch="main")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4); queue = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_builder, args=(control, f"fixture{n}", barrier, queue)) for n in range(4)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(45)
        if worker.is_alive(): worker.kill(); worker.join(); pytest.fail("builder hung")
        assert worker.exitcode == 0
    results = [queue.get(timeout=2) for _ in workers]
    assert all(ok for ok, _ in results), results
    assert len(bundle_list(control)) == 5


def test_build_uses_committed_snapshot_and_leaves_checkout_unchanged(fixture_repo):
    control, repo, commit = fixture_repo
    before = subprocess.check_output(["git", "symbolic-ref", "HEAD"], cwd=repo)
    (repo / "main.c").write_text("#error dirty source must not compile\n")
    m = bundle_build(control, "fixture", commit=commit)
    assert subprocess.check_output(["git", "symbolic-ref", "HEAD"], cwd=repo) == before
    assert "dirty source" in (repo / "main.c").read_text()
    assert bundle_verify(control, m["id"])["status"] == "ok"


def test_full_project_name_participates_in_bounded_configuration_identity(fixture_repo):
    from llama_broker import runtime_control as rc
    control, _, _ = fixture_repo
    plan = bundle_plan(control, "fixture")
    args = [plan["source_url"], plan["resolved_commit"], plan["build_recipe"], plan["configure_flags"], plan["architecture"], plan["toolchain"]]
    a = rc._identity("a" * 100 + "x", *args)
    b = rc._identity("a" * 100 + "y", *args)
    assert len(a) <= 128 and a != b


def test_system_cpp_dependencies_are_packaged_transitively_and_run(fixture_repo):
    from llama_broker import runtime_control as rc
    control, repo, _ = fixture_repo
    (repo / "main.cpp").write_text('#include <iostream>\nint main() { std::cout << "inert\\n"; }\n')
    (repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\nproject(inert CXX)\nadd_executable(llama-server main.cpp)\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "inert C++"], cwd=repo, check=True)
    m = bundle_build(control, "fixture")
    assert "lib/libstdc++.so.6" in {link["path"] for link in m["links"]}
    assert any("libgcc_s" in name for name in m["files"])
    args = [rc.shutil.which("bwrap"), "--unshare-all", "--tmpfs", "/", "--proc", "/proc",
            "--ro-bind", str(control / "bundles" / m["id"]), "/bundle", "--clearenv"]
    for record in m["dependencies"]["host"].values(): args += ["--ro-bind", record["path"], record["path"]]
    result = subprocess.run([*args, "/bundle/bin/llama-server"], check=True, capture_output=True, text=True)
    assert result.stdout == "inert\n"


def _crash_after_bundle_rename(control):
    from llama_broker import runtime_control as rc
    replace = rc.os.replace
    def crash(src, dest):
        replace(src, dest)
        if Path(src).name.startswith(".bundle-"): os._exit(77)
    rc.os.replace = crash
    bundle_build(control, "fixture")


def test_killed_publisher_is_rolled_back_before_next_catalog_read(fixture_repo):
    import multiprocessing
    control, repo, _ = fixture_repo
    first = bundle_build(control, "fixture")
    (repo / "marker.txt").write_text("next")
    subprocess.run(["git", "add", "marker.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "next"], cwd=repo, check=True)
    child = multiprocessing.get_context("spawn").Process(target=_crash_after_bundle_rename, args=(control,))
    child.start(); child.join(30)
    assert child.exitcode == 77
    assert [m["id"] for m in bundle_list(control)] == [first["id"]]
    assert {p.name for p in (control / "bundles").iterdir() if p.is_dir()} == {first["id"]}


def test_verify_recomputes_dependency_facts_without_executing_artifact(fixture_repo, monkeypatch):
    from llama_broker import runtime_control as rc
    control, _, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    m["dependencies"]["objects"]["bin/llama-server"]["needed"] = []
    reanchor(control, m)
    run = rc.subprocess.run
    def only_readelf(argv, *args, **kwargs):
        assert argv[0] == "/usr/bin/readelf", "verification executed an artifact or used ldd"
        return run(argv, *args, **kwargs)
    monkeypatch.setattr(rc.subprocess, "run", only_readelf)
    result = bundle_verify(control, m["id"])
    assert result["status"] == "invalid" and "dependencies" in result["errors"]


def test_malformed_nested_interpreter_reports_controlled_error(fixture_repo):
    control, _, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    m["dependencies"]["objects"]["bin/llama-server"]["interpreter"] = {}
    reanchor(control, m)
    with pytest.raises(RuntimeControlError): bundle_list(control)


def test_failure_after_index_replace_restores_prior_publication(fixture_repo, monkeypatch):
    from llama_broker import runtime_control as rc
    control, repo, _ = fixture_repo
    first = bundle_build(control, "fixture")
    before = (control / "bundles/index.json").read_bytes()
    (repo / "marker.txt").write_text("next")
    subprocess.run(["git", "add", "marker.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "next"], cwd=repo, check=True)
    replace = rc.os.replace
    def fail_after_replace(src, dest):
        replace(src, dest)
        if Path(dest).name == "index.json": raise OSError("post-replace failure")
    monkeypatch.setattr(rc.os, "replace", fail_after_replace)
    with pytest.raises(RuntimeControlError, match="post-replace failure"):
        bundle_build(control, "fixture")
    assert (control / "bundles/index.json").read_bytes() == before
    assert {p.name for p in (control / "bundles").iterdir() if p.is_dir()} == {first["id"]}


def test_loader_search_must_actually_reach_packaged_libraries(fixture_repo):
    control, repo, _ = fixture_repo
    with (repo / "CMakeLists.txt").open("a") as out:
        out.write('set_target_properties(llama-server PROPERTIES INSTALL_RPATH "$ORIGIN")\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "insufficient loader search"], cwd=repo, check=True)
    with pytest.raises(RuntimeControlError, match="RPATH|search"):
        bundle_build(control, "fixture")


def test_source_cannot_silently_override_reviewed_recipe(fixture_repo):
    control, repo, _ = fixture_repo
    with (repo / "CMakeLists.txt").open("a") as out:
        out.write('set(GGML_CUDA ON CACHE BOOL "" FORCE)\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "override recipe"], cwd=repo, check=True)
    with pytest.raises(RuntimeControlError, match="configuration|recipe"):
        bundle_build(control, "fixture")


def test_fetched_default_ref_records_remote_ref_and_plan_uses_it(fixture_repo):
    from llama_broker import runtime_control as rc
    control, repo, commit = fixture_repo
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True)
    item = rc.project_fetch(control, "fixture")
    assert item["resolved_default_ref"] == "refs/remotes/origin/main"
    plan = rc.bundle_plan(control, "fixture")
    assert plan["ref"] == item["resolved_default_ref"] and plan["resolved_commit"] == commit


def test_interactively_selected_commit_plans_exact_commit(fixture_repo, monkeypatch, capsys):
    from llama_broker import cli
    control, _, commit = fixture_repo
    monkeypatch.setattr(cli, "select_source", lambda *_: {"branch": "main", "commit": commit})
    assert cli.main(["runtime", "--control-root", str(control), "bundle", "plan", "fixture", "--select"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_commit"] == plan["ref"] == commit and plan["selection"] == "commit"


@pytest.mark.parametrize("kind", ["server-link", "library-link", "library-parent", "fifo", "manifest-link", "index-link"])
def test_actual_unsafe_filesystem_entries_cannot_be_consumed(fixture_repo, kind):
    import shutil
    control, _, commit = fixture_repo
    m = bundle_build(control, "fixture", commit=commit)
    bundle = control / "bundles" / m["id"]
    outside = control / "outside"; outside.mkdir()
    (outside / "secret").write_text("private")
    if kind == "server-link":
        path = bundle / "bin/llama-server"; path.unlink(); path.symlink_to(outside / "secret")
    elif kind == "library-link":
        path = bundle / m["links"][0]["path"]; path.unlink(); path.symlink_to(outside / "secret")
    elif kind == "library-parent":
        shutil.rmtree(bundle / "lib"); (bundle / "lib").symlink_to(outside, target_is_directory=True)
    elif kind == "fifo":
        path = bundle / "bin/llama-server"; path.unlink(); os.mkfifo(path)
    else:
        path = bundle / "manifest.json" if kind == "manifest-link" else control / "bundles/index.json"
        path.unlink(); path.symlink_to(outside / "secret")
    with pytest.raises(RuntimeControlError): bundle_verify(control, m["id"])


def test_identical_concurrent_builders_publish_once_without_mixing_outputs(fixture_repo):
    import multiprocessing
    control, repo, _ = fixture_repo
    ctx = multiprocessing.get_context("spawn"); barrier = ctx.Barrier(2); queue = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_builder, args=(control, "fixture", barrier, queue)) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(30)
        if worker.is_alive(): worker.kill(); worker.join(); pytest.fail("builder hung")
        assert worker.exitcode == 0
    results = [queue.get(timeout=2) for _ in workers]
    assert sum(ok for ok, _ in results) == 1, results
    assert "immutable bundle" in next(msg for ok, msg in results if not ok)
    manifests = bundle_list(control)
    assert len(manifests) == 1 and bundle_verify(control, manifests[0]["id"])["status"] == "ok"
    assert subprocess.check_output(["git", "symbolic-ref", "HEAD"], cwd=repo).strip() == b"refs/heads/main"


def test_foreign_elf_machine_cannot_claim_host_architecture(fixture_repo):
    import platform
    control, repo, _ = fixture_repo
    other_machine = 183 if platform.machine() == "x86_64" else 62
    (repo / "patch-elf.py").write_text(
        "import pathlib, sys\np=pathlib.Path(sys.argv[1]); b=bytearray(p.read_bytes()); "
        f"b[18:20]=({other_machine}).to_bytes(2,'little'); p.write_bytes(b)\n"
    )
    with (repo / "CMakeLists.txt").open("a") as out:
        out.write("add_custom_command(TARGET llama-server POST_BUILD COMMAND /usr/bin/python3 /tmp/src/patch-elf.py $<TARGET_FILE:llama-server>)\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "foreign machine fixture"], cwd=repo, check=True)
    with pytest.raises(RuntimeControlError, match="machine|architecture"):
        bundle_build(control, "fixture")


def _list_in_child(control, queue):
    try: bundle_list(control)
    except RuntimeControlError: queue.put("controlled")
    except Exception as exc: queue.put(type(exc).__name__)
    else: queue.put("accepted")


@pytest.mark.parametrize("entry", [".pending.json", ".lock"])
def test_catalog_control_files_must_be_regular_without_blocking(fixture_repo, entry):
    import multiprocessing
    control, _, _ = fixture_repo
    bundle_build(control, "fixture")
    path = control / "bundles" / entry
    path.unlink(missing_ok=True); os.mkfifo(path)
    ctx = multiprocessing.get_context("spawn"); queue = ctx.Queue()
    child = ctx.Process(target=_list_in_child, args=(control, queue))
    child.start(); child.join(3)
    if child.is_alive():
        child.kill(); child.join(); pytest.fail("catalog read blocked on a nonregular control file")
    assert child.exitcode == 0 and queue.get(timeout=2) == "controlled"
