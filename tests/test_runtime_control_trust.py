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
    (repo / "main.c").write_text("int main(void) { return 0; }\n")
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
    control, repo, commit = fixture_repo
    # The production bwrap invocation is unshare-all; this fixture proves the
    # actual build cannot use a host secret or connect to loopback.
    (repo / "CMakeLists.txt").write_text((repo / "CMakeLists.txt").read_text().replace(
        'if(EXISTS /etc/passwd)',
        'execute_process(COMMAND sh -c "test -e /etc/passwd && exit 1 || true")\nif(EXISTS /etc/passwd)'))
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "sandbox"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    log = control / "network.log"
    manifest = bundle_build(control, "fixture", commit=commit, jobs=1, log=lambda line: log.open("a").write(line + "\n"))
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
