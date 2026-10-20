import subprocess
from pathlib import Path
import shutil
import pytest
from unittest.mock import patch
from llama_broker import runtime_control as rc

def test_recipe_disables_actual_subprocess_option():
    flags,_=rc._cmake_recipe('cmake-cuda',cuda_arch='120a')
    assert '-DLLAMA_SUBPROCESS=OFF' in flags

def test_blackwell_supports_verified_cuda_134():
    version=subprocess.CompletedProcess([],0,'Cuda compilation tools, release 13.4, V13.4.92','')
    with patch.object(rc.shutil,'which',return_value='/usr/local/cuda-13.4/bin/nvcc'),patch.object(rc.subprocess,'run',return_value=version):
        assert rc._cuda_toolchain_check() is None

def test_blackwell_rejects_cuda_before_128():
    version=subprocess.CompletedProcess([],0,'Cuda compilation tools, release 12.4, V12.4.0','')
    with patch.object(rc.shutil,'which',return_value='/usr/local/cuda/bin/nvcc'),patch.object(rc.subprocess,'run',return_value=version):
        assert rc._cuda_toolchain_check() is not None


def test_actual_cuda_toolchain_preflight_and_same_ref_cpu_cuda_bundles(tmp_path):
    control = tmp_path / "control"; repo = control / "fixture"; repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-qb", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
    (repo / "main.c").write_text("int main(void) { return 0; }\n")
    (repo / "inert.cu").write_text("__device__ int unused_device_function(void) { return 7; }\n")
    (repo / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(inert C CXX)
if(EXISTS /etc/passwd OR EXISTS /home OR EXISTS /mnt)
  message(FATAL_ERROR "host private paths visible")
endif()
add_executable(llama-server main.c)
if(GGML_CUDA)
  enable_language(CUDA)
  target_sources(llama-server PRIVATE inert.cu)
endif()
''')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "inert CPU and CUDA"], cwd=repo, check=True)
    rc.project_add(control, "fixture", "https://fixture.example/inert.git", default_branch="main")
    plan = rc.bundle_plan(control, "fixture", recipe="cmake-cuda", cuda_arch="120a")
    if plan["status"] == "blocked":
        assert plan["capabilities"]["cuda_compile"] == "unvalidated"
        with pytest.raises(rc.RuntimeControlError):
            rc.bundle_build(control, "fixture", recipe="cmake-cuda", cuda_arch="120a")
        pytest.skip("CUDA sandbox capability unavailable; real CUDA acceptance remains unvalidated")
    assert plan["status"] == "planned", plan["toolchain_issue"]
    assert plan["capabilities"]["cuda_compile"] == "verified-in-sandbox"
    cpu = rc.bundle_build(control, "fixture")
    logs = []
    cuda = rc.bundle_build(control, "fixture", recipe="cmake-cuda", cuda_arch="120a", log=logs.append)
    assert cuda["id"] == plan["id"] and cpu["id"] != cuda["id"]
    assert cuda["cuda"] is True and cuda["recipe"] == "cmake-cuda"
    assert cuda["architecture"]["cuda"] == "120a"
    assert cuda["ref"] == cuda["branch"] == "main"
    assert cuda["toolchain"]["cuda"]["path"] == str(Path(shutil.which("nvcc")).resolve())
    assert "The CUDA compiler identification is NVIDIA" in "\n".join(logs)
    assert len(rc.bundle_list(control)) == 2
    assert rc.bundle_verify(control, cuda["id"])["status"] == "ok"
    assert not list(control.glob(".build-*"))


def test_private_path_nvcc_is_rejected_before_host_execution(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    compiler = tmp_path / "nvcc"
    compiler.write_text(f'#!/bin/sh\ntouch "{marker}"\necho "release 13.4"\n')
    compiler.chmod(0o755)
    monkeypatch.setattr(rc.shutil, "which", lambda name: str(compiler))
    assert rc._cuda_toolchain_check() is not None
    assert not marker.exists()
