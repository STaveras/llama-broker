import subprocess
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
