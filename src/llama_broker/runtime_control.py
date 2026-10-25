"""Deterministic, opt-in runtime project and bundle control plane.

This module deliberately does not touch broker profiles or running services.  It
supports only the reviewed ``llama.cpp`` CMake recipe and publishes immutable
bundles into a separate catalog.
"""
from __future__ import annotations

import hashlib, json, os, re, shutil, subprocess, tempfile, time, fcntl, stat, selectors, signal, platform, tarfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
ALLOWED_SCHEMES = {"https", "ssh"}
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
CUDA_ARCHES = {"120a"}
SONAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]*$")


def _cmake_recipe(recipe: str, *, cuda_arch: str | None = None) -> tuple[list[str], bool]:
    if recipe not in {"cmake-cpu", "cmake-cuda"}:
        raise RuntimeControlError("recipe must be cmake-cpu or cmake-cuda")
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
              "-DLLAMA_OPENSSL=ON", "-DLLAMA_BUILD_BORINGSSL=OFF",
              "-DLLAMA_SUBPROCESS=OFF", "-DGGML_RPC=OFF",
              "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
              "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TOOLS=OFF",
              "-DLLAMA_BUILD_WEBUI=OFF"]
    if recipe == "cmake-cpu":
        return ["-DGGML_CUDA=OFF", *common], False
    if cuda_arch not in CUDA_ARCHES:
        raise RuntimeControlError("cmake-cuda requires --cuda-arch 120a")
    return ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}", *common], True


def _cuda_toolchain_check() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return "nvcc not found; install a CUDA toolkit with nvcc before building"
    try:
        selected = Path(nvcc).resolve()
        if not re.fullmatch(r"/(?:usr/local|opt)/cuda-[0-9.]+/bin/nvcc", str(selected)):
            return "unsupported CUDA toolkit path; select nvcc from a versioned system CUDA installation"
        version = subprocess.run([str(selected), "--version"], text=True, capture_output=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not inspect nvcc: {exc}"
    match = re.search(r"release\s+(\d+)\.(\d+)", version, re.I)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        return "nvcc version is incompatible; CUDA 12.8 or newer is required for arch 120a"
    return None

class RuntimeControlError(ValueError): pass

def _safe_relative(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeControlError(f"path escapes bundle root: {path}") from exc
    if any(part in ("", ".", "..") for part in rel.parts):
        raise RuntimeControlError(f"unsafe bundle path: {rel}")
    return rel.as_posix()

def _assert_safe_path(path: Path, root: Path, *, directory=False) -> Path:
    """Reject symlinked components and paths escaping the trusted root."""
    if ".." in path.parts or ".." in root.parts: raise RuntimeControlError("path traversal is not permitted")
    path = path.absolute()
    if not path.is_relative_to(root.absolute()):
        raise RuntimeControlError(f"path escapes trusted root: {path}")
    current = root.absolute()
    if current.is_symlink(): raise RuntimeControlError("symlink trusted root is not permitted")
    for part in path.relative_to(current).parts:
        current /= part
        if current.is_symlink():
            raise RuntimeControlError(f"symlink path component is not permitted: {current}")
    if directory and path.exists() and not path.is_dir():
        raise RuntimeControlError(f"not a directory: {path}")
    return path

def _copy_tree_preserve_links(source: Path, destination: Path) -> list[dict]:
    """Copy a tree without dereferencing links; reject links escaping source."""
    source = source.resolve(); destination.mkdir(parents=True, exist_ok=True)
    records = []
    for item in sorted(source.rglob("*")):
        rel = _safe_relative(item, source); target = destination / rel
        if item.is_symlink():
            link = os.readlink(item)
            resolved = (item.parent / link).resolve()
            if not resolved.is_relative_to(source):
                raise RuntimeControlError(f"symlink escapes source tree: {rel} -> {link}")
            target.parent.mkdir(parents=True, exist_ok=True); target.symlink_to(link)
            records.append({"path": rel, "type": "symlink", "target": link})
        elif item.is_dir(): target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(item, target)
            records.append({"path": rel, "type": "file", "sha256": _sha(item)})
        else: raise RuntimeControlError(f"unsupported bundle entry: {rel}")
    return records

# Only glibc and its loader remain host supplied, pinned to this host ABI.
_HOST_LIBS = {"libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
              "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1"}
_SYSTEM_LIB_DIRS = [Path("/usr/lib") / (platform.machine()+"-linux-gnu"),
                    Path("/lib") / (platform.machine()+"-linux-gnu"), Path("/lib64"), Path("/usr/lib64")]

def _elf_info(binary: Path) -> dict:
    try:
        out = subprocess.run(["/usr/bin/readelf", "-hW", "-dW", "-lW", str(binary)], text=True,
                             capture_output=True, check=True, timeout=30, env={"LC_ALL": "C"}).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeControlError(f"cannot inspect ELF dependencies: {exc}") from exc
    machines = {"x86_64": "Advanced Micro Devices X86-64", "aarch64": "AArch64"}
    machine = re.search(r"Machine:\s+([^\n]+)", out)
    if not machine or machine.group(1).strip() != machines.get(platform.machine()):
        raise RuntimeControlError("ELF machine does not match supported host architecture")
    if not re.search(r"Class:\s+ELF64\b", out) or not re.search(r"Data:.*little endian", out) or not re.search(r"Type:\s+(DYN|EXEC)\b", out):
        raise RuntimeControlError("unsupported ELF architecture or file type")
    needed = sorted(set(re.findall(r"Shared library: \[(.+?)\]", out)))
    paths = re.findall(r"\(RUNPATH\).*?\[(.*?)\]", out)
    rpaths = re.findall(r"\(RPATH\).*?\[(.*?)\]", out)
    for name in needed:
        if not SONAME.fullmatch(name): raise RuntimeControlError("unsafe ELF dependency name")
    interpreter = re.search(r"Requesting program interpreter: (.*?)\]", out)
    return {"needed": needed, "runpath": paths, "rpath": rpaths, "interpreter": interpreter.group(1) if interpreter else None}

def _host_library(name: str) -> Path:
    if name not in _HOST_LIBS: raise RuntimeControlError("unsupported host library: " + name)
    if name == "ld-linux-x86-64.so.2": return Path("/lib64") / name
    if name == "ld-linux-aarch64.so.1": return Path("/lib") / name
    for directory in _SYSTEM_LIB_DIRS:
        path = directory / name
        if path.is_file(): return path
    raise RuntimeControlError("unresolved host library: " + name)

def _dependency_inventory(bundle: Path) -> dict:
    objects = {}; host = {}; pending = [bundle / "bin/llama-server"]
    inherited = _elf_info(pending[0])["rpath"]
    pending += sorted(p for p in (bundle / "lib").glob("*") if not p.is_symlink())
    while pending:
        path = pending.pop(0); rel = _safe_relative(path, bundle)
        if rel in objects: continue
        info = _elf_info(path); objects[rel] = info
        for runpath in info["runpath"] + info["rpath"]:
            if any(x not in {"$ORIGIN", "$ORIGIN/../lib"} for x in runpath.split(":")):
                raise RuntimeControlError("unsupported ELF RPATH/RUNPATH: " + rel)
        for name in info["needed"]:
            packaged = bundle / "lib" / name
            if packaged.exists():
                target = packaged.resolve(strict=True)
                if not target.is_relative_to(bundle / "lib"):
                    raise RuntimeControlError("dependency symlink escapes bundle")
                direct = info["runpath"] or info["rpath"]
                search = {x for entry in direct for x in entry.split(":")}
                visible = "$ORIGIN/../lib" in search or rel.startswith("lib/") and "$ORIGIN" in search
                if not direct: visible = any("$ORIGIN/../lib" in entry.split(":") for entry in inherited)
                if not visible: raise RuntimeControlError("ELF RPATH search cannot reach packaged dependency: " + rel)
                pending.append(target)
            else:
                external = _host_library(name)
                host[name] = {"path": str(external), "sha256": _sha(external)}
        if info["interpreter"]:
            interp = Path(info["interpreter"])
            if interp.name not in _HOST_LIBS or interp != Path("/lib64/ld-linux-x86-64.so.2") and interp != Path("/lib/ld-linux-aarch64.so.1"):
                raise RuntimeControlError("unsupported ELF interpreter")
            host[interp.name] = {"path": str(interp), "sha256": _sha(interp)}
    inspected = set()
    while set(host) - inspected:
        name = next(iter(set(host) - inspected)); inspected.add(name)
        for needed in _elf_info(Path(host[name]["path"]))["needed"]:
            external = _host_library(needed)
            host.setdefault(needed, {"path": str(external), "sha256": _sha(external)})
    return {"mode": "verified-elf-closure-v1", "objects": objects, "host": host}

def _package_libraries(build: Path, bundle: Path, toolchain: dict) -> list[dict]:
    libdir = bundle / "lib"; libdir.mkdir(); links = []
    candidates = {}
    for lib in sorted(build.rglob("*.so*")):
        _assert_safe_path(lib.parent, build)
        if lib.name in candidates: raise RuntimeControlError("ambiguous library name: " + lib.name)
        candidates[lib.name] = lib
    search = list(_SYSTEM_LIB_DIRS)
    if "cuda" in toolchain:
        search.insert(0, Path(toolchain["cuda"]["path"]).parent.parent / "targets" / (platform.machine()+"-linux") / "lib")
    def copy_library(src):
        dest = libdir / src.name
        if dest.exists() or dest.is_symlink(): return
        if src.is_symlink():
            target = os.readlink(src)
            if not target or "/" in target or target in {".", ".."}:
                raise RuntimeControlError("unsafe library symlink: " + str(src))
            resolved = src.resolve(strict=True)
            if src in candidates.values() and not resolved.is_relative_to(build):
                raise RuntimeControlError("dependency symlink escapes build")
            # Copy the target first, so cycles and dangling links cannot be published.
            copy_library(src.parent / target)
            dest.symlink_to(target); links.append({"path": "lib/"+src.name, "target": target})
        elif src.is_file(): shutil.copy2(src, dest)
        else: raise RuntimeControlError("missing dependency: " + str(src))
    for src in candidates.values(): copy_library(src)
    inspected = set()
    while True:
        pending = [bundle / "bin/llama-server", *sorted(p for p in libdir.iterdir() if not p.is_symlink())]
        pending = [p for p in pending if p not in inspected]
        if not pending: break
        for obj in pending:
            inspected.add(obj)
            for name in _elf_info(obj)["needed"]:
                if name in _HOST_LIBS or (libdir / name).exists(): continue
                src = next((d / name for d in search if (d / name).is_file()), None)
                if src is None: raise RuntimeControlError("unsupported unresolved ELF dependency: " + name)
                copy_library(src)
    return sorted(links, key=lambda x: x["path"])

def _run(argv, *, cwd=None, timeout=900, env=None, log=None):
    if not isinstance(argv, (list, tuple)) or any(not isinstance(x, str) for x in argv):
        raise RuntimeControlError("command must be an argv list")
    proc = subprocess.Popen(list(argv), cwd=str(cwd) if cwd else None, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0, shell=False, start_new_session=True)
    lines=[]
    pending = bytearray()
    def record(data):
        line = data.decode("utf-8", errors="replace").rstrip("\r")
        lines.append(line)
        if log: log(line)
    deadline = time.monotonic() + timeout
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise subprocess.TimeoutExpired(argv, timeout)
            for key, _ in sel.select(remaining):
                # readline() can block past the deadline on a partial log line.
                data = os.read(key.fd, 8192)
                if not data:
                    if pending: record(pending); pending.clear()
                    sel.unregister(key.fileobj)
                    continue
                pending.extend(data)
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    record(line)
                    pending = bytearray(remainder)
        code = proc.wait(timeout=max(0, deadline - time.monotonic()))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise RuntimeControlError("command timed out or cancelled")
    finally:
        sel.close()
        proc.stdout.close()
    if code: raise RuntimeControlError(f"command failed ({code}): {' '.join(argv)}\n" + "\n".join(lines[-20:]))
    return "\n".join(lines)

def validate_source_url(url: str) -> str:
    p=urlsplit(url)
    if p.scheme not in ALLOWED_SCHEMES or not p.netloc or p.username or p.password:
        raise RuntimeControlError("source URL must be https:// or ssh:// without embedded credentials")
    if ".." in p.path.split("/"):
        raise RuntimeControlError("source URL contains path traversal")
    return url

def _state(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True); return root / "projects.json"

def load_projects(root: Path) -> list[dict]:
    p=_state(root)
    if not p.exists(): return []
    value=json.loads(p.read_text())
    return value if isinstance(value,list) else []

def save_projects(root: Path, items: list[dict]) -> None:
    p=_state(root); tmp=p.with_suffix(".tmp"); payload=json.dumps(items,indent=2,sort_keys=True)+"\n"
    with p.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tmp.open("w", encoding="utf-8") as out:
            out.write(payload); out.flush(); os.fsync(out.fileno())
        os.replace(tmp,p)
        fd=os.open(p.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)

def project_add(root: Path, name: str, url: str, *, default_branch: str|None=None) -> dict:
    if not SAFE_NAME.fullmatch(name): raise RuntimeControlError("invalid project name")
    url=validate_source_url(url)
    items=load_projects(root)
    if any(x["name"]==name for x in items): raise RuntimeControlError("project already exists")
    item={"name":name,"source_url":url,"default_branch":default_branch,"path":str((root/name).resolve())}
    items.append(item); save_projects(root,items); return item

def project_list(root: Path) -> list[dict]: return load_projects(root)

def _project(root,name):
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
        raise RuntimeControlError("invalid project name")
    root = _assert_safe_path(Path(root), Path(root).parent, directory=True)
    item=next((x for x in load_projects(root) if isinstance(x, dict) and x.get("name")==name),None)
    if not item: raise RuntimeControlError(f"unknown project: {name}")
    if not SAFE_NAME.fullmatch(item.get("name", "")):
        raise RuntimeControlError("stored project has invalid name")
    expected = _assert_safe_path(root / name, root)
    stored = item.get("path")
    if not isinstance(stored, str) or Path(stored).absolute() != expected.absolute():
        raise RuntimeControlError("stored project path does not match project root")
    _assert_safe_path(expected, root, directory=True)
    return item

def project_fetch(root: Path,name: str) -> dict:
    item=_project(root,name); path=_assert_safe_path(Path(item["path"]), Path(root), directory=True); root.mkdir(parents=True,exist_ok=True)
    if not (path/".git").exists(): _run(["git","clone","--no-checkout",item["source_url"],str(path)])
    else: _run(["git","fetch","--no-tags","--prune","origin"],cwd=path)
    remote=_run(["git","symbolic-ref","refs/remotes/origin/HEAD"],cwd=path).strip()
    branch=remote.removeprefix("refs/remotes/origin/") if remote else None
    item["default_branch"]=item.get("default_branch") or branch
    item["resolved_default_ref"] = "refs/remotes/origin/" + item["default_branch"] if item["default_branch"] else None
    items=load_projects(root); items=[item if x["name"]==name else x for x in items]
    save_projects(root,items); return item

def project_commits(root: Path,name: str, ref: str|None=None) -> list[dict]:
    item=_project(root,name); path=_assert_safe_path(Path(item["path"]), Path(root), directory=True); ref=ref or item.get("default_branch")
    if not ref: raise RuntimeControlError("default branch is unknown; pass --ref explicitly")
    out=_run(["git","log","--format=%H%x09%s","-20",ref],cwd=path)
    return [{"hash":h,"subject":s} for h,s in (line.split("\t",1) for line in out.splitlines() if "\t" in line)]

def _sha(path):
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _toolchain(cuda: bool) -> dict:
    selected = {"c": "/usr/bin/cc", "cxx": "/usr/bin/c++", "cmake": "/usr/bin/cmake",
                "linker": "/usr/bin/ld", "assembler": "/usr/bin/as", "make": "/usr/bin/make"}
    if cuda:
        issue = _cuda_toolchain_check()
        if issue: raise RuntimeControlError(issue)
        selected["cuda"] = shutil.which("nvcc")
    result = {}
    for kind, executable in selected.items():
        path = Path(executable).resolve(strict=True)
        result[kind] = {"path": str(path), "version": _run([str(path), "--version"], timeout=10), "sha256": _sha(path)}
    return result

def _identity(name, source_url, resolved, recipe, flags, architecture, toolchain):
    payload = {"project": name, "source_url": source_url, "resolved_commit": resolved, "recipe": recipe,
               "configure_flags": flags, "architecture": architecture, "toolchain": toolchain}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{name[:48]}-{resolved[:12]}-{digest[:32]}"

def _sandbox(source: Path, build: Path, toolchain: dict) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap: raise RuntimeControlError("bubblewrap (bwrap) is required for sandboxed builds")
    common = [bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL", "--tmpfs", "/"]
    # Mount distribution tools/libraries read only; exclude /usr/local, homes,
    # host devices, /mnt and /etc except the loader cache and compiler alternatives.
    for directory in ("/usr/bin", "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/include", "/usr/share"):
        if Path(directory).exists(): common += ["--ro-bind", directory, directory]
    for target, link in (("usr/bin", "/bin"), ("usr/lib", "/lib"), ("usr/lib64", "/lib64")):
        common += ["--symlink", target, link]
    for file in ("/etc/alternatives", "/etc/ld.so.cache"):
        if Path(file).exists(): common += ["--ro-bind", file, file]
    if "cuda" in toolchain:
        toolkit = Path(toolchain["cuda"]["path"]).parent.parent
        if not re.fullmatch(r"/(?:usr/local|opt)/cuda-[0-9.]+", str(toolkit)):
            raise RuntimeControlError("unsupported CUDA toolkit path; expected a versioned system CUDA installation")
        common += ["--ro-bind", str(toolkit), str(toolkit)]
    common += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", str(source), "/tmp/src", "--bind", str(build), "/tmp/build",
               "--chdir", "/tmp/build", "--clearenv"]
    for key, value in {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}.items():
        common += ["--setenv", key, value]
    return common

def _cuda_sandbox_preflight(toolchain, flags):
    with tempfile.TemporaryDirectory(prefix="runtime-cuda-preflight-") as directory:
        source = Path(directory) / "source"; source.mkdir()
        build = Path(directory) / "build"; build.mkdir()
        (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\nproject(probe CXX CUDA)\nadd_executable(probe main.cu)\n")
        (source / "main.cu").write_text("__device__ int inert_device_function() { return 0; }\nint main() { return 0; }\n")
        common = _sandbox(source, build, toolchain)
        _run(common + [toolchain["cmake"]["path"], "-S", "/tmp/src", "-B", "/tmp/build", *flags], timeout=120)
        _run(common + [toolchain["cmake"]["path"], "--build", "/tmp/build", "--target", "probe"], timeout=120)
        _elf_info(build / "probe")

def bundle_plan(root: Path,name: str, *, branch=None, commit=None, build_command=None, allow_network=False, recipe="cmake-cpu", cuda_arch=None) -> dict:
    item=_project(root,name); path=_assert_safe_path(Path(item["path"]), Path(root), directory=True)
    if branch and commit: raise RuntimeControlError("choose branch or commit, not both")
    if allow_network: raise RuntimeControlError("--allow-network is unsupported; builds are always offline")
    if commit and not FULL_COMMIT.fullmatch(commit): raise RuntimeControlError("--commit requires an exact 40-hex commit")
    ref=commit or branch or item.get("resolved_default_ref") or item.get("default_branch") or "origin/HEAD"
    if not ref: raise RuntimeControlError("no default branch known; pass --branch or --commit")
    flags, cuda = _cmake_recipe(recipe, cuda_arch=cuda_arch)
    resolved = _run(["git", "rev-parse", "--verify", "--end-of-options", ref+"^{commit}"], cwd=path).strip()
    if not FULL_COMMIT.fullmatch(resolved): raise RuntimeControlError("source did not resolve to a full commit")
    issue = None
    try: toolchain = _toolchain(cuda)
    except (RuntimeControlError, OSError) as exc: issue = str(exc); toolchain = {}
    architecture = {"host": platform.machine(), "cuda": cuda_arch if cuda else None}
    flags += ["-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib;$ORIGIN", "-DGGML_NATIVE=OFF", "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags"]
    for key, flag in (("c", "CMAKE_C_COMPILER"), ("cxx", "CMAKE_CXX_COMPILER"), ("cuda", "CMAKE_CUDA_COMPILER")):
        if key in toolchain: flags.append(f"-D{flag}={toolchain[key]['path']}")
    if cuda and "cuda" in toolchain:
        flags.append(f"-DCUDAToolkit_ROOT={Path(toolchain['cuda']['path']).parent.parent}")
        flags.append(f"-DCMAKE_CUDA_HOST_COMPILER={toolchain['cxx']['path']}")
    capabilities = {"cuda_compile": "not-requested"}
    if cuda:
        capabilities["cuda_compile"] = "unvalidated"
        if not issue:
            try:
                _cuda_sandbox_preflight(toolchain, flags)
                capabilities["cuda_compile"] = "verified-in-sandbox"
            except (OSError, RuntimeControlError) as exc: issue = "CUDA sandbox capability preflight failed: " + str(exc)
    ident = _identity(name, item["source_url"], resolved, recipe, flags, architecture, toolchain)
    return {
        "project": name, "source_url": item["source_url"], "ref": ref,
        "resolved_commit": resolved, "id": ident, "architecture": architecture,
        "toolchain": toolchain, "selection": "commit" if commit else "branch",
        "build_recipe": recipe, "configure_flags": flags, "network": False,
        "cuda": cuda, "capabilities": capabilities, "toolchain_issue": issue,
        "status": "blocked" if issue else "planned",
    }

def bundle_build(root: Path,name: str, *, branch=None, commit=None, allow_network=False, jobs=2, log=None, recipe="cmake-cpu", cuda_arch=None) -> dict:
    plan=bundle_plan(root,name,branch=branch,commit=commit,allow_network=allow_network,recipe=recipe,cuda_arch=cuda_arch)
    if plan.get("toolchain_issue"): raise RuntimeControlError(plan["toolchain_issue"])
    with tempfile.TemporaryDirectory(prefix=f".build-{plan['id']}-", dir=root) as directory:
        try:
            return _build_plan(root, name, plan, Path(directory), jobs=jobs, log=log)
        except (OSError, tarfile.TarError, RuntimeError) as exc:
            raise RuntimeControlError(f"bundle build failed: {exc}") from exc

def _check_configured_recipe(build, plan):
    cache = _assert_safe_path(build / "CMakeCache.txt", build)
    if not cache.is_file(): raise RuntimeControlError("recipe configuration cache is missing")
    values = dict(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*):[^=\n]+=([^\n]*)$", cache.read_text(), re.M))
    for flag in plan["configure_flags"]:
        key, expected = flag[2:].split("=", 1)
        actual = values.get(key)
        if actual != expected:
            raise RuntimeControlError("source overrode reviewed recipe configuration: " + key)
    if plan["cuda"]:
        compilers = list((build / "CMakeFiles").glob("*/CMakeCUDACompiler.cmake"))
        if len(compilers) != 1: raise RuntimeControlError("CUDA recipe did not configure the selected CUDA compiler")
        _assert_safe_path(compilers[0], build)

def _build_plan(root, name, plan, workspace, *, jobs, log):
    item=_project(root,name); repo=_assert_safe_path(Path(item["path"]), Path(root), directory=True)
    resolved=plan["resolved_commit"]
    source = workspace / "source"; source.mkdir()
    archive = workspace / "source.tar"
    _run(["git", "archive", "--format=tar", "--output="+str(archive), resolved], cwd=repo)
    with tarfile.open(archive) as tree: tree.extractall(source, filter="data")
    build = workspace / "build"; build.mkdir()
    bwrap = shutil.which("bwrap")
    if not bwrap: raise RuntimeControlError("bubblewrap (bwrap) is required for sandboxed builds")
    common = _sandbox(source, build, plan["toolchain"])
    _run(common + ["cmake", "-S", "/tmp/src", "-B", "/tmp/build", *plan["configure_flags"]], timeout=1200, log=log)
    _check_configured_recipe(build, plan)
    _run(common + ["cmake", "--build", "/tmp/build", "--target", "llama-server", "--parallel", str(max(1,min(int(jobs),4)))], timeout=1800, log=log)
    # Preserve reviewed fixture targets without ever falling back to an arbitrary executable.
    targets = _run(common + ["cmake", "--build", "/tmp/build", "--target", "help"], timeout=120, log=log)
    if "network-proof" in targets:
        _run(common + ["cmake", "--build", "/tmp/build", "--target", "network-proof"], timeout=1800, log=log)
    _check_configured_recipe(build, plan)
    binaries=list(build.rglob("llama-server"))
    if len(binaries) != 1: raise RuntimeControlError("recipe must produce exactly one llama-server")
    src = _assert_safe_path(binaries[0], build)
    if not src.is_file() or not os.access(src, os.X_OK):
        raise RuntimeControlError("recipe llama-server is missing or not executable")
    catalog=root/"bundles"; catalog.mkdir(parents=True,exist_ok=True)
    bundle=catalog/plan["id"]; tmp=Path(tempfile.mkdtemp(prefix=".bundle-",dir=catalog))
    try:
        (tmp/"bin").mkdir()
        shutil.copy2(src,tmp/"bin"/src.name)
        links = _package_libraries(build, tmp, plan["toolchain"])
        dependency_inventory = _dependency_inventory(tmp)
        files={"bin/"+src.name:_sha(tmp/"bin"/src.name)}
        files.update({p.relative_to(tmp).as_posix(): _sha(p) for p in (tmp/"lib").glob("*") if p.is_file() and not p.is_symlink()} if (tmp/"lib").exists() else {})
        manifest = {
            "schema_version": 2, "id": bundle.name, "project": name,
            "runtime_family": "llamacpp", "source_url": item["source_url"],
            "resolved_commit": resolved, "ref": plan["ref"],
            "branch": plan["ref"] if plan["selection"] == "branch" else None,
            "subject": _run(["git", "show", "-s", "--format=%s", "--end-of-options", resolved], cwd=repo),
            "recipe": plan["build_recipe"], "cuda": plan["cuda"],
            "architecture": plan["architecture"], "configure_flags": plan["configure_flags"],
            "toolchain": plan["toolchain"], "network": False,
            "files": files, "links": links, "dependencies": dependency_inventory,
        }
        (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        _strict_manifest(tmp / "manifest.json", bundle.name)
        _publish_bundle(catalog, tmp, bundle)
    finally:
        shutil.rmtree(tmp,ignore_errors=True)
    return manifest

def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _atomic_json(path: Path, value):
    fd, temporary = tempfile.mkstemp(prefix=".index-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True); out.write("\n")
            out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)

@contextmanager
def _catalog_lock(catalog):
    _assert_safe_path(catalog, catalog.parent, directory=True)
    catalog.mkdir(parents=True, exist_ok=True)
    lockpath = _assert_safe_path(catalog / ".lock", catalog)
    try: fd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError as exc: raise RuntimeControlError(f"cannot open catalog lock: {exc}") from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeControlError("catalog lock must be a regular file")
    with os.fdopen(fd, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _recover_publication(catalog)
        yield

def _recover_publication(catalog):
    journal = _assert_safe_path(catalog / ".pending.json", catalog)
    if not journal.exists(): return
    if not journal.is_file(): raise RuntimeControlError("publication journal must be a regular file")
    try:
        record = json.loads(journal.read_text())
        if not isinstance(record, dict) or set(record) != {"id", "digest", "staged"}:
            raise RuntimeControlError("invalid publication journal")
        ident = record["id"]; staged = record["staged"]
        if not isinstance(ident, str) or len(ident) > 128 or not SAFE_NAME.fullmatch(ident) or not _digest(record["digest"]) or not isinstance(staged, str) or not re.fullmatch(r"\.bundle-[a-zA-Z0-9_-]+", staged):
            raise RuntimeControlError("unsafe publication journal")
        index = catalog / "index.json"
        anchors = _catalog_index(catalog)["bundles"] if index.exists() else {}
        bundle = _assert_safe_path(catalog / ident, catalog, directory=True)
        if anchors.get(ident) == record["digest"]:
            if _sha(_manifest_path(catalog, ident)) != record["digest"]:
                raise RuntimeControlError("committed publication has an invalid manifest")
        elif ident in anchors: raise RuntimeControlError("publication journal conflicts with trust index")
        elif bundle.exists(): shutil.rmtree(bundle)
        temporary = _assert_safe_path(catalog / staged, catalog, directory=True)
        if temporary.exists(): shutil.rmtree(temporary)
        journal.unlink(); _fsync_dir(catalog)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeControlError(f"cannot recover bundle publication: {exc}") from exc

def _publish_bundle(catalog, staged, bundle):
    with _catalog_lock(catalog):
        _assert_safe_path(bundle, catalog)
        if bundle.exists(): raise RuntimeControlError("refusing to overwrite existing immutable bundle")
        index = _assert_safe_path(catalog / "index.json", catalog)
        had_index = index.exists()
        prior = _catalog_index(catalog) if had_index else {"schema_version": 1, "bundles": {}}
        updated = {"schema_version": 1, "bundles": dict(prior["bundles"])}
        updated["bundles"][bundle.name] = _sha(staged / "manifest.json")
        _atomic_json(catalog / ".pending.json", {"id": bundle.name, "digest": updated["bundles"][bundle.name], "staged": staged.name})
        published = False
        try:
            for path in staged.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as file: os.fsync(file.fileno())
            for path in sorted((p for p in staged.rglob("*") if p.is_dir()), reverse=True): _fsync_dir(path)
            _fsync_dir(staged)
            os.replace(staged, bundle); published = True
            _fsync_dir(catalog)
            _atomic_json(index, updated)
            (catalog / ".pending.json").unlink(); _fsync_dir(catalog)
        except OSError as exc:
            committed = index.exists() and _catalog_index(catalog)["bundles"].get(bundle.name) == updated["bundles"][bundle.name]
            if committed:
                try:
                    if had_index: _atomic_json(index, prior)
                    else: index.unlink(); _fsync_dir(catalog)
                except OSError as rollback:
                    restored = _catalog_index(catalog) == prior if index.exists() else not had_index
                    if not restored:
                        raise RuntimeControlError(f"publication committed; rollback failed: {rollback}") from exc
            if published and bundle.exists(): shutil.rmtree(bundle)
            _recover_publication(catalog)
            raise RuntimeControlError(f"bundle publication failed: {exc}") from exc

_MANIFEST_KEYS = {"project","schema_version","id","runtime_family","source_url","resolved_commit","branch","subject","recipe","cuda","network","files","links","dependencies","ref","architecture","configure_flags","toolchain"}

def _manifest_path(catalog: Path, ident: str) -> Path:
    if not isinstance(ident, str) or len(ident) > 128 or not SAFE_NAME.fullmatch(ident): raise RuntimeControlError("invalid bundle id")
    bundle = _assert_safe_path(catalog / ident, catalog, directory=True)
    path = _assert_safe_path(bundle / "manifest.json", bundle)
    if not path.is_file() or path.is_symlink(): raise RuntimeControlError("bundle manifest is missing or unsafe")
    return path

def _catalog_index(catalog: Path) -> dict:
    index = _assert_safe_path(catalog / "index.json", catalog)
    if not index.is_file() or index.is_symlink(): raise RuntimeControlError("bundle trust index is missing")
    try: value=json.loads(index.read_text())
    except Exception as exc: raise RuntimeControlError("invalid bundle trust index") from exc
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or set(value) != {"schema_version", "bundles"} or not isinstance(value.get("bundles"), dict):
        raise RuntimeControlError("invalid bundle trust index")
    for ident, digest in value["bundles"].items():
        if not isinstance(ident, str) or len(ident) > 128 or not SAFE_NAME.fullmatch(ident) or not _digest(digest):
            raise RuntimeControlError("invalid bundle trust index entry")
    return value

def _relative_text(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or any(x in {"", ".", ".."} for x in value.split("/")) or value.startswith("/"):
        raise RuntimeControlError("unsafe bundle relative path")
    return value

def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

def _strict_manifest(path: Path, ident: str) -> dict:
    try: return _read_manifest(path, ident)
    except RuntimeControlError: raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeControlError(f"invalid bundle manifest: {exc}") from exc

def _read_manifest(path: Path, ident: str) -> dict:
    try: value=json.loads(path.read_text())
    except (OSError, ValueError) as exc: raise RuntimeControlError("invalid bundle manifest") from exc
    def require(ok, message="invalid bundle manifest schema"):
        if not ok: raise RuntimeControlError(message)
    require(isinstance(value, dict))
    if value.get("schema_version") == 1:
        raise RuntimeControlError("unsafe legacy bundle schema 1; rebuild required (no in-place migration)")
    require(set(value) == _MANIFEST_KEYS and type(value.get("schema_version")) is int and value["schema_version"] == 2 and value["id"] == ident)
    require(isinstance(value["project"], str) and SAFE_NAME.fullmatch(value["project"]))
    require(value["runtime_family"] == "llamacpp" and value["recipe"] in ("cmake-cpu", "cmake-cuda"))
    require(type(value["cuda"]) is bool and value["cuda"] == (value["recipe"] == "cmake-cuda") and value["network"] is False)
    require(isinstance(value["source_url"], str)); validate_source_url(value["source_url"])
    require(isinstance(value["resolved_commit"], str) and FULL_COMMIT.fullmatch(value["resolved_commit"]))
    require(isinstance(value["ref"], str) and bool(value["ref"]) and isinstance(value["subject"], str))
    require(value["branch"] is None or isinstance(value["branch"], str) and value["branch"] == value["ref"])
    arch = value["architecture"]
    require(isinstance(arch, dict) and set(arch) == {"host", "cuda"})
    require(isinstance(arch["host"], str) and SAFE_NAME.fullmatch(arch["host"]))
    require(arch["cuda"] in CUDA_ARCHES if value["cuda"] else arch["cuda"] is None)
    tools = value["toolchain"]
    require(isinstance(tools, dict) and set(tools) == ({"c", "cxx", "cmake", "linker", "assembler", "make", "cuda"} if value["cuda"] else {"c", "cxx", "cmake", "linker", "assembler", "make"}))
    for tool in tools.values():
        require(isinstance(tool, dict) and set(tool) == {"path", "version", "sha256"})
        require(isinstance(tool["path"], str) and tool["path"].startswith(("/usr/", "/opt/cuda")) and ".." not in Path(tool["path"]).parts)
        require(isinstance(tool["version"], str) and bool(tool["version"]) and _digest(tool["sha256"]))
    flags = value["configure_flags"]
    require(isinstance(flags, list) and all(isinstance(x, str) for x in flags))
    required, _ = _cmake_recipe(value["recipe"], cuda_arch=arch["cuda"])
    required += ["-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib;$ORIGIN", "-DGGML_NATIVE=OFF", "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags"]
    for key, flag in (("c", "CMAKE_C_COMPILER"), ("cxx", "CMAKE_CXX_COMPILER"), ("cuda", "CMAKE_CUDA_COMPILER")):
        if key in tools: required.append(f"-D{flag}={tools[key]['path']}")
    if value["cuda"]:
        required.append(f"-DCUDAToolkit_ROOT={Path(tools['cuda']['path']).parent.parent}")
        required.append(f"-DCMAKE_CUDA_HOST_COMPILER={tools['cxx']['path']}")
    require(flags == required, "manifest does not describe a supported recipe")
    require(ident == _identity(value["project"], value["source_url"], value["resolved_commit"], value["recipe"], flags, arch, tools), "bundle identity does not match configuration")
    files = value["files"]; links = value["links"]
    require(isinstance(files, dict) and "bin/llama-server" in files and isinstance(links, list))
    for rel, digest in files.items():
        _relative_text(rel)
        require(rel == "bin/llama-server" or rel.startswith("lib/") and rel.count("/") == 1)
        require(_digest(digest))
    linkmap = {}
    for link in links:
        require(isinstance(link, dict) and set(link) == {"path", "target"})
        rel = _relative_text(link["path"]); target = _relative_text(link["target"])
        require(rel.startswith("lib/") and rel.count("/") == 1 and "/" not in target)
        require(rel not in files and rel not in linkmap)
        linkmap[rel] = "lib/" + target
    for rel in linkmap:
        seen = set(); target = rel
        while target in linkmap:
            require(target not in seen, "cyclic dependency symlink")
            seen.add(target); target = linkmap[target]
        require(target in files, "dangling dependency symlink")
    deps = value["dependencies"]
    require(isinstance(deps, dict) and set(deps) == {"mode", "objects", "host"} and deps["mode"] == "verified-elf-closure-v1")
    require(isinstance(deps["objects"], dict) and set(deps["objects"]) == set(files) and isinstance(deps["host"], dict))
    for name, record in deps["host"].items():
        require(name in _HOST_LIBS and isinstance(record, dict) and set(record) == {"path", "sha256"})
        allowed = {str(d / name) for d in _SYSTEM_LIB_DIRS} | {"/lib64/ld-linux-x86-64.so.2", "/lib/ld-linux-aarch64.so.1"}
        require(isinstance(record["path"], str) and record["path"] in allowed and Path(record["path"]).name == name and _digest(record["sha256"]))
    for rel, info in deps["objects"].items():
        require(isinstance(info, dict) and set(info) == {"needed", "runpath", "rpath", "interpreter"})
        require(isinstance(info["needed"], list) and all(isinstance(x, str) and SONAME.fullmatch(x) for x in info["needed"]))
        require(isinstance(info["runpath"], list) and all(isinstance(x, str) and all(y in {"$ORIGIN", "$ORIGIN/../lib"} for y in x.split(":")) for x in info["runpath"]))
        require(isinstance(info["rpath"], list) and all(isinstance(x, str) and all(y in {"$ORIGIN", "$ORIGIN/../lib"} for y in x.split(":")) for x in info["rpath"]))
        require(info["interpreter"] is None or info["interpreter"] in {"/lib64/ld-linux-x86-64.so.2", "/lib/ld-linux-aarch64.so.1"})
        for name in info["needed"]:
            require("lib/"+name in files or "lib/"+name in linkmap or name in deps["host"], "incomplete dependency closure")
    # Validate all declared paths before a caller can hash or inspect any artifact.
    for rel in files:
        artifact = _assert_safe_path(path.parent / rel, path.parent)
        if artifact.exists(): require(stat.S_ISREG(artifact.lstat().st_mode), "unsafe bundle file type")
    for rel, target in linkmap.items():
        artifact = path.parent / rel
        _assert_safe_path(artifact.parent, path.parent)
        if artifact.is_symlink():
            require(os.readlink(artifact) == Path(target).name, "dependency symlink target mismatch")
    return value

def bundle_list(root: Path) -> list[dict]:
    catalog=_assert_safe_path(Path(root)/"bundles", Path(root), directory=True)
    result=[]
    with _catalog_lock(catalog):
        index = _catalog_index(catalog)
        for ident, digest in sorted(index["bundles"].items()):
            path=_manifest_path(catalog, ident)
            if _sha(path) != digest: raise RuntimeControlError(f"bundle trust anchor mismatch: {ident}")
            result.append(_strict_manifest(path, ident))
    return result

def bundle_verify(root: Path,name: str) -> dict:
    m=next((x for x in bundle_list(root) if x["id"]==name),None)
    if not m: raise RuntimeControlError("unknown bundle")
    p=_assert_safe_path(Path(root)/"bundles"/name, Path(root)/"bundles", directory=True); errors=[]
    expected=set(m.get("files",{})) | {x["path"] for x in m.get("links",[]) if isinstance(x,dict) and "path" in x} | {"manifest.json"}
    actual=set()
    for entry in p.rglob("*"):
        rel=_safe_relative(entry,p)
        if entry.is_dir() and entry.is_symlink(): errors.append("symlink-directory:"+rel)
        elif entry.is_file() or entry.is_symlink(): actual.add(rel)
    errors += ["unexpected:"+x for x in sorted(actual-expected)]
    for rel,digest in m["files"].items():
        if not (p/rel).is_file() or (p/rel).is_symlink() or _sha(p/rel)!=digest: errors.append(rel)
    for link in m.get("links", []):
        path = p / link["path"]
        _assert_safe_path(path.parent, p)
        if not path.is_relative_to(p) or not path.is_symlink() or os.readlink(path) != link["target"]: errors.append(link["path"])
        if not link["target"] or os.path.isabs(link["target"]) or ".." in Path(link["target"]).parts or not (path.parent / link["target"]).resolve().is_relative_to(p): errors.append(link["path"])
    if not os.access(p / "bin/llama-server", os.X_OK): errors.append("executable")
    if not errors:
        try:
            if _dependency_inventory(p) != m["dependencies"]: errors.append("dependencies")
        except RuntimeControlError: errors.append("dependencies")
    return {"id":name,"status":"ok" if not errors else "invalid","errors":errors,"manifest":m}
