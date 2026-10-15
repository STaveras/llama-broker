"""Deterministic, opt-in runtime project and bundle control plane.

This module deliberately does not touch broker profiles or running services.  It
supports only the reviewed ``llama.cpp`` CMake recipe and publishes immutable
bundles into a separate catalog.
"""
from __future__ import annotations

import hashlib, json, os, re, shutil, subprocess, tempfile, time, fcntl, stat, selectors, signal
from pathlib import Path
from urllib.parse import urlsplit

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
ALLOWED_SCHEMES = {"https", "ssh"}
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")

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
    path = path.absolute()
    if not path.is_relative_to(root.absolute()):
        raise RuntimeControlError(f"path escapes trusted root: {path}")
    current = root.absolute()
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

def _elf_dependencies(binary: Path) -> list[str]:
    try:
        out = subprocess.run(["readelf", "-d", str(binary)], text=True, capture_output=True,
                             check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeControlError(f"cannot inspect ELF dependencies: {exc}") from exc
    return sorted(set(re.findall(r"Shared library: \[(.+?)\]", out)))

def _dependency_inventory(binary: Path) -> dict:
    deps = _elf_dependencies(binary)
    resolved = {}
    missing = []
    for dep in deps:
        result = shutil.which("ldconfig")
        found = None
        if result:
            try:
                text = subprocess.run([result, "-p"], text=True, capture_output=True, check=True, timeout=10).stdout
                for line in text.splitlines():
                    match = re.search(r"\b" + re.escape(dep) + r"\s+\([^)]*\)\s+=>\s+(/[^ ]+)", line)
                    if match:
                        found = match.group(1); break
            except (OSError, subprocess.SubprocessError): pass
        if found: resolved[dep] = {"path": found, "sha256": _sha(Path(found))}
        else: missing.append(dep)
    if missing: raise RuntimeControlError("unsupported unresolved ELF dependencies: " + ", ".join(missing))
    return {"mode": "readelf+ldconfig", "needed": deps, "resolved": resolved}

def _run(argv, *, cwd=None, timeout=900, env=None, log=None):
    if not isinstance(argv, (list, tuple)) or any(not isinstance(x, str) for x in argv):
        raise RuntimeControlError("command must be an argv list")
    proc = subprocess.Popen(list(argv), cwd=str(cwd) if cwd else None, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, shell=False, start_new_session=True)
    lines=[]
    deadline = time.monotonic() + timeout
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise subprocess.TimeoutExpired(argv, timeout)
            for key, _ in sel.select(remaining):
                data = key.fileobj.readline()
                if data == "": sel.unregister(key.fileobj); continue
                line = data.rstrip("\n")
                lines.append(line)
                if log: log(line)
        code = proc.wait(timeout=max(0, deadline - time.monotonic()))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise RuntimeControlError("command timed out or cancelled")
    finally:
        sel.close()
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
    item["resolved_default_ref"] = branch or "unknown"
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

def bundle_plan(root: Path,name: str, *, branch=None, commit=None, build_command=None, allow_network=False) -> dict:
    item=_project(root,name); path=_assert_safe_path(Path(item["path"]), Path(root), directory=True)
    if branch and commit: raise RuntimeControlError("choose branch or commit, not both")
    if allow_network: raise RuntimeControlError("--allow-network is unsupported; builds are always offline")
    if commit and not FULL_COMMIT.fullmatch(commit): raise RuntimeControlError("--commit requires an exact 40-hex commit")
    ref=commit or branch or item.get("default_branch") or "origin/HEAD"
    if not ref: raise RuntimeControlError("no default branch known; pass --branch or --commit")
    if build_command and not (build_command == ["cmake"] or build_command[:2] == ["cmake","--build"]):
        raise RuntimeControlError("only the reviewed CMake recipe is supported")
    return {"project":name,"source_url":item["source_url"],"ref":ref,"selection":"commit" if commit else "branch","build_recipe":"cmake-cpu","network":False,"cuda":False,"status":"planned"}

def bundle_build(root: Path,name: str, *, branch=None, commit=None, allow_network=False, jobs=2, log=None) -> dict:
    plan=bundle_plan(root,name,branch=branch,commit=commit,allow_network=allow_network)
    item=_project(root,name); repo=_assert_safe_path(Path(item["path"]), Path(root), directory=True)
    resolved=_run(["git","rev-parse","--verify","--end-of-options",plan["ref"]+"^{commit}"],cwd=repo).strip()
    if commit and resolved != commit: raise RuntimeControlError("commit does not resolve exactly")
    _run(["git","checkout","--detach","--force","--no-recurse-submodules","--end-of-options",resolved],cwd=repo)
    build=repo/".broker-build"/resolved[:12]; build.mkdir(parents=True,exist_ok=True)
    bwrap = shutil.which("bwrap")
    if not bwrap: raise RuntimeControlError("bubblewrap (bwrap) is required for sandboxed builds")
    source = repo.resolve(); build = build.resolve()
    sandbox_env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                   "HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C",
                   "SOURCE_DATE_EPOCH": "0"}
    common = [bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL",
              "--tmpfs", "/", "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
              "--ro-bind", "/sbin", "/sbin", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
              "--ro-bind", "/etc/alternatives", "/etc/alternatives", "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
              "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/src", "--dir", "/tmp/build",
              "--ro-bind", str(source), "/tmp/src", "--bind", str(build), "/tmp/build",
              "--chdir", "/tmp/build", "--clearenv"]
    for key, value in sandbox_env.items(): common += ["--setenv", key, value]
    _run(common + ["cmake", "-S", "/tmp/src", "-B", "/tmp/build", "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"], timeout=1200, log=log)
    _run(common + ["cmake", "--build", "/tmp/build", "--parallel", str(max(1,min(int(jobs),4)))], timeout=1800, log=log)
    binaries=sorted(p for p in build.rglob("*") if p.is_file() and os.access(p,os.X_OK) and not p.is_symlink())
    if not binaries: raise RuntimeControlError("recipe produced no executable")
    catalog=root/"bundles"; catalog.mkdir(parents=True,exist_ok=True)
    bundle=catalog/(name+"-"+resolved[:12]); tmp=Path(tempfile.mkdtemp(prefix=".bundle-",dir=catalog))
    try:
        (tmp/"bin").mkdir(); src=next((p for p in binaries if p.name == "llama-server"), binaries[0])
        shutil.copy2(src,tmp/"bin"/src.name)
        links = []
        for lib in sorted(build.rglob("*.so*")):
            if lib.is_symlink():
                target = os.readlink(lib); (tmp/"lib").mkdir(exist_ok=True); (tmp/"lib"/lib.name).symlink_to(target)
                links.append({"path": "lib/"+lib.name, "target": target})
            elif lib.is_file():
                (tmp/"lib").mkdir(exist_ok=True); shutil.copy2(lib,tmp/"lib"/lib.name)
        dependency_inventory = _dependency_inventory(src)
        files={"bin/"+src.name:_sha(tmp/"bin"/src.name)}
        files.update({p.relative_to(tmp).as_posix(): _sha(p) for p in (tmp/"lib").glob("*") if p.is_file() and not p.is_symlink()} if (tmp/"lib").exists() else {})
        manifest={"schema_version":1,"id":bundle.name,"runtime_family":"llamacpp","source_url":item["source_url"],"resolved_commit":resolved,"branch":branch,"subject":_run(["git","show","-s","--format=%s","--end-of-options",resolved],cwd=repo),"recipe":"cmake-cpu","cuda":False,"network":False,"files":files,"links":links,"dependencies":dependency_inventory}
        (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        if bundle.exists(): raise RuntimeControlError("refusing to overwrite existing immutable bundle")
        os.replace(tmp,bundle)
        index = catalog / "index.json"
        anchors = {} if not index.exists() else json.loads(index.read_text()).get("bundles", {})
        anchors[bundle.name] = _sha(bundle / "manifest.json")
        index.write_text(json.dumps({"schema_version": 1, "bundles": anchors}, indent=2, sort_keys=True) + "\n")
    finally:
        shutil.rmtree(tmp,ignore_errors=True)
    return manifest

_MANIFEST_KEYS = {"schema_version","id","runtime_family","source_url","resolved_commit","branch","subject","recipe","cuda","network","files","links","dependencies"}

def _manifest_path(catalog: Path, ident: str) -> Path:
    if not SAFE_NAME.fullmatch(ident): raise RuntimeControlError("invalid bundle id")
    bundle = _assert_safe_path(catalog / ident, catalog, directory=True)
    path = _assert_safe_path(bundle / "manifest.json", bundle)
    if not path.is_file() or path.is_symlink(): raise RuntimeControlError("bundle manifest is missing or unsafe")
    return path

def _catalog_index(catalog: Path) -> dict:
    index = _assert_safe_path(catalog / "index.json", catalog)
    if not index.is_file() or index.is_symlink(): raise RuntimeControlError("bundle trust index is missing")
    try: value=json.loads(index.read_text())
    except Exception as exc: raise RuntimeControlError("invalid bundle trust index") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("bundles"), dict):
        raise RuntimeControlError("invalid bundle trust index")
    return value

def _strict_manifest(path: Path, ident: str) -> dict:
    try: value=json.loads(path.read_text())
    except Exception as exc: raise RuntimeControlError("invalid bundle manifest") from exc
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS or value.get("schema_version") != 1 or value.get("id") != ident:
        raise RuntimeControlError("invalid bundle manifest schema")
    return value

def bundle_list(root: Path) -> list[dict]:
    catalog=_assert_safe_path(Path(root)/"bundles", Path(root), directory=True); index=_catalog_index(catalog)
    result=[]
    for ident, digest in sorted(index["bundles"].items()):
        path=_manifest_path(catalog, ident)
        if not isinstance(digest,str) or _sha(path) != digest: raise RuntimeControlError(f"bundle trust anchor mismatch: {ident}")
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
    return {"id":name,"status":"ok" if not errors else "invalid","errors":errors,"manifest":m}
