"""Deterministic, opt-in runtime project and bundle control plane.

This module deliberately does not touch broker profiles or running services.  It
supports only the reviewed ``llama.cpp`` CMake recipe and publishes immutable
bundles into a separate catalog.
"""
from __future__ import annotations

import hashlib, json, os, re, shutil, subprocess, tempfile, time
from pathlib import Path
from urllib.parse import urlsplit

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
ALLOWED_SCHEMES = {"https", "ssh"}

class RuntimeControlError(ValueError): pass

def _safe_relative(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeControlError(f"path escapes bundle root: {path}") from exc
    if any(part in ("", ".", "..") for part in rel.parts):
        raise RuntimeControlError(f"unsafe bundle path: {rel}")
    return rel.as_posix()

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
                            text=True, bufsize=1, shell=False)
    lines=[]
    try:
        for line in iter(proc.stdout.readline, ""):
            lines.append(line.rstrip("\n"))
            if log: log(line.rstrip("\n"))
        code=proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        proc.kill(); proc.wait(); raise RuntimeControlError("command timed out or cancelled")
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
    p=_state(root); tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items,indent=2,sort_keys=True)+"\n"); os.replace(tmp,p)

def project_add(root: Path, name: str, url: str, *, default_branch: str|None=None) -> dict:
    if not SAFE_NAME.fullmatch(name): raise RuntimeControlError("invalid project name")
    url=validate_source_url(url)
    items=load_projects(root)
    if any(x["name"]==name for x in items): raise RuntimeControlError("project already exists")
    item={"name":name,"source_url":url,"default_branch":default_branch,"path":str((root/name).resolve())}
    items.append(item); save_projects(root,items); return item

def project_list(root: Path) -> list[dict]: return load_projects(root)

def _project(root,name):
    item=next((x for x in load_projects(root) if x["name"]==name),None)
    if not item: raise RuntimeControlError(f"unknown project: {name}")
    return item

def project_fetch(root: Path,name: str) -> dict:
    item=_project(root,name); path=Path(item["path"]); root.mkdir(parents=True,exist_ok=True)
    if not (path/".git").exists(): _run(["git","clone","--no-checkout",item["source_url"],str(path)])
    else: _run(["git","fetch","--no-tags","--prune","origin"],cwd=path)
    remote=_run(["git","symbolic-ref","refs/remotes/origin/HEAD"],cwd=path).strip()
    branch=remote.removeprefix("refs/remotes/origin/") if remote else None
    item["default_branch"]=item.get("default_branch") or branch
    item["resolved_default_ref"] = branch or "unknown"
    save_projects(root,load_projects(root)); return item

def project_commits(root: Path,name: str, ref: str|None=None) -> list[dict]:
    item=_project(root,name); path=Path(item["path"]); ref=ref or item.get("default_branch")
    if not ref: raise RuntimeControlError("default branch is unknown; pass --ref explicitly")
    out=_run(["git","log","--format=%H%x09%s","-20",ref],cwd=path)
    return [{"hash":h,"subject":s} for h,s in (line.split("\t",1) for line in out.splitlines() if "\t" in line)]

def _sha(path):
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def bundle_plan(root: Path,name: str, *, branch=None, commit=None, build_command=None, allow_network=False) -> dict:
    item=_project(root,name); path=Path(item["path"])
    if branch and commit: raise RuntimeControlError("choose branch or commit, not both")
    ref=commit or branch or item.get("default_branch")
    if not ref: raise RuntimeControlError("no default branch known; pass --branch or --commit")
    if build_command and not (build_command == ["cmake"] or build_command[:2] == ["cmake","--build"]):
        raise RuntimeControlError("only the reviewed CMake recipe is supported")
    return {"project":name,"source_url":item["source_url"],"ref":ref,"selection":"commit" if commit else "branch","build_recipe":"cmake-cpu","network":bool(allow_network),"cuda":False,"status":"planned"}

def bundle_build(root: Path,name: str, *, branch=None, commit=None, allow_network=False, jobs=2, log=None) -> dict:
    plan=bundle_plan(root,name,branch=branch,commit=commit,allow_network=allow_network)
    item=_project(root,name); repo=Path(item["path"])
    resolved=_run(["git","rev-parse",plan["ref"]],cwd=repo).strip()
    if commit and resolved != commit and not resolved.startswith(commit): raise RuntimeControlError("commit does not resolve exactly")
    build=repo/".broker-build"/resolved[:12]; build.mkdir(parents=True,exist_ok=True)
    _run(["cmake","-S",str(repo),"-B",str(build),"-DGGML_CUDA=OFF","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"],cwd=repo,timeout=1200,log=log)
    _run(["cmake","--build",str(build),"--parallel",str(max(1,min(int(jobs),4)))],cwd=repo,timeout=1800,log=log)
    binaries=[p for p in build.rglob("*") if p.is_file() and os.access(p,os.X_OK)]
    if not binaries: raise RuntimeControlError("recipe produced no executable")
    catalog=root/"bundles"; catalog.mkdir(parents=True,exist_ok=True)
    bundle=catalog/(name+"-"+resolved[:12]); tmp=Path(tempfile.mkdtemp(prefix=".bundle-",dir=catalog))
    try:
        (tmp/"bin").mkdir(); src=binaries[0]; shutil.copy2(src,tmp/"bin"/src.name)
        dependency_inventory = _dependency_inventory(src)
        manifest={"schema_version":1,"id":bundle.name,"runtime_family":"llamacpp","source_url":item["source_url"],"resolved_commit":resolved,"branch":branch,"subject":_run(["git","show","-s","--format=%s",resolved],cwd=repo),"recipe":"cmake-cpu","cuda":False,"network":bool(allow_network),"files":{"bin/"+src.name:_sha(tmp/"bin"/src.name)},"links":[],"dependencies":dependency_inventory}
        (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        if bundle.exists(): raise RuntimeControlError("refusing to overwrite existing immutable bundle")
        os.replace(tmp,bundle)
    finally:
        shutil.rmtree(tmp,ignore_errors=True)
    return manifest

def bundle_list(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root/"bundles").glob("*/manifest.json"))]

def bundle_verify(root: Path,name: str) -> dict:
    m=next((x for x in bundle_list(root) if x["id"]==name),None)
    if not m: raise RuntimeControlError("unknown bundle")
    p=root/"bundles"/name; errors=[]
    for rel,digest in m["files"].items():
        if not (p/rel).is_file() or (p/rel).is_symlink() or _sha(p/rel)!=digest: errors.append(rel)
    for link in m.get("links", []):
        path = p / link["path"]
        if not path.is_symlink() or os.readlink(path) != link["target"]: errors.append(link["path"])
    return {"id":name,"status":"ok" if not errors else "invalid","errors":errors,"manifest":m}
