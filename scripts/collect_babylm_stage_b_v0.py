"""Read-only SSH inventory and append-preserving incremental Stage B evidence copy.

The existing connection file is the only connection source. No credentials,
complete process command lines, or environments are emitted or archived.
Changed remote files fail their transfer hash check and remain as incoming
evidence; rerunning never replaces an earlier artifact with different bytes.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONNECTION = ROOT / "logs/cloud-connection-optimization-current.json"
MAX_BYTES = 8 * 1024**3
MAX_FILES = 500
MAX_INLINE_BYTES = 16 * 1024**2
MAX_INLINE_TOTAL_BYTES = 64 * 1024**2
INLINE_SUFFIXES = (".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml", ".csv", ".tsv", ".sha256", ".out", ".err")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
RESERVED_WINDOWS_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value, required_prefix=None):
    require(isinstance(value, str) and value and "\\" not in value, "Use a safe POSIX relative evidence path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and value == path.as_posix(), "Path must be canonical and relative")
    require(len(path.parts) >= 2 and path.parts[0] in ("results", "logs"), "Evidence path must start with results/ or logs/")
    require(required_prefix is None or path.parts[0] == required_prefix, "Unexpected evidence path prefix")
    for part in path.parts:
        require(part not in (".", "..") and bool(SAFE_COMPONENT.fullmatch(part))
                and not part.endswith(".") and part.split(".", 1)[0].upper() not in RESERVED_WINDOWS_NAMES,
                "Evidence path contains an unsafe component")
    return path.as_posix()


def local_target(output, relative):
    candidate = output.joinpath(*PurePosixPath(relative).parts)
    require(candidate.resolve().is_relative_to(output), "Local destination escapes output directory")
    return candidate


def write_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


# This code is sent through a base64 envelope. It only reads the two requested
# roots and /proc metadata. Process argv is inspected in memory solely to find
# whitelisted script basenames; argv and environment are never returned.
REMOTE_INVENTORY = r'''
import base64, hashlib, json, os, re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

root = Path(request['root']).resolve(strict=True)
require(root.is_absolute(), 'Remote root must be absolute')
files, seen, total_bytes, inline_bytes = [], set(), 0, 0
for relative in request['relative_paths']:
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and pure.parts[0] in ('results','logs')
            and all(p not in ('.','..') for p in pure.parts), 'Invalid remote request')
    target = root.joinpath(*pure.parts)
    require(target.exists(), 'Requested evidence path does not exist: '+relative)
    require(not target.is_symlink() and target.resolve().is_relative_to(root), 'Unsafe remote evidence root')
    # Check every traversed path, including directory symlinks, instead of
    # silently following or excluding evidence outside the requested tree.
    stack = [target]
    while stack:
        current = stack.pop()
        require(not current.is_symlink(), 'Symlink in requested evidence tree')
        resolved = current.resolve(strict=True)
        require(resolved.is_relative_to(root), 'Evidence escaped project root')
        if current.is_dir():
            stack.extend(sorted(current.iterdir(), reverse=True))
            continue
        require(current.is_file(), 'Nonregular artifact in requested evidence tree')
        rel = current.relative_to(root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        before = current.stat()
        if (current.suffix.lower() in request['inline_suffixes']
                and before.st_size <= request['max_inline_bytes']
                and inline_bytes+before.st_size <= request['max_inline_total_bytes']):
            # One bounded read fixes the exact byte snapshot even if a log
            # continues appending. Do not hash then reopen its live pathname.
            with current.open('rb') as stream:
                captured = stream.read(request['max_inline_bytes']+1)
            if (len(captured) <= request['max_inline_bytes']
                    and inline_bytes+len(captured) <= request['max_inline_total_bytes']):
                total_bytes += len(captured)
                inline_bytes += len(captured)
                require(len(files)+1 <= request['max_files'] and total_bytes <= request['max_bytes'], 'Remote evidence limit exceeded')
                files.append({'relative_path':rel,'size_bytes':len(captured),
                              'sha256':hashlib.sha256(captured).hexdigest(),'mtime_ns':before.st_mtime_ns,
                              'transfer_method':'inline_single_read_snapshot',
                              'snapshot_note':'Exact captured bytes; an actively appended text record may be incomplete at EOF',
                              'inline_base64':base64.b64encode(captured).decode('ascii')})
                del captured
                continue
            del captured
        total_bytes += before.st_size
        require(len(files)+1 <= request['max_files'] and total_bytes <= request['max_bytes'], 'Remote evidence limit exceeded')
        file_sha = digest(current)
        after = current.stat()
        require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
                'Remote artifact changed during hashing: '+rel)
        files.append({'relative_path':rel,'size_bytes':after.st_size,'sha256':file_sha,'mtime_ns':after.st_mtime_ns,
                      'transfer_method':'scp_sha_verified'})

processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc/'comm').read_text().strip().lower()
        if not (comm.startswith('python') or comm in ('timeout','bash','sh')):
            continue
        argv = (proc/'cmdline').read_bytes().split(b'\0')
        script = None
        for argument in argv:
            name = argument.decode('utf-8','replace').rsplit('/',1)[-1]
            if re.fullmatch(r'(?:run|collect)_babylm_[A-Za-z0-9_.-]+\.py',name):
                script = name
                break
        del argv
        if script is None:
            continue
        text = (proc/'stat').read_text()
        tail = text[text.rfind(')')+2:].split()
        processes.append({'pid':int(proc.name),'ppid':int(tail[1]),'pgid':int(tail[2]),
                          'start_ticks':int(tail[19]),'state':tail[0],'script_basename':script})
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        continue
print(json.dumps({'schema_version':1,'utc':datetime.now(timezone.utc).isoformat(),
                  'files':sorted(files,key=lambda x:x['relative_path']),
                  'total_size_bytes':total_bytes,'processes':sorted(processes,key=lambda x:x['pid']),
                  'process_visibility':'Current container namespace; disappearing/inaccessible processes omitted; not host-wide proof',
                  'model_calls':0},allow_nan=False))
'''


def connect_arguments(connection):
    require(re.fullmatch(r"[A-Za-z0-9.-]+", connection["ssh_host"]), "Invalid SSH host")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", connection["ssh_user"]), "Invalid SSH user")
    require(type(connection["ssh_port"]) is int and 0 < connection["ssh_port"] <= 65535, "Invalid SSH port")
    key = Path(connection["ssh_key_path"]).expanduser()
    require(key.is_file(), "Existing SSH key is unavailable")
    remote_root = PurePosixPath(connection["root"])
    require(remote_root.is_absolute() and ".." not in remote_root.parts, "Invalid remote project root")
    interpreter = PurePosixPath(connection["python"])
    require(interpreter.is_absolute() and ".." not in interpreter.parts, "Invalid remote Python path")
    common = ["-i", str(key), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
              "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=2"]
    address = connection["ssh_user"] + "@" + connection["ssh_host"]
    return common, address


def inventory(connection, relatives):
    common, address = connect_arguments(connection)
    request = {"root": connection["root"], "relative_paths": relatives, "max_bytes": MAX_BYTES, "max_files": MAX_FILES,
               "max_inline_bytes": MAX_INLINE_BYTES, "max_inline_total_bytes": MAX_INLINE_TOTAL_BYTES,
               "inline_suffixes": INLINE_SUFFIXES}
    source = "request=" + repr(request) + "\n" + REMOTE_INVENTORY
    encoded = base64.b64encode(source.encode()).decode()
    payload = "import base64;exec(base64.b64decode('" + encoded + "'))"
    remote_command = shlex.quote(connection["python"]) + " -c " + shlex.quote(payload)
    completed = subprocess.run(["ssh", *common, "-p", str(connection["ssh_port"]), address, remote_command],
                               capture_output=True, text=True, timeout=600)
    # Never relay arbitrary ssh stderr or the raw command; keep errors free of
    # key paths, private argv and server-supplied shell output.
    require(completed.returncode == 0, f"Remote read-only inventory failed (exit {completed.returncode}); no transfer attempted")
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError("Remote inventory did not return the expected JSON") from error
    require(result.get("schema_version") == 1 and isinstance(result.get("files"), list), "Unexpected remote inventory schema")
    require(len(result["files"]) <= MAX_FILES, "Too many remote files")
    seen, total = set(), 0
    for item in result["files"]:
        rel = safe_relative(item["relative_path"])
        require(any(rel == requested or rel.startswith(requested + "/") for requested in relatives), "Remote file outside requested paths")
        require(rel not in seen, "Repeated inventory path")
        seen.add(rel)
        require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0, "Invalid file size")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "Invalid remote file SHA")
        if item.get("transfer_method") == "inline_single_read_snapshot":
            require(item["size_bytes"] <= MAX_INLINE_BYTES and isinstance(item.get("inline_base64"), str), "Invalid inline snapshot")
        else:
            require(item.get("transfer_method") == "scp_sha_verified" and "inline_base64" not in item, "Unexpected transfer method")
        total += item["size_bytes"]
    require(total == result["total_size_bytes"] and total <= MAX_BYTES, "Remote byte limit or tally mismatch")
    safe_processes = []
    for process in result.get("processes", []):
        allowed = {key: process[key] for key in ("pid", "ppid", "pgid", "start_ticks", "state", "script_basename")}
        require(all(type(allowed[key]) is int and allowed[key] >= 0 for key in ("pid", "ppid", "pgid", "start_ticks")), "Invalid process metadata")
        require(re.fullmatch(r"(?:run|collect)_babylm_[A-Za-z0-9_.-]+\.py", allowed["script_basename"]), "Invalid process script basename")
        require(isinstance(allowed["state"], str) and len(allowed["state"]) == 1, "Invalid process state")
        safe_processes.append(allowed)
    return {"schema_version": 1, "utc": result["utc"], "files": result["files"], "total_size_bytes": total,
            "processes": safe_processes, "process_visibility": "Current container namespace only; inaccessible processes may be omitted"}


def collect(result_relative, launch_relative, output_directory):
    relatives = [safe_relative(result_relative, "results"), safe_relative(launch_relative, "logs")]
    connection = json.loads(CONNECTION.read_text(encoding="utf-8"))
    common, address = connect_arguments(connection)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(output.is_dir(), "Invalid output directory")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    manifest = {"schema_version": 1, "collection_id": run_id, "started_utc": utc(), "status": "running",
                "connection_record": str(CONNECTION.relative_to(ROOT)).replace("\\", "/"),
                "connection_record_sha256": sha(CONNECTION), "pod_id": connection.get("pod_id"),
                "requested_paths": relatives, "max_files": MAX_FILES, "max_bytes": MAX_BYTES,
                "receipts": [], "downloaded_bytes": 0, "reused_bytes": 0,
                "credential_material_archived": False, "model_calls": 0}
    incoming_root = output / ".incoming" / run_id
    manifest_path = output / ("manifest-" + run_id + ".json")
    try:
        remote = inventory(connection, relatives)
        # Payloads are kept only in memory until decoded to their exact files.
        # Public manifests and receipts never contain base64 payload text.
        inline_payloads = {item["relative_path"]: item.pop("inline_base64")
                           for item in remote["files"] if "inline_base64" in item}
        manifest["remote_inventory"] = remote
        write_exclusive(output / ("inventory-" + run_id + ".json"), remote)
        for item in remote["files"]:
            remote_relative, expected = item["relative_path"], item["sha256"]
            target = local_target(output, remote_relative)
            receipt = {**item, "started_utc": utc(), "status": "pending"}
            manifest["receipts"].append(receipt)
            if target.exists():
                require(target.is_file(), "Existing evidence destination is not a regular file")
                previous = sha(target)
                if previous != expected:
                    receipt["preserved_existing_sha256"] = previous
                    receipt["preserved_existing_path"] = remote_relative
                    target = local_target(output, ".versions/" + expected + "/" + remote_relative)
            if target.exists():
                require(target.is_file() and target.stat().st_size == item["size_bytes"] and sha(target) == expected,
                        "Existing immutable evidence version differs; refusing overwrite")
                receipt.update(status="reused_sha_verified", local_relative_path=target.relative_to(output).as_posix(), completed_utc=utc())
                manifest["reused_bytes"] += item["size_bytes"]
                continue
            temporary = local_target(output, ".incoming/" + run_id + "/" + remote_relative)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            require(not temporary.exists(), "Fresh incoming file required")
            receipt["incoming_relative_path"] = temporary.relative_to(output).as_posix()
            if item["transfer_method"] == "inline_single_read_snapshot":
                captured = base64.b64decode(inline_payloads.pop(remote_relative), validate=True)
                require(len(captured) <= MAX_INLINE_BYTES, "Inline snapshot exceeds byte limit")
                with temporary.open("xb") as stream:
                    stream.write(captured)
                    stream.flush()
                    os.fsync(stream.fileno())
                del captured
            else:
                remote_absolute = (PurePosixPath(connection["root"]) / remote_relative).as_posix()
                # Paths are restricted to safe POSIX components; no wildcards,
                # whitespace, colons or shell metacharacters are accepted.
                copied = subprocess.run(["scp", "-q", *common, "-P", str(connection["ssh_port"]),
                                         address + ":" + remote_absolute, str(temporary)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
                require(copied.returncode == 0, f"SCP failed (exit {copied.returncode}); incoming bytes preserved")
            actual_size, actual = temporary.stat().st_size, sha(temporary)
            receipt.update(downloaded_size_bytes=actual_size, downloaded_sha256=actual)
            manifest["downloaded_bytes"] += actual_size
            require(actual_size == item["size_bytes"] and actual == expected,
                    "Transferred evidence SHA/size mismatch; incoming bytes preserved and collection stopped")
            target.parent.mkdir(parents=True, exist_ok=True)
            require(target.resolve().is_relative_to(output) and not target.exists(), "Destination changed during transfer; incoming bytes preserved")
            # On Windows rename refuses an existing target. The second check
            # above also prevents ordinary sequential overwrite on POSIX.
            temporary.rename(target)
            receipt.update(status="downloaded_sha_verified", local_relative_path=target.relative_to(output).as_posix(), completed_utc=utc())
        manifest["status"] = "collection_complete_all_files_sha_verified"
    except BaseException as error:
        if manifest["receipts"] and manifest["receipts"][-1]["status"] == "pending":
            manifest["receipts"][-1].update(status="failed_preserved", completed_utc=utc())
        # A subprocess TimeoutExpired exception includes the entire command
        # (and key path), so record its type and a fixed explanation only.
        message = str(error) if isinstance(error, ValueError) else "Collector failed; downloaded/incoming evidence remains preserved"
        manifest.update(status="collection_failed_evidence_preserved", error_type=type(error).__name__, error=message)
    finally:
        manifest["completed_utc"] = utc()
        write_exclusive(manifest_path, manifest)
        # Every historical manifest is immutable. This small current index is
        # the sole replaceable file and names the new immutable manifest.
        index = {"schema_version": 1, "latest_manifest": manifest_path.name,
                 "latest_manifest_sha256": sha(manifest_path), "status": manifest["status"], "utc": utc()}
        temporary_index = output / (".manifest-index-" + run_id + ".tmp")
        write_exclusive(temporary_index, index)
        os.replace(temporary_index, output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-relative", required=True)
    parser.add_argument("--launch-relative", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.result_relative, args.launch_relative, args.output_directory)
    print(json.dumps({"status": result["status"], "collection_id": result["collection_id"],
                      "receipts": len(result["receipts"]), "downloaded_bytes": result["downloaded_bytes"],
                      "reused_bytes": result["reused_bytes"], "model_calls": 0}, allow_nan=False))
    return 0 if result["status"] == "collection_complete_all_files_sha_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
