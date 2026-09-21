"""Local filesystem/transport mocks only; no SSH, CUDA, or model imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "logs/backup_dw_model_ranges_20260921.py"
spec = importlib.util.spec_from_file_location("range_backup", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class LocalReadTransport:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.mode = None

    def __call__(self, entry, action, offset, length, timeout):
        self.calls.append((action, offset, length))
        if self.mode == "timeout" and action == "chunk":
            raise TimeoutError("Mock transport timeout; no retry")
        request = dict(root=str(self.root), entry=entry, action=action, offset=offset, length=length)
        proc = subprocess.run([sys.executable, "-c", mod.REMOTE], input=json.dumps(request).encode(),
                              capture_output=True, timeout=timeout)
        if proc.returncode:
            raise ValueError("Local source admission rejected")
        header, _, body = proc.stdout.partition(b"\n")
        meta = json.loads(header)
        if self.mode in ("bad_chunk", "bad_whole") and action == "chunk":
            body = bytes([body[0] ^ 1]) + body[1:]
            if self.mode == "bad_whole":
                meta["chunk_sha256"] = mod.digest(body)
        return meta, body


class RangeBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.relative = mod.PREFIX + "/D/train/run/checkpoint.pt"
        self.path = self.root / self.relative
        self.path.parent.mkdir(parents=True)
        self.content = (bytes(range(256)) * (2 * mod.MIB // 256)) + b"final-test-17-byte"
        self.path.write_bytes(self.content)
        (self.path.parent / "summary.json").write_text(json.dumps({"status":"epoch_complete",
            "counts":{"updates":1413}, "cursor":{"epoch":1,"position":0}}))
        self.entry = dict(relative_path=self.relative, size_bytes=len(self.content),
                          mtime_ns=self.path.stat().st_mtime_ns, sha256=mod.digest(self.content))
        self.inv = self.root / "inventory.json"
        self.inv.write_text(json.dumps({"files":[self.entry]}))
        self.out = self.root / "backup"
        self.transport = LocalReadTransport(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def backup(self, **kw):
        return mod.backup(self.entry, {"sha256":mod.file_sha(self.inv)}, self.out,
                          self.transport, max_seconds=30, **kw)

    def test_inventory_pin_and_path_rejections(self):
        self.assertEqual(mod.inventory_entry(self.inv, mod.file_sha(self.inv), self.relative), self.entry)
        with self.assertRaises(ValueError): mod.inventory_entry(self.inv, "0"*64, self.relative)
        for path in (self.relative.replace("/D/", "/X/"), self.relative.replace("checkpoint.pt", "../checkpoint.pt"),
                     self.relative.replace("checkpoint.pt", "events.jsonl"), "C:/outside.pt"):
            with self.subTest(path=path), self.assertRaises(ValueError): mod.validate_relative(path)

    def test_partial_resume_whole_sha_and_immutable_receipts(self):
        first = self.backup(max_bytes=mod.MIB)
        self.assertEqual(first["status"], "bounded_partial_verified_chunks_retained")
        self.assertEqual(first["downloaded_bytes"], mod.MIB)
        receipt = next(self.out.rglob("chunks/000000000000.json"))
        raw = receipt.read_bytes()
        second = self.backup(max_bytes=4*mod.MIB)
        self.assertEqual(second["status"], "complete_whole_sha_verified")
        self.assertEqual(second["reused_bytes"], mod.MIB)
        self.assertEqual(Path(second["artifact_path"]).read_bytes(), self.content)
        self.assertEqual(receipt.read_bytes(), raw)
        offsets = [c[1] for c in self.transport.calls if c[0] == "chunk"]
        self.assertEqual(offsets, [0, mod.MIB, 2*mod.MIB])
        count = len(self.transport.calls)
        third = self.backup(max_bytes=0)
        self.assertEqual(third["status"], "complete_existing_artifact_sha_verified")
        self.assertEqual(len(self.transport.calls), count)
        self.assertEqual(len(list(self.out.rglob("attempts/*.json"))), 3)

    def test_corrupt_existing_chunk_rejected_without_overwrite(self):
        self.backup(max_bytes=mod.MIB)
        chunk = next(self.out.rglob("chunks/*.bin"))
        chunk.write_bytes(b"damaged")
        result = self.backup(max_bytes=4*mod.MIB)
        self.assertEqual(result["status"], "failed_evidence_preserved")
        self.assertEqual(chunk.read_bytes(), b"damaged")
        self.assertFalse(list(self.out.rglob("artifact.pt")))

    def test_remote_fingerprint_change_rejected(self):
        self.backup(max_bytes=mod.MIB)
        os.utime(self.path, ns=(self.entry["mtime_ns"]+1000000, self.entry["mtime_ns"]+1000000))
        before = len([x for x in self.transport.calls if x[0] == "chunk"])
        result = self.backup(max_bytes=4*mod.MIB)
        self.assertEqual(result["status"], "failed_evidence_preserved")
        self.assertEqual(len([x for x in self.transport.calls if x[0] == "chunk"]), before)

    def test_unfinished_source_rejected(self):
        (self.path.parent / "summary.json").write_text(json.dumps({"status":"running"}))
        result = self.backup(max_bytes=mod.MIB)
        self.assertEqual(result["status"], "failed_evidence_preserved")
        self.assertFalse(list(self.out.rglob("chunks/*.bin")))

    def test_chunk_transport_sha_and_whole_sha_both_enforced(self):
        self.transport.mode = "bad_chunk"
        result = self.backup(max_bytes=4*mod.MIB)
        self.assertEqual(result["status"], "failed_evidence_preserved")
        self.assertFalse(list(self.out.rglob("chunks/*.json")))
        self.transport.mode = "bad_whole"
        result = self.backup(max_bytes=4*mod.MIB)
        self.assertEqual(result["status"], "failed_evidence_preserved")
        self.assertIn("Final whole-file SHA", result["error"])
        self.assertFalse(list(self.out.rglob("artifact.pt")))
        self.assertFalse(list(self.out.rglob("complete.json")))
        self.assertTrue(list(self.out.rglob("incoming/*assembled.pt")))

    def test_timeout_no_retry_and_payload_limit(self):
        first = self.backup(max_bytes=mod.MIB-1)
        self.assertEqual(first["downloaded_bytes"], 0)
        self.assertEqual([x[0] for x in self.transport.calls], ["probe"])
        self.transport.mode = "timeout"
        second = self.backup(max_bytes=mod.MIB)
        self.assertEqual(second["status"], "bounded_partial_verified_chunks_retained")
        self.assertEqual(second["automatic_retries"], 0)
        self.assertEqual(len([x for x in self.transport.calls if x[0] == "chunk"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
