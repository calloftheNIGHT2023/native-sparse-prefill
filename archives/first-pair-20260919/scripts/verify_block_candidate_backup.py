"""Verify base+overlay archives, restore a new directory, check union of file hashes."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    destination=a.destination.resolve();assert not destination.exists()
    assert sha(a.base)=='38c4fc394a4bd544aa616019a45d2fe1b64a6aa0ddde96c93e3f94e7b99e9ed3'
    assert sha(a.overlay)=='d49da95be40cdc5c295338f08745ede3731fce15f886656621c54b1e29afda03'
    expected={};counts=[]
    destination.mkdir(parents=True,exist_ok=False)
    for archive,manifest_path in [(a.base,'provenance/topk-efficiency-cloud-stage-manifest-v0.json'),
        (a.overlay,'provenance/block-candidate-stage-overlay-v0.json')]:
        with tarfile.open(archive,'r:gz') as tar:
            manifest=json.load(tar.extractfile(manifest_path));records={r['path']:r['sha256'] for r in manifest['files']}
            members=tar.getmembers();assert len(members)==len(records)+1
            for m in members:
                path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts
                blob=tar.extractfile(m).read();h=hashlib.sha256(blob).hexdigest()
                if m.name in records:assert records[m.name]==h
                else:assert m.name==manifest_path
                target=(destination/m.name).resolve();assert destination in target.parents
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
                expected[m.name]=h
            counts.append(dict(archive=str(archive),members=len(members),sha256=sha(archive)))
    for path,h in expected.items():assert sha(destination/path)==h,path
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),destination=str(destination),
        archives=counts,verified_restored_files=len(expected),base_then_overlay_order_verified=True)
    a.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())
