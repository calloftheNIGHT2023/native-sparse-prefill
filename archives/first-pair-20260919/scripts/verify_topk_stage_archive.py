"""Verify a stage archive and extract only into a new explicit directory."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone

def main(a):
    digest=hashlib.sha256(a.archive.read_bytes()).hexdigest();assert digest==a.sha256
    destination=a.extract_to.resolve();assert not destination.exists()
    with tarfile.open(a.archive,'r:gz') as t:
        manifest=json.load(t.extractfile('provenance/topk-efficiency-cloud-stage-manifest-v0.json'))
        expected={r['path']:r['sha256'] for r in manifest['files']}
        ms=t.getmembers();assert len(ms)==len(expected)+1
        for m in ms:
            path=PurePosixPath(m.name)
            assert m.isfile() and not path.is_absolute() and '..' not in path.parts
            if m.name in expected:assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==expected[m.name]
            else:assert m.name=='provenance/topk-efficiency-cloud-stage-manifest-v0.json'
        destination.mkdir(parents=True,exist_ok=False)
        for m in ms:
            p=destination/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
    assert all(hashlib.sha256((destination/p).read_bytes()).hexdigest()==h for p,h in expected.items())
    proof=dict(utc=datetime.now(timezone.utc).isoformat(),archive_sha256=digest,
               extracted_files_verified=len(expected),destination=str(destination),status='passed')
    print(json.dumps(proof))
    if a.report:a.report.write_text(json.dumps(proof,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--extract-to',required=True,type=Path);p.add_argument('--report',type=Path);main(p.parse_args())
