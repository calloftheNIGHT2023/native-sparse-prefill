import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
records=[]
for name in ['upload-manifest.json','stage-upload-manifest.json']:
    for r in json.loads((root/'logs'/name).read_text()):
        p=root/r['path']; assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
        records.append(r['path'])
print(json.dumps(dict(verified_upload_files=len(records))))
