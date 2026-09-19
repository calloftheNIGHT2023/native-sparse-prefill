import importlib.util, json, os, shutil, subprocess
import torch
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
    transformers=bool(importlib.util.find_spec('transformers')),
    runpodctl=shutil.which('runpodctl'), pod_id=os.getenv('RUNPOD_POD_ID'),
    api_key_present=bool(os.getenv('RUNPOD_API_KEY')), torch_cuda=torch.cuda.is_available())))
if shutil.which('runpodctl'):
    subprocess.run(['runpodctl', 'stop', 'pod', '--help'],check=False)
