"""Check training dependencies without altering unrelated image packages."""
import json,subprocess,sys
from pathlib import Path
import torch,transformers,pyarrow
result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
lines=[x.strip() for x in result.stdout.splitlines() if x.strip()]
known='pygobject 3.42.1 requires pycairo, which is not installed.'
unexpected=[x for x in lines if x!=known and x!='No broken requirements found.']
report=dict(torch=str(torch.__version__),transformers=transformers.__version__,pyarrow=pyarrow.__version__,
    cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),pip_check_exit_code=result.returncode,
    pip_check_lines=lines,preexisting_image_gui_dependency_missing=known in lines,
    unexpected_dependency_errors=unexpected,training_checks_passed=not unexpected and torch.cuda.is_available())
root=Path(__file__).resolve().parents[1]
(root/'logs/cloud-training-environment-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if not report['training_checks_passed']: sys.exit(1)
