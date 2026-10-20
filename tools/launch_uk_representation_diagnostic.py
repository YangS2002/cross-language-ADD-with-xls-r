import json
import subprocess
import sys
from pathlib import Path
import psutil

repo=Path(__file__).resolve().parents[1]
root=repo/'models/UK_representation_diagnostic_seed20'
log=repo/'experiments/logs/uk_representation_diagnostic_seed20.log'
root.mkdir(parents=True,exist_ok=True);log.parent.mkdir(exist_ok=True)
if log.exists() or (root/'launch.json').exists(): raise RuntimeError('Already launched')
with log.open('x') as stream:
    p=subprocess.Popen([sys.executable,'-u','-m','tools.run_uk_representation_diagnostic'],cwd=repo,
        stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record=dict(pid=p.pid,process_create_time=psutil.Process(p.pid).create_time(),log=str(log))
(root/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
