"""Switch only CKA computation device; reuse all completed feature caches."""
import json
import subprocess
import sys
from pathlib import Path
import psutil
repo=Path(__file__).resolve().parents[1]
root=repo/'models/UK_representation_diagnostic_seed20'
state=json.loads((root/'status.json').read_text())
launch=json.loads((root/'launch.json').read_text())
if (root/'analysis_restart.json').exists():
    launch.update(json.loads((root/'analysis_restart.json').read_text()))
if state['status']!='analysing' or (root/'results.json').exists():
    raise RuntimeError('Do not interrupt completed/partly fitted probes')
for m in ('original','finetuned'):
    for d in ('a','uk'):
        if not (root/f'features_{m}_{d}.npz').is_file(): raise RuntimeError('Incomplete extraction')
p=psutil.Process(launch['pid'])
if abs(p.create_time()-launch['process_create_time'])>1 or 'tools.run_uk_representation_diagnostic' not in p.cmdline():
    raise RuntimeError('PID identity mismatch')
p.terminate();p.wait(timeout=10)
with Path(launch['log']).open('a') as stream:
    stream.write('\nANALYSIS_CACHE_FIX materialize_NPZ_once; CUDA_CKA; all features reused, no encoder rerun\n');stream.flush()
    q=subprocess.Popen([sys.executable,'-u','-m','tools.run_uk_representation_diagnostic','--analyse-only'],
        cwd=repo,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=q.pid,process_create_time=psutil.Process(q.pid).create_time(),reason='Materialize lazy NPZ once and CUDA CKA acceleration',
       features_reused=True,encoder_rerun=False,old_pid=p.pid)
(root/'analysis_restart.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
