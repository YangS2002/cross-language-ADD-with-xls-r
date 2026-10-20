"""Launch the registered candidate once; persistent log and exact process identity."""
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
import yaml


def main():
    repo=Path(__file__).resolve().parents[1]
    config=repo/'experiments/configs/sls_dependency_control_seed20.yaml'
    c=yaml.safe_load(config.read_text());root=Path(c['output_dir']).resolve()
    if not root.is_relative_to(Path('/root/rivermind-data')):raise ValueError('Persistent output required')
    if (root/'experiment.json').exists() or (root/'launch.json').exists():raise RuntimeError('Run already registered')
    for proc in psutil.process_iter(['pid','cmdline']):
        if 'tools.train_sls_dependency_control' in (proc.info['cmdline'] or []):
            raise RuntimeError('Candidate already running')
    log=repo/'experiments/logs/sls_dependency_control_seed20.log'
    log.parent.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        process=subprocess.Popen([sys.executable,'-u','-m','tools.train_sls_dependency_control',
            '--config',str(config)],cwd=repo,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(pid=process.pid,process_create_time=psutil.Process(process.pid).create_time(),
        log=str(log),config=str(config),port=30646,unix_started_at=time.time())
    (root/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
