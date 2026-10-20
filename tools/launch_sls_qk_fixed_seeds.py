"""Sequential fixed-seed queue; fail once, never restart a failed experiment."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

from tools.train_relational_preservation import atomic_json


REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO/"models/SLS_pretrained_attention_fixed_seeds_20261010"
CONFIGS = [REPO/f"experiments/configs/sls_pretrained_attention_seed{seed}.yaml"
           for seed in (1001, 7)]


def worker():
    for index, config in enumerate(CONFIGS):
        c = yaml.safe_load(config.read_text())
        root = Path(c["output_dir"])
        log = REPO/f"experiments/logs/sls_pretrained_attention_seed{c['seed']}.log"
        if any((root/name).exists() for name in ("experiment.json", "launch.json", "warmup_protocol.json")):
            raise RuntimeError("Existing child run: no overwrite")
        root.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", "tools.train_sls_qk_seed", "--config", str(config)],
                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        launch = dict(pid=child.pid, process_create_time=psutil.Process(child.pid).create_time(),
                      seed=c["seed"], log=str(log), config=str(config), port=30646,
                      unix_started_at=time.time(), queue_pid=psutil.Process().pid)
        atomic_json(root/"launch.json", launch)
        atomic_json(QUEUE/"status.json", dict(status="running", seed=c["seed"], index=index,
                                              child=launch, fixed_seed_order=[1001, 7]))
        print("CHILD_LAUNCH", json.dumps(launch), flush=True)
        result = child.wait()
        state = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
        if result != 0 or state.get("status") != "complete":
            atomic_json(QUEUE/"status.json", dict(status="failed", seed=c["seed"], exit_code=result,
                        child_status=state, remaining_seeds_not_started=[7] if index == 0 else [],
                        automatic_restart=False))
            raise RuntimeError(f"Seed {c['seed']} did not complete; no restart or next-seed launch")
        print("CHILD_COMPLETE", c["seed"], flush=True)
    atomic_json(QUEUE/"status.json", dict(status="complete", completed_seeds=[1001, 7],
                                          unix_ended_at=time.time(), automatic_restart=False))
    print("QUEUE_COMPLETE", flush=True)


def launch():
    if (QUEUE/"launch.json").exists():
        raise RuntimeError("Queue already registered; no replay")
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(token.startswith("tools.train") for token in process.info["cmdline"] or []):
            raise RuntimeError(f"Another experiment trainer is running: {process.pid}")
    for config in CONFIGS:
        c = yaml.safe_load(config.read_text())
        root = Path(c["output_dir"]).resolve()
        if not root.is_relative_to(Path("/root/rivermind-data")):
            raise ValueError("Persistent outputs only")
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Existing child directory: {root}")
    QUEUE.mkdir(parents=True, exist_ok=True)
    log = REPO/"experiments/logs/sls_pretrained_attention_fixed_seeds_20261010.log"
    with log.open("x") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "tools.launch_sls_qk_fixed_seeds", "--worker"],
            cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  log=str(log), configs=[str(p) for p in CONFIGS], fixed_seed_order=[1001, 7],
                  port=30646, unix_started_at=time.time())
    atomic_json(QUEUE/"launch.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker() if args.worker else launch()
