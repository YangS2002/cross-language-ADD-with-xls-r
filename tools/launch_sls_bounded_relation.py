"""Launch only this registered candidate, refusing replay/concurrent training."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    config = repo/"experiments/configs/sls_bounded_relation_seed20.yaml"
    c = yaml.safe_load(config.read_text())
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent disk required")
    if (root/"experiment.json").exists() or (root/"launch.json").exists():
        raise RuntimeError("Experiment already launched")
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(token.startswith("tools.train") for token in process.info["cmdline"] or []):
            raise RuntimeError(f"Another trainer is running: {process.pid}")
    log = repo/"experiments/logs/sls_bounded_relation_seed20.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", "tools.train_sls_bounded_relation",
            "--config", str(config)], cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  module="tools.train_sls_bounded_relation", log=str(log), config=str(config),
                  port=30646, unix_started_at=time.time())
    (root/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
