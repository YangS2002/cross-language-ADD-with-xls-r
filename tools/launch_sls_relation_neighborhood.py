"""Launch the registered candidate once, without restarting any failed process."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    config = repo/"experiments/configs/sls_relation_neighborhood_seed20.yaml"
    c = yaml.safe_load(config.read_text())
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent output required")
    if root.exists():
        raise RuntimeError("Output exists: no overwrite/restart")
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(token.startswith("tools.train") for token in process.info["cmdline"] or []):
            raise RuntimeError(f"Another trainer is running: {process.pid}")
    log = repo/"experiments/logs/sls_relation_neighborhood_seed20.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", "tools.train_sls_relation_neighborhood",
                                    "--config", str(config)], cwd=repo, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  log=str(log), config=str(config), port=30646, unix_started_at=time.time())
    (root/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
