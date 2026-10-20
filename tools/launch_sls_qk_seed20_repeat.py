"""Launch the user-requested original Q/K seed20 repeat, never overwrite history."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    config = repo/"experiments/configs/sls_pretrained_attention_seed20_repeat_20261011.yaml"
    c = yaml.safe_load(config.read_text())
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")) or root.exists():
        raise RuntimeError("New persistent experiment directory required")
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(token.startswith("tools.train") for token in process.info["cmdline"] or []):
            raise RuntimeError(f"Another trainer is running: {process.pid}")
    log = repo/"experiments/logs/sls_pretrained_attention_seed20_repeat_20261011.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", "tools.train_sls_qk_seed20_repeat",
                                    "--config", str(config)], cwd=repo, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  log=str(log), config=str(config), port=30925, unix_started_at=time.time())
    (root/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
