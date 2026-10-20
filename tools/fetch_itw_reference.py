"""Download the author's ITW artifact by checked HTTP ranges; verify LFS SHA256."""
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

URL = "https://hf-mirror.com/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
SIZE = 8161489023
SHA256 = "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a"
ROOT = Path("/root/rivermind-data/datasets/itw")
CHUNK = 32 * 1024**2


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    partial = ROOT / "release_in_the_wild.parallel.partial"
    progress = ROOT / "parallel_progress.json"
    complete = ROOT / "release_in_the_wild.zip"
    if complete.exists():
        raise RuntimeError("Final ITW archive already exists; refusing overwrite")
    chunks = (SIZE + CHUNK - 1) // CHUNK
    done = set()
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved["size"] != SIZE or saved["sha256"] != SHA256 or not partial.exists():
            raise RuntimeError("Partial download signature mismatch")
        done = set(saved["done"])
    if not partial.exists():
        with open(partial, "xb") as stream:
            stream.truncate(SIZE)

    def fetch(index):
        start = index * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        for attempt in range(4):
            try:
                with requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError("Server ignored requested HTTP range")
                    offset = start
                    with open(partial, "r+b") as stream:
                        stream.seek(start)
                        for block in response.iter_content(1024**2):
                            if offset + len(block) > end + 1:
                                raise RuntimeError("Oversized HTTP range")
                            stream.write(block)
                            offset += len(block)
                        stream.flush()
                    if offset != end + 1:
                        raise RuntimeError("Incomplete HTTP range")
                return index
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    start_time = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, i) for i in range(chunks) if i not in done]
        for future in concurrent.futures.as_completed(futures):
            done.add(future.result())
            temp = progress.with_suffix(".tmp")
            temp.write_text(json.dumps({"size": SIZE, "sha256": SHA256, "done": sorted(done)}))
            os.replace(temp, progress)
            if len(done) % 10 == 0:
                print("DOWNLOAD", len(done), chunks, "seconds", round(time.monotonic() - start_time, 1), flush=True)
    h = hashlib.sha256()
    with open(partial, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise RuntimeError(f"ITW SHA256 mismatch: {h.hexdigest()}")
    os.replace(partial, complete)
    print("ITW_VERIFIED", SIZE, SHA256, flush=True)


if __name__ == "__main__":
    main()
