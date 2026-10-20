"""Bounded official ASR download, exclusively to the persistent data volume."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

URL = 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--seconds', type=int, default=180)
    a = p.parse_args()
    directory = Path(a.directory).resolve()
    if not directory.is_relative_to(Path('/root/rivermind-data')):
        raise ValueError('Persistent volume required')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / URL.rsplit('/', 1)[1]
    if target.exists():
        raise FileExistsError('Do not overwrite existing weights: ' + str(target))
    partial = target.with_suffix('.pth.partial')
    start = time.monotonic()
    size = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=20) as response, partial.open('xb') as out:
            expected = int(response.headers.get('Content-Length', 0))
            last = start
            while True:
                if time.monotonic() - start > a.seconds:
                    raise TimeoutError('Download budget exhausted; partial retained')
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block); sha.update(block); size += len(block)
                now = time.monotonic()
                if now-last >= 10:
                    rate = size / (now-start)
                    print(json.dumps(dict(event='progress', bytes=size, expected=expected,
                        mib_per_second=rate/2**20, eta_seconds=(expected-size)/rate if expected else None)), flush=True)
                    last = now
                    if now-start >= 30 and rate < 0.5*2**20:
                        raise TimeoutError('Slow download below 0.5 MiB/s; partial retained')
            if not expected or size != expected:
                raise ValueError('Missing or mismatched Content-Length')
            out.flush(); os.fsync(out.fileno())
        os.replace(partial, target)
        result = dict(event='completed', url=URL, path=str(target), bytes=size,
            sha256=sha.hexdigest(), seconds=time.monotonic()-start,
            verification='HTTP length and local SHA; no independently published SHA comparison')
        target.with_suffix('.download.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception as error:
        print(json.dumps(dict(event='failed', error=repr(error), bytes=size,
            seconds=time.monotonic()-start, partial=str(partial))), flush=True)
        raise


if __name__ == '__main__':
    main()
