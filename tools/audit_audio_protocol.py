"""Read audio headers from existing protocols without changing audio files."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import soundfile as sf
import yaml


def inspect(item):
    path, label = item
    info = sf.info(path)
    return dict(sample_rate=info.samplerate, channels=info.channels,
                frames=info.frames, path=str(path), label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = {}
    for split in ('train','dev','test'):
        items=[]
        for line in Path(config[f'{split}_label_path']).read_text().splitlines():
            fields=line.split()
            if not fields:
                continue
            name=fields[1]
            if '.' not in name:
                name += '.wav'
            items.append((Path(config[f'{split}_data_path'])/name, fields[-1]))
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(inspect,items))
        counts=Counter(f"sr={r['sample_rate']},channels={r['channels']},label={r['label']}" for r in rows)
        mismatched=[r for r in rows if r['sample_rate']!=16000 or r['channels']!=1]
        result[split]=dict(n=len(rows),counts=dict(counts),non_16k_mono=len(mismatched),examples=mismatched[:10])
        print(split,json.dumps(result[split]),flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
