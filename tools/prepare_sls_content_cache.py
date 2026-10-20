"""Freeze the validated source readout and cache only small source content targets."""
import argparse
import json
import time
from pathlib import Path
import torch
import torchaudio
from data import pad
from tools.prepare_sls_content_pilot import digest
from tools.sls_content_adversarial_model import FrozenContentReadout


def main():
    p=argparse.ArgumentParser(); p.add_argument('--directory',required=True)
    p.add_argument('--seconds',type=int,default=300); a=p.parse_args()
    root=Path(a.directory).resolve()
    if not root.is_relative_to(Path('/root/rivermind-data')): raise ValueError('Persistent disk required')
    output=root/'source_content_targets.pth'
    if output.exists(): raise FileExistsError(output)
    checked=json.loads((root/'source_readout/pilot_results.json').read_text())
    if not checked.get('all_genuine') or not checked['reference_pilot_passed']:
        raise ValueError('Full genuine source reference has not passed validation')
    source=Path('/root/rivermind-data/data/exp/ASVspoof2019.LA.cm.train.trn.txt')
    if digest(source)!=checked['source_protocol_sha256']: raise ValueError('Source protocol changed')
    base=Path('/root/rivermind-data/cross-language-ADD-with-xls-r/models/pretrained/xlsr2_300m.pt')
    if digest(base)!='b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9':
        raise ValueError('Base changed')
    readout_path=root/'source_readout/pilot_readout.pth'
    saved=torch.load(readout_path,map_location='cpu',weights_only=False)
    torch.set_num_threads(4)
    teacher=FrozenContentReadout(base,saved['gate']).cuda().eval()
    teacher.readout.load_state_dict(saved['readout']); teacher.requires_grad_(False)
    assert not any(p.requires_grad for p in teacher.parameters())
    letters=[i for i,v in enumerate(saved['alphabet']) if v.isalpha()]
    rows=[r.split() for r in source.read_text().splitlines() if r.strip()]
    audio_root=Path('/root/rivermind-data/data/asv2019_la_train')
    q=[]; valid=[]; mass=[]; rates={}; start=time.monotonic(); manifest=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(rows),8):
            if time.monotonic()-start>a.seconds: raise TimeoutError('Source cache budget exhausted')
            waves=[]
            for r in rows[i:i+8]:
                path=audio_root/r[1]; stat=path.stat()
                manifest.append([r[1],stat.st_size,stat.st_mtime_ns])
                wave,sr=torchaudio.load(str(path)); rates[sr]=rates.get(sr,0)+1
                waves.append(pad(wave).squeeze(0))
            frames=teacher.frames(torch.stack(waves).cuda())
            # Stable FP32 readout/posteriors, not autocast-dependent cache targets.
            with torch.autocast('cuda',enabled=False):
                qi,vi,mi=teacher.content_distribution(frames,letters)
            q.append(qi.cpu()); valid.append(vi.cpu()); mass.append(mi.cpu())
            if i%4000==0: print('CACHE',i,len(rows),time.monotonic()-start,flush=True)
    q=torch.cat(q); valid=torch.cat(valid); mass=torch.cat(mass)
    labels=torch.tensor([int(r[-1]=='bonafide') for r in rows])
    groups={}
    for group in ('bonafide','A01','A02','A03','A04','A05','A06'):
        ids=torch.tensor([r[-2]==group for r in rows])
        groups[group]=dict(count=int(ids.sum()),valid=int(valid[ids].sum()),
                           mean_letter_mass=float(mass[ids].mean()))
    import hashlib
    manifest_sha=hashlib.sha256(json.dumps(manifest,separators=(',',':')).encode()).hexdigest()
    metadata=dict(format='sls_source_content_v1',source_protocol_sha256=digest(source),
        base_sha256=digest(base),readout_sha256=digest(readout_path),
        audio_stat_manifest_sha256=manifest_sha,preprocessing='original data.pad64600, direct torchaudio; no resampling',
        letters=[saved['alphabet'][i] for i in letters],rows=len(rows),sample_rates=rates,
        coverage=groups,seconds=time.monotonic()-start,
        caveat='CTC pseudo-content only; attack-dependent reference errors remain possible')
    torch.save(dict(metadata=metadata,file_names=[r[1] for r in rows],labels=labels,
                    q=q,valid=valid,mass=mass),output)
    (root/'content_cache_manifest.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__': main()
