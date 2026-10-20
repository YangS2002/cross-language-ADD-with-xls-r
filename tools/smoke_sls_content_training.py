"""One disposable real-model step; no checkpoint or target EER selection."""
import argparse
import json
import time
from pathlib import Path
import torch
import torchaudio
from data import pad
from tools.prepare_sls_content_pilot import digest
from tools.sls_content_adversarial_model import (ContentAdversarialSLS,
    content_optimizer,content_training_step)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); a=p.parse_args()
    root=Path(a.directory).resolve()
    if not root.is_relative_to(Path('/root/rivermind-data')): raise ValueError('Persistent disk required')
    if (root/'gpu_smoke.json').exists(): raise FileExistsError('Do not overwrite smoke results')
    base=Path('/root/rivermind-data/cross-language-ADD-with-xls-r/models/pretrained/xlsr2_300m.pt')
    warm=Path('/root/rivermind-data/cross-language-ADD-with-xls-r/models/SLS_conditional_residual_seed20/warmup.pth')
    cached=torch.load(root/'source_content_targets.pth',map_location='cpu',weights_only=False)
    if digest(base)!=cached['metadata']['base_sha256'] or digest(warm)!='32454f00414be2172ca7d84bfa631c3a8b551f45d415816354b0597645904742':
        raise ValueError('Base or common warmup changed')
    torch.set_num_threads(4); torch.manual_seed(20)
    model=ContentAdversarialSLS(checkpoint=base).cuda().train()
    model.backend.load_state_dict(torch.load(warm,map_location='cpu',weights_only=False)['backend'])
    optimizer=content_optimizer(model)
    # A pressure-test batch only, not the formal loader and not a sampling proposal.
    ids=torch.cat([(cached['labels']==y).nonzero().flatten()[:8] for y in (0,1)])
    waves=[]
    for i in ids.tolist():
        wave,_=torchaudio.load('/root/rivermind-data/data/asv2019_la_train/'+cached['file_names'][i])
        waves.append(pad(wave).squeeze(0))
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.monotonic()
    stats=content_training_step(model,optimizer,torch.stack(waves).cuda(),cached['labels'][ids].cuda(),
                                cached['q'][ids].cuda(),cached['valid'][ids].cuda(),diagnostics=True)
    torch.cuda.synchronize(); seconds=time.monotonic()-start
    del stats['log_probs']
    result=dict(status='disposable_one_step_only',batch_size=16,source_real=8,source_fake=8,
        **stats,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        task_trainable_parameters=sum(p.numel() for p in model.task.parameters() if p.requires_grad),
        classifier_parameters=sum(p.numel() for p in model.backend.parameters()),
        adversary_parameters=sum(p.numel() for p in model.content.parameters()),
        formal_training_started=False,checkpoint_saved=False,
        timing_caveat='One batch includes optimizer allocation; not a measured full epoch duration')
    (root/'gpu_smoke.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
