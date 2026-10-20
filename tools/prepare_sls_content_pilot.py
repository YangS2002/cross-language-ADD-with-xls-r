"""Bounded source-only pilot; not a detector training run or EER experiment."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
import torchaudio
from torch.nn import functional as F
from data import pad
from tools.sls_content_adversarial_model import FrozenContentReadout


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(ids):
    return [v for i, v in enumerate(ids) if v != 0 and (i == 0 or v != ids[i-1])]


def edit(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+int(x != y)))
        previous = row
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--all-genuine', action='store_true')
    a = parser.parse_args()
    root = Path(a.output).resolve()
    if not root.is_relative_to(Path('/root/rivermind-data')):
        raise ValueError('Persistent output required')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'pilot_results.json').exists():
        raise FileExistsError('Existing pilot must not be overwritten')
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > a.seconds:
            raise TimeoutError('Pilot preparation budget exhausted; do not restart automatically')
    torch.set_num_threads(4); torch.manual_seed(20)
    labels_file = Path('/root/rivermind-data/data/exp/ASVspoof2019.LA.cm.train.trn.txt')
    rows = [line.split() for line in labels_file.read_text().splitlines() if line.strip()]
    if len(rows) != 25380 or sum(r[-1]=='bonafide' for r in rows) != 2580:
        raise ValueError('Source protocol mismatch')
    def heldout(speaker):
        return int(hashlib.sha256(speaker.encode()).hexdigest(),16)%5 == 0
    # Fixed speaker-separated readout pilot. Never changes detector training sampling.
    def speaker_round_robin(candidates, count):
        groups={speaker:sorted([r for r in candidates if r[0]==speaker],
            key=lambda r:hashlib.sha256(r[1].encode()).hexdigest())
            for speaker in sorted({r[0] for r in candidates})}
        chosen=[]; offset=0
        while len(chosen)<count:
            added=False
            for group in groups.values():
                if offset<len(group) and len(chosen)<count:
                    chosen.append(group[offset]); added=True
            if not added: break
            offset+=1
        return chosen
    fit_candidates=[r for r in rows if r[-1]=='bonafide' and not heldout(r[0])]
    check_candidates=[r for r in rows if r[-1]=='bonafide' and heldout(r[0])]
    fit = speaker_round_robin(fit_candidates,len(fit_candidates) if a.all_genuine else 96)
    check = speaker_round_robin(check_candidates,len(check_candidates) if a.all_genuine else 32)
    fake = sum(([r for r in rows if r[-2]==attack][:4]
                 for attack in ('A01','A02','A03','A04','A05','A06')), [])
    if (not a.all_genuine and (len(fit)!=96 or len(check)!=32)) or len(fake)!=24:
        raise ValueError('Pilot coverage mismatch')
    nfit=len(fit); nreal=nfit+len(check)
    if a.all_genuine and nreal!=2580: raise ValueError('Genuine source count mismatch')
    selected = fit+check+fake
    assert not ({r[0] for r in fit} & {r[0] for r in check})
    audio_root = Path('/root/rivermind-data/data/asv2019_la_train')
    waves=[]; rates={}
    for r in selected:
        wave,sr = torchaudio.load(str(audio_root/r[1]))
        waves.append(pad(wave).squeeze(0)); rates[sr]=rates.get(sr,0)+1
    waves=torch.stack(waves)
    # Same direct load+pad as detector. No teacher-specific resampling.
    asr_dir=Path('/root/rivermind-data/models/content_asr')
    asset=asr_dir/'wav2vec2_fairseq_base_ls960_asr_ls960.pth'
    if digest(asset)!='488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d':
        raise ValueError('Downloaded asset changed')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    alphabet=bundle.get_labels(); letters=[i for i,v in enumerate(alphabet) if v.isalpha()]
    if len(letters)!=26:
        raise ValueError('Unexpected ASR alphabet')
    asr=bundle.get_model(dl_kwargs={'model_dir':str(asr_dir)}).cuda().eval()
    transcripts=[]
    with torch.no_grad():
        for i in range(0,len(waves),8):
            budget()
            # FP32 teacher decoding to avoid AMP-dependent pseudo labels.
            emission,_=asr(waves[i:i+8].cuda())
            transcripts.extend(decode(ids) for ids in emission.argmax(-1).cpu().tolist())
    print('ASR_DONE',len(transcripts),time.monotonic()-start,flush=True)
    del asr; gc.collect(); torch.cuda.empty_cache()
    checkpoint=Path('/root/rivermind-data/cross-language-ADD-with-xls-r/models/pretrained/xlsr2_300m.pt')
    warm_path=Path('/root/rivermind-data/cross-language-ADD-with-xls-r/models/SLS_conditional_residual_seed20/warmup.pth')
    if digest(checkpoint)!='b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9':
        raise ValueError('Original XLS-R changed')
    if digest(warm_path)!='32454f00414be2172ca7d84bfa631c3a8b551f45d415816354b0597645904742':
        raise ValueError('Shared warmup changed')
    warm=torch.load(warm_path,map_location='cpu',weights_only=False)
    gate={k.removeprefix('fc0.'):v for k,v in warm['backend'].items() if k.startswith('fc0.')}
    teacher=FrozenContentReadout(checkpoint,gate).cuda().eval()
    features=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(waves),8):
            budget(); features.append(teacher.frames(waves[i:i+8].cuda()).cpu())
    frames=torch.cat(features).cuda()
    del teacher.reference; gc.collect(); torch.cuda.empty_cache()
    readout=teacher.readout
    frame_count=frames.shape[1]
    usable=[bool(t) and len(t)+sum(x==y for x,y in zip(t,t[1:]))<=frame_count for t in transcripts]
    train_ids=[i for i in range(nfit) if usable[i]]
    val_ids=[i for i in range(nfit,nreal) if usable[i]]
    if len(train_ids)<.8*nfit or len(val_ids)<.75*len(check):
        raise ValueError('Too few valid CTC sequences')
    lengths=torch.tensor([len(t) for t in transcripts],device='cuda',dtype=torch.long)
    def ctc(head, ids, mapping):
        mapped=[mapping[i] for i in ids]
        targets=torch.tensor(sum((transcripts[i] for i in mapped),[]),device='cuda',dtype=torch.long)
        logits=head(frames[ids]).log_softmax(-1).transpose(0,1)
        return F.ctc_loss(logits,targets,torch.full((len(ids),),frame_count,device='cuda',dtype=torch.long),
                          lengths[mapped],blank=0,reduction='mean',zero_infinity=False)
    identity=list(range(len(selected)))
    shifted=identity.copy()
    for n,i in enumerate(train_ids): shifted[i]=train_ids[(n+1)%len(train_ids)]
    initial={k:v.detach().clone() for k,v in readout.state_dict().items()}
    histories={}; states={}
    for arm,mapping in [('content',identity),('shuffled',shifted)]:
        readout.load_state_dict(initial)
        opt=torch.optim.Adam(readout.parameters(),lr=1e-3)
        best=float('inf'); history=[]
        for epoch in range(100):
            budget(); total=0.
            for begin in range(0,len(train_ids),16):
                ids=train_ids[begin:begin+16]; opt.zero_grad(set_to_none=True)
                loss=ctc(readout,ids,mapping)
                if not torch.isfinite(loss): raise FloatingPointError('CTC readout diverged')
                loss.backward(); torch.nn.utils.clip_grad_norm_(readout.parameters(),5.,error_if_nonfinite=True)
                opt.step(); total+=float(loss)*len(ids)
            # Checkpoint chosen only by complete source readout fit loss.
            with torch.no_grad():
                fit_loss=sum(float(ctc(readout,train_ids[j:j+16],mapping))*len(train_ids[j:j+16])
                    for j in range(0,len(train_ids),16))/len(train_ids)
            if fit_loss<best:
                best=fit_loss; states[arm]={k:v.detach().cpu().clone() for k,v in readout.state_dict().items()}
            if epoch in (0,9,49,99):
                history.append(dict(epoch=epoch+1,fit_ctc=fit_loss))
                print('READOUT',arm,epoch+1,fit_loss,flush=True)
        histories[arm]=history
    letter_index={v:i for i,v in enumerate(letters)}
    def histogram(tokens):
        result=torch.zeros(26,device='cuda')
        for t in tokens:
            if t in letter_index: result[letter_index[t]]+=1
        return result/result.sum().clamp_min(1)
    reference=torch.stack([histogram(t) for t in transcripts])
    const=reference[train_ids].mean(0).clamp_min(1e-8)
    most_common=max((transcripts[i] for i in train_ids),key=lambda t: sum(transcripts[j]==t for j in train_ids))
    constant_cer=sum(edit(most_common,transcripts[i]) for i in val_ids)/sum(len(transcripts[i]) for i in val_ids)
    constant_kl=float(F.kl_div(const.log().expand(len(val_ids),-1),reference[val_ids],reduction='batchmean'))
    metrics={}
    for arm in ('content','shuffled'):
        readout.load_state_dict(states[arm]); readout.eval()
        with torch.no_grad():
            logits=readout(frames)
            decoded=[decode(ids) for ids in logits.argmax(-1).cpu().tolist()]
            q,valid,mass=teacher.content_distribution(frames,letters)
            cer=sum(edit(decoded[i],transcripts[i]) for i in val_ids)/sum(len(transcripts[i]) for i in val_ids)
            kl=float(F.kl_div(q[val_ids].clamp_min(1e-8).log(),reference[val_ids],reduction='batchmean'))
            metrics[arm]=dict(holdout_pseudo_cer=cer,holdout_histogram_kl=kl,
                holdout_ctc=float(ctc(readout,val_ids,identity)),valid_count=int(valid.sum()),
                nonblank_letter_mass=float(mass.mean()), fake_histogram_kl_by_attack={
                    attack:float(F.kl_div(q[nreal+i*4:nreal+4+i*4].clamp_min(1e-8).log(),
                        reference[nreal+i*4:nreal+4+i*4],reduction='batchmean'))
                    for i,attack in enumerate(('A01','A02','A03','A04','A05','A06'))})
    passed=(metrics['content']['holdout_pseudo_cer']<min(1.,constant_cer,metrics['shuffled']['holdout_pseudo_cer'])
        and metrics['content']['holdout_histogram_kl']<min(constant_kl,metrics['shuffled']['holdout_histogram_kl']))
    result=dict(status='source_readout_only',all_genuine=a.all_genuine,selection='speaker_round_robin_v2',sample_count=len(selected),fit_real=len(train_ids),holdout_real=len(val_ids),
        fake_audit=24,sample_rates=rates,source_protocol_sha256=digest(labels_file),
        fit_speakers=sorted({r[0] for r in fit}),holdout_speakers=sorted({r[0] for r in check}),
        histories=histories,metrics=metrics,constant_histogram_kl=constant_kl,constant_transcript_cer=constant_cer,
        reference_pilot_passed=passed,seconds=time.monotonic()-start,
        limitations='Pseudo-ASR labels; no verified human transcripts or target EER. '
            'Passing readout checks is not evidence of cross-language improvement; fake audit only four per attack.')
    (root/'pilot_results.json').write_text(json.dumps(result,indent=2))
    (root/'pilot_transcripts.json').write_text(json.dumps([
        dict(file=r[1],speaker=r[0],attack=r[-2],split='fit' if i<nfit else 'check' if i<nreal else 'fake_audit',
             transcript=''.join(alphabet[t] for t in transcripts[i]),ctc_usable=usable[i])
        for i,r in enumerate(selected)],indent=2))
    torch.save(dict(readout=states['content'],gate=gate,alphabet=alphabet,
        approved_for_detector_training=False),root/'pilot_readout.pth')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
