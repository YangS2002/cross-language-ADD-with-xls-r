"""Summarize completed diagnostic only; no fits, encoder runs or new selection."""
import json
from pathlib import Path
import numpy as np
from tools.evaluate_four_domains_shared_xlsr import eer, digest
from tools.run_uk_representation_diagnostic import OUT, STAGES

def main():
    state=json.loads((OUT/'status.json').read_text())
    if state['status']!='complete': raise RuntimeError('Diagnostic incomplete')
    results=json.loads((OUT/'results.json').read_text())
    manifest=json.loads((OUT/'split_manifest.json').read_text())
    predictions=json.loads((OUT/'predictions.json').read_text())
    full=[]
    for model in ('original','finetuned'):
        with np.load(OUT/f'features_{model}_uk.npz') as f:
            full.append(dict(model=model,domain='uk',n=len(f['y']),eer=eer(f['y'],f['native'])[0]))
    uk=[r for r in manifest['uk']['rows'] if r['split']=='test']
    y=np.array([r['label'] for r in uk]);groups=np.array([r['group'] for r in uk])
    bygroup=[]
    for model in ('original','finetuned'):
        for stage in STAGES:
            for trained in ('a','uk'):
                score=np.array(predictions['|'.join(('linear_probe',model,stage,trained,'uk','test'))])
                for group in np.unique(groups):
                    ix=np.flatnonzero(groups==group)
                    if len(np.unique(y[ix]))==2:
                        bygroup.append(dict(model=model,stage=stage,trained_on=trained,group=group,
                                            n=len(ix),real=int(y[ix].sum()),eer=eer(y[ix],score[ix])[0]))
    summary=dict(native_full_uk=full,uk_test=[r for r in results['results']
        if r['evaluated_on']=='uk' and r['split']=='test'],by_group=bygroup,
        all_probes_converged=all(r['converged'] for r in results['results']),
        analysis_script_sha256=digest(Path(__file__).with_name('run_uk_representation_diagnostic.py')),
        original_training_checkpoint=json.loads((OUT/'finetuned_checkpoint.json').read_text()),
        note='UK-supervised probe performance is diagnostic, not zero-target detection; test has three book groups')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(native_full_uk=full,all_probes_converged=summary['all_probes_converged'],
        classifier_embedding_groups=[r for r in bygroup if r['stage']=='classifier_embedding']),indent=2))

if __name__=='__main__':main()
