"""User-authorized removal of old experiment weights/caches, not datasets/code."""
import json
import shutil
import time
from pathlib import Path
import psutil

REPO = Path('/root/rivermind-data/cross-language-ADD-with-xls-r')
ROOTS = [
    'M1_heterogeneous_multi_order_seed20_10ep', 'M1a_local_transition_preservation_seed20_10ep',
    'M2_dual_transition_second_order_residual_seed20_10ep',
    'M2_dual_transition_second_order_residual_seed1001_10ep',
    'M3_dual_transition_low_high_consistency_seed20_10ep',
    'M3_dual_transition_low_high_consistency_seed1001_10ep',
    'M4_dual_transition_low_high_vector_adapter_seed20_10ep',
    'M4_dual_transition_low_high_vector_adapter_seed1001_10ep',
    'M4b_dual_transition_low_high_vector_adapter_scale002_seed1001_10ep',
    'M5_deterministic_transition', 'E2_raw_vs_transition', 'E3_dual_xlsr_reference',
    'RP_temporal_seed20', 'SLS_reference_seed20', 'SLS_conditional_residual_seed20',
    'SLS_content_adversarial_seed20', 'SLS_dependency_control_seed20',
    'SLS_subspace_restoration_seed20', 'TL_xlsr_transition',
    'TL_dual_transition_consistency', 'TL_dual_transition_consistency_seed20',
    'TL_dual_transition_consistency_seed1001_10ep',
]

def main():
    if REPO.resolve() != REPO or not (REPO / 'tools').is_dir():
        raise RuntimeError('Unexpected repository target')
    for p in psutil.process_iter(['pid', 'cmdline']):
        args = p.info['cmdline'] or []
        if any(x.startswith('tools.train_') for x in args):
            raise RuntimeError(f'Training still active: PID {p.pid}')
    models = REPO / 'models'
    keep = {(models / 'SLS_conditional_residual_seed20/sls/best.pth').resolve(),
            (models / 'SLS_conditional_residual_seed20/warmup.pth').resolve()}
    for path in keep:
        if not path.is_file(): raise FileNotFoundError(path)
    selected = []
    for name in ROOTS:
        folder = models / name
        if not folder.exists(): continue
        if folder.is_symlink() or folder.resolve().parent != models.resolve():
            raise RuntimeError(f'Unsafe experiment directory: {folder}')
        for path in folder.rglob('*'):
            if path.is_symlink():
                # Remove the old experiment's pointer, never follow/delete its target.
                absolute=path.absolute()
                if not absolute.is_relative_to(folder): raise RuntimeError(absolute)
                selected.append(dict(path=str(absolute),bytes=path.lstat().st_size,kind='symlink'))
                continue
            if not path.is_file() or path.suffix not in {'.pth','.pt','.npy','.npz','.ckpt','.safetensors','.pkl'}:
                continue
            absolute = path.resolve()
            if not absolute.is_relative_to(folder.resolve()): raise RuntimeError(absolute)
            if absolute in keep: continue
            selected.append(dict(path=str(absolute),bytes=path.stat().st_size))
    report = REPO / 'experiments/OLD_ARTIFACT_CLEANUP_2026-10-08.json'
    if report.exists(): raise FileExistsError('Cleanup already recorded; do not replay')
    manifest = dict(user_authorized=True,time=time.time(),targets=selected,
        keep=[str(p) for p in sorted(keep)],bytes=sum(r['bytes'] for r in selected),
        free_before=shutil.disk_usage(models).free,status='planned',
        scope='Old experiment weights and feature caches only; small records, code, data and pretrained assets preserved',
        recovery='Deleted artifacts are not recoverable from this operation')
    report.write_text(json.dumps(manifest,indent=2))
    deleted=[]
    try:
        for row in selected:
            path=Path(row['path'])
            stat=path.lstat() if row.get('kind')=='symlink' else path.stat()
            if stat.st_size != row['bytes']: raise RuntimeError('Target changed')
            path.unlink(); deleted.append(row['path'])
        manifest.update(status='complete',deleted=deleted,free_after=shutil.disk_usage(models).free)
    finally:
        if manifest['status']!='complete': manifest.update(status='partial',deleted=deleted)
        report.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'targets','deleted'}},indent=2))

if __name__=='__main__': main()
