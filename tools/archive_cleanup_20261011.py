"""One-off manifest-based cleanup of historical experiment artifacts, not datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import zipfile
import psutil

ROOT = Path('/root/rivermind-data/cross-language-ADD-with-xls-r')
OUT = ROOT / 'experiments/handoff_2026-10-11'
PLAN = OUT / 'remote_cleanup_plan.json'
RESULT = OUT / 'remote_cleanup_result.json'
ARCHIVE = OUT / 'remote_records_and_code.zip'


def dump(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def no_experiments():
    active = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        if p.pid == os.getpid():
            continue
        try:
            args = p.info['cmdline'] or []
            # Do not collect unrelated service commands, which may contain credentials.
            modules = [a for a in args if a.startswith(('tools.train_', 'tools.evaluate_',
                       'tools.launch_sls', 'tools.run_m5_deterministic'))]
            scripts = [a for a in args if a in ('train.py', str(ROOT / 'train.py'))]
            if modules or scripts:
                active.append(dict(pid=p.pid, experiment=modules + scripts))
        except psutil.Error:
            pass
    if active:
        raise RuntimeError(f'Active experiment processes: {active}')


def validate(path):
    if ROOT.resolve() != ROOT or not path.is_relative_to(ROOT / 'models'):
        raise RuntimeError(f'Outside explicit models root: {path}')
    if path.resolve() != path or any(p.is_symlink() for p in [path, *path.parents]):
        raise RuntimeError(f'Symlink or redirected path: {path}')
    if 'pretrained' in path.relative_to(ROOT / 'models').parts or path.name == 'warmup.pth':
        raise RuntimeError(f'Protected artifact: {path}')
    if path.suffix.lower() not in ('.pth', '.pt', '.npz', '.npy'):
        raise RuntimeError(f'Not an authorized generated binary: {path}')
    if not path.is_file():
        raise RuntimeError(f'Not a regular file: {path}')


def make_plan():
    no_experiments()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() or RESULT.exists():
        raise RuntimeError('Existing cleanup record; refuse to silently replace it')
    protected, candidates = [], []
    for path in sorted((ROOT / 'models').rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if 'pretrained' in path.relative_to(ROOT / 'models').parts or path.name == 'warmup.pth':
            protected.append(dict(path=rel, **identity(path)))
        elif path.suffix.lower() in ('.pth', '.pt', '.npz', '.npy'):
            validate(path)
            candidates.append(dict(path=rel, **identity(path)))
    records = []
    for folder in ('experiments', 'models', 'tools', 'src'):
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or path.is_symlink() or OUT in path.parents:
                continue
            rel = path.relative_to(ROOT)
            if 'pretrained' in rel.parts or '__pycache__' in rel.parts:
                continue
            if path.suffix.lower() not in ('.md', '.json', '.csv', '.log', '.yaml', '.yml', '.py', '.txt', '.svg', '.html'):
                continue
            if path.name.startswith('scores') or path.name in ('predictions.json', 'source_energy_samples.csv'):
                continue  # Retained remotely, not needed for the small aggregate handoff archive.
            records.append(path)
    records += [ROOT / p for p in ('train.py', 'data.py', 'requirements.txt', '.gitignore') if (ROOT / p).is_file()]
    manifest = []
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(records)):
            rel = str(path.relative_to(ROOT))
            z.write(path, rel)
            manifest.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest(path)))
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
        z.writestr('REMOTE_TRACKED_WORKTREE_DIFF.patch', diff)
        z.writestr('ARCHIVE_FILES_SHA256.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed')
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                authorization='User requested cleanup of historical training artifacts; preserve reports/data/code/pretrained/warmup',
                disk_before=dict(zip(('total', 'used', 'free'), shutil.disk_usage(ROOT))),
                candidates=candidates, total_candidate_bytes=sum(p['bytes'] for p in candidates),
                protected=protected, archive=str(ARCHIVE.relative_to(ROOT)),
                archive_sha256=digest(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
                archived_files=len(manifest), per_sample_scores_retained=True)
    dump(PLAN, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def apply_plan():
    no_experiments()
    if RESULT.exists():
        raise RuntimeError('Cleanup already has a result; refuse implicit retry')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if plan['root'] != str(ROOT) or digest(ARCHIVE) != plan['archive_sha256']:
        raise RuntimeError('Root or archive signature changed')
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed')
    for entry in plan['protected']:
        if identity(ROOT / entry['path']) != {k: entry[k] for k in ('bytes', 'mtime_ns', 'inode')}:
            raise RuntimeError(f'Protected artifact changed: {entry["path"]}')
    for entry in plan['candidates']:
        path = ROOT / entry['path']
        validate(path)
        if identity(path) != {k: entry[k] for k in ('bytes', 'mtime_ns', 'inode')}:
            raise RuntimeError(f'Candidate identity changed: {path}')
    result = dict(status='deleting', started_at=datetime.now(timezone.utc).isoformat(),
                  deleted=[], deleted_bytes=0, recovery='Deleted binaries are not in the records archive; not recoverable here')
    dump(RESULT, result)
    try:
        for entry in plan['candidates']:
            path = ROOT / entry['path']
            validate(path)
            path.unlink()  # Only individually validated files, never a recursive directory deletion.
            result['deleted'].append(entry)
            result['deleted_bytes'] += entry['bytes']
            dump(RESULT, result)
        for entry in plan['protected']:
            if identity(ROOT / entry['path']) != {k: entry[k] for k in ('bytes', 'mtime_ns', 'inode')}:
                raise RuntimeError(f'Protected artifact changed after cleanup: {entry["path"]}')
        result.update(status='complete', ended_at=datetime.now(timezone.utc).isoformat(),
                      disk_after=dict(zip(('total', 'used', 'free'), shutil.disk_usage(ROOT))),
                      protected_verified=True, remaining_candidate_files=[e['path'] for e in plan['candidates'] if (ROOT / e['path']).exists()])
    except Exception as exc:
        result.update(status='partial_failure', error=repr(exc))
        raise
    finally:
        dump(RESULT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plan', 'apply'))
    mode = parser.parse_args().mode
    make_plan() if mode == 'plan' else apply_plan()
