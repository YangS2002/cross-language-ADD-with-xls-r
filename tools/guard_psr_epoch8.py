"""User-requested epoch8 futility stop for the already-running exact PID."""
import argparse
import json
import os
import signal
import time
from pathlib import Path
from tools.summarize_sls_psr_results import audit,DOMAINS

def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args(); proc=Path(f'/proc/{args.pid}')
    expected=b'tools.train_sls_subspace_restoration'
    if expected not in (proc/'cmdline').read_bytes(): raise RuntimeError('Wrong target process')
    start=(proc/'stat').read_text().split()[21]
    def live():
        try: return (proc/'stat').read_text().split()[21]==start and expected in (proc/'cmdline').read_bytes()
        except FileNotFoundError: return False
    write(args.root/'user_epoch8_override.json',dict(pid=args.pid,start_ticks=start,
        instruction='User requests stop after full epoch8 evaluation if ineffective; overrides prior no-stop policy',
        rule='four-language mean must beat matched SLS epoch8 and ITW <= 8.46%; otherwise stop',
        recorded_after_epoch4_observed=True))
    while live():
        path=args.root/'eval_psr_epoch8.json'
        if not path.exists(): time.sleep(2); continue
        candidate=audit(path,'psr',20,8)
        baseline=audit(args.baseline/'eval_sls_epoch8.json','sls',20,8)
        avg=sum(candidate[d]['eer'] for d in DOMAINS[:4])/4
        baseline_avg=sum(baseline[d]['eer'] for d in DOMAINS[:4])/4
        effective=avg<baseline_avg and candidate['itw']['eer']<=8.46
        record=dict(epoch=8,pid=args.pid,four_language_mean=avg,matched_sls_mean=baseline_avg,
            itw=candidate['itw']['eer'],effective=effective,
            action='continue fixed epoch10' if effective else 'terminate user-authorized ineffective run',
            candidate_eer={d:candidate[d]['eer'] for d in DOMAINS},
            not_a_source_checkpoint_selection_rule=True)
        if effective:
            write(args.root/'user_epoch8_decision.json',record); print(json.dumps(record),flush=True); return
        # Allow the five CSV appends to finish before stopping; no model changes.
        time.sleep(1)
        if not live(): raise RuntimeError('Target exited before authorized stop')
        os.kill(args.pid,signal.SIGSTOP)
        try:
            status=json.loads((args.root/'status.json').read_text())
            record['observed_status_before_stop']=status
            write(args.root/'user_epoch8_decision.json',record)
        except BaseException:
            os.kill(args.pid,signal.SIGCONT); raise
        os.kill(args.pid,signal.SIGTERM); os.kill(args.pid,signal.SIGCONT)
        deadline=time.monotonic()+30
        while live() and time.monotonic()<deadline: time.sleep(1)
        if live(): raise RuntimeError('SIGTERM did not terminate; no automatic escalation')
        write(args.root/'status.json',dict(status='stopped_by_user_epoch8_rule',epoch=8,
            decision='user_epoch8_decision.json',partial_next_epoch_possible=status.get('epoch',8)>8))
        print(json.dumps(record),flush=True); return
    print('Target exited before epoch8; no restart',flush=True)

if __name__=='__main__': main()
