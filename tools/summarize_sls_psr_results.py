"""Read-only audit of all PSR milestones; never selects a target checkpoint."""
import argparse
import json
import math
from pathlib import Path

DOMAINS=("ch","de","uk","es","itw")
COUNTS=dict(ch=18124,de=17696,uk=9709,es=13655,itw=31779)
PAPER_SLS=dict(ch=12.26,de=19.71,uk=21.13,es=9.69,itw=7.46)
PAPER_MAMBA=dict(ch=17.32,de=23.57,uk=11.89,es=15.83,itw=6.71)


def audit(path,arm,seed,milestone):
    rows=json.loads(path.read_text(encoding="utf-8"))
    if len(rows)!=5 or {r["domain"] for r in rows}!=set(DOMAINS):
        raise ValueError(f"Incomplete/duplicate five-domain evaluation: {path}")
    for row in rows:
        if (row["arm"],row["seed"],row["milestone"])!=(arm,seed,milestone):
            raise ValueError(f"Run identity mismatch: {path}")
        if row["samples"]!=COUNTS[row["domain"]] or not math.isfinite(row["eer"]) or not 0<=row["eer"]<=100:
            raise ValueError(f"Invalid count/EER: {path}")
    if len({r["checkpoint_epoch"] for r in rows})!=1:
        raise ValueError("Five domains must use the same source-selected checkpoint")
    return {r["domain"]:r for r in rows}


def summarize(root,baseline,seed):
    stages=[]
    for milestone in (4,8,10):
        path=root/f"eval_psr_epoch{milestone}.json"
        if not path.exists(): continue
        candidate=audit(path,"psr",seed,milestone)
        control=audit(baseline/f"eval_sls_epoch{milestone}.json","sls",seed,milestone)
        values={d:candidate[d]["eer"] for d in DOMAINS}
        avg=sum(values[d] for d in DOMAINS[:4])/4
        matched_avg=sum(control[d]["eer"] for d in DOMAINS[:4])/4
        stages.append(dict(milestone=milestone,checkpoint_epoch=candidate["ch"]["checkpoint_epoch"],
            eer=values,matched_sls_delta={d:values[d]-control[d]["eer"] for d in DOMAINS},
            paper_sls_delta={d:values[d]-PAPER_SLS[d] for d in DOMAINS},
            paper_mamba_delta={d:values[d]-PAPER_MAMBA[d] for d in DOMAINS},
            four_language_mean=avg,worst_language=max(values[d] for d in DOMAINS[:4]),
            ch_de_mean=(values["ch"]+values["de"])/2,
            provisional_candidate_criteria=dict(mean_below_paper_sls=avg<15.6975,
                matched_mean_improvement_at_least_one_pp=matched_avg-avg>=1,
                itw_small_cost=values["itw"]<=8.46,
                no_language_over_paper_sls_plus_three=all(values[d]<=PAPER_SLS[d]+3 for d in DOMAINS[:4]))))
    complete=(root/"completed.json").exists() and len(stages)==3
    return dict(seed=seed,complete=complete,stages=stages,
        final_candidate_pass=complete and all(stages[-1]["provisional_candidate_criteria"].values()),
        stability_verified=False,
        caveat="Only final source-loss-selected epoch10 stage decides expansion; no target-selected stage. Single seed cannot establish robustness.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True); parser.add_argument("--seed",type=int,default=20)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root,args.baseline,args.seed),ensure_ascii=False,indent=2))
