"""CPU-only summaries of the completed endpoint diagnostic; no fitting."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "models/UK_query_localization_seed20"
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if json.loads((root / "status.json").read_text())["state"] != "completed":
        raise RuntimeError("Diagnostic incomplete")
    with (root / "group_results.csv").open(encoding="utf-8") as f:
        groups = list(csv.DictReader(f))
    with (root / "cross_group_auc.csv").open(encoding="utf-8") as f:
        matrix = list(csv.DictReader(f))
    with (root / "layer_diagnostics.csv").open(encoding="utf-8") as f:
        layers = list(csv.DictReader(f))
    native = {r["group"]: r for r in groups if r["variant"] == "native"}
    comparisons = {}
    for variant in result["uk"]:
        if variant == "seconds":
            continue
        selected = [r for r in groups if r["variant"] == variant]
        changed = [dict(group=r["group"], eer=float(r["eer"]), samples=int(r["samples"]),
                        auc=float(r["auc"]), delta_vs_native=float(r["eer"])-float(native[r["group"]]["eer"]))
                   for r in selected]
        comparisons[variant] = dict(
            improved_groups=sum(r["delta_vs_native"] < -1e-9 for r in changed),
            worsened_groups=sum(r["delta_vs_native"] > 1e-9 for r in changed),
            sample_weighted_group_eer=sum(r["eer"]*r["samples"] for r in changed)/sum(r["samples"] for r in changed),
            groups=sorted(changed, key=lambda r: r["eer"], reverse=True))
    decomposition = {}
    for variant in ("native", "original_warm_backend"):
        decomposition[variant] = {}
        for kind in ("within_group", "cross_group", "all"):
            selected = [r for r in matrix if r["variant"] == variant and
                        (kind == "all" or ((r["real_group"] == r["fake_group"]) == (kind == "within_group")))]
            pairs = sum(int(r["pairs"]) for r in selected)
            auc = sum(float(r["auc"])*int(r["pairs"]) for r in selected)/pairs
            decomposition[variant][kind] = dict(auc=auc, pairs=pairs, pair_error_rate=1-auc)
    layer_summary = {}
    for domain in ("a", "uk", "itw"):
        layer_summary[domain] = {}
        for label in (0, 1):
            selected = [r for r in layers if r["domain"] == domain and int(r["label"]) == label]
            layer_summary[domain][str(label)] = {}
            for name in ("attention_kl", "entropy", "original_entropy", "key_relative_l2", "value_relative_l2"):
                layer_summary[domain][str(label)][name] = dict(
                    mean_over_layers=sum(float(r[name]) for r in selected)/24,
                    mean_excluding_layer23=sum(float(r[name]) for r in selected if int(r["layer"]) != 23)/23,
                    layer8=float(selected[7][name]), layer16=float(selected[15][name]),
                    layer23=float(selected[22][name]),
                    layer24=float(selected[23][name]))
    summary = dict(domain_results=result, book_comparisons=comparisons,
                   auc_pair_decomposition=decomposition, layer_summary=layer_summary,
                   interpretation_caveat="Different group thresholds: weighted group EER is not global EER")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(domain_results=result, auc_pair_decomposition=decomposition,
                          layer_summary=layer_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
