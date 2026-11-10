"""Read results.json -> LaTeX tables, figures, summary.txt (facts only), section5_1.tex.

  python make_outputs.py --results results.json --outdir outputs
"""
import argparse
import json
import math
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"central": "Centralized MLP", "tree": "Decision tree (central, depth 10)", "local": "Local-only",
         "fedavg": "FedAvg", "fedprox": "FedProx", "fedavg_comp": "FedAvg + sparsification/int8",
         "fedper": "FedPer (personalized head, dense)", "ditto": "Ditto", "fedlitep": "Proposed (FedLite-P)"}
ORDER = ["central", "tree", "local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]
CLS = ["benign", "dos", "recon", "bruteforce", "web", "malware"]


def mci(x):
    x = np.array([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def pm(x, nd=3):
    m, h = mci(x)
    return "--" if np.isnan(m) else ("%.*f $\\pm$ %.*f" % (nd, m, nd, 0 if np.isnan(h) else h))


def get(R, m, f):
    return [f(r) for r in R[m]]


def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    return mci(d)


def table_main(R, ds):
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Method & Macro-F1 (client) & Macro-F1 (global) & Accuracy & False-alarm rate & Uplink (B/client/round) & Latency (ms)\\\\",
         "\\midrule"]
    for m in ORDER:
        if m not in R:
            continue
        up = R[m][0]["uplink"]
        L.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (
            NAMES[m], pm(get(R, m, lambda r: r["client"]["f1"])), pm(get(R, m, lambda r: r["glob"]["f1"])),
            pm(get(R, m, lambda r: r["glob"]["acc"])), pm(get(R, m, lambda r: r["glob"]["far"])),
            "--" if up is None else ("0" if up == 0 else str(int(up))), pm(get(R, m, lambda r: r["latency_ms"]), 4)))
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def table_class(R):
    L = ["\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule", "Method & " + " & ".join(CLS) + "\\\\", "\\midrule"]
    for m in ORDER:
        if m in R:
            L.append(NAMES[m] + " & " + " & ".join(pm(get(R, m, lambda r, i=i: r["glob"]["recall"][i]), 2)
                                                  for i in range(6)) + "\\\\")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}"])


def table_cross(R):
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Method & Macro-F1 & Accuracy & False-alarm rate\\\\", "\\midrule"]
    for m in ORDER:
        if m in R:
            g = lambda k: get(R, m, lambda r: r["extra"]["cross"][k])
            L.append("%s & %s & %s & %s\\\\" % (NAMES[m], pm(g("f1")), pm(g("acc")), pm(g("far"))))
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}"])


def fig_alpha(RA, Rmain, path):
    keys = ["0.1", "0.5", "1.0", "inf"]
    keys = [k for k in keys if k in RA]
    plt.figure(figsize=(5.2, 3.6))
    for m in ["local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]:
        mm = [mci(get(RA[k], m, lambda r: r["client"]["f1"])) for k in keys]
        plt.errorbar(range(len(keys)), [a for a, _ in mm], yerr=[0 if np.isnan(b) else b for _, b in mm],
                     marker="o", ms=3, capsize=2, label=NAMES[m].split(" (")[0], lw=2 if m == "fedlitep" else 1)
    if "central" in Rmain:
        plt.axhline(mci(get(Rmain, "central", lambda r: r["client"]["f1"]))[0], ls="--", c="k", lw=.8, label="Centralized MLP")
    plt.xticks(range(len(keys)), [k if k != "inf" else "IID" for k in keys])
    plt.xlabel("Dirichlet concentration $\\alpha$"); plt.ylabel("Client-test macro-F1")
    plt.legend(fontsize=6, ncol=2); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path); plt.close()


def fig_rho(RR, Rmain, path):
    keys = sorted(RR, key=float)
    plt.figure(figsize=(5.2, 3.6))
    for m, st in [("fedlitep", "o-"), ("fedavg_comp", "s--")]:
        xs = [np.mean([r["uplink"] for r in RR[k][m]]) for k in keys]
        ys = [mci(get(RR[k], m, lambda r: r["client"]["f1"]))[0] for k in keys]
        plt.plot(xs, ys, st, label=NAMES[m].split(" (")[0], ms=4)
        for x, y, k in zip(xs, ys, keys):
            plt.annotate(k, (x, y), fontsize=6, xytext=(2, 3), textcoords="offset points")
    for m, mk in [("fedavg", "^"), ("fedper", "v")]:
        plt.plot(Rmain[m][0]["uplink"], mci(get(Rmain, m, lambda r: r["client"]["f1"]))[0], mk, c="k", label=NAMES[m].split(" (")[0] + " (dense)")
    plt.xscale("log"); plt.xlabel("Uplink bytes per client per round"); plt.ylabel("Client-test macro-F1")
    plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path); plt.close()


def fig_rounds(R, path):
    plt.figure(figsize=(5.2, 3.6))
    for m in ["local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep", "central"]:
        if m not in R:
            continue
        A = np.array([r["curve"] for r in R[m]]); mu = A.mean(0)
        plt.plot(range(1, len(mu) + 1), mu, label=NAMES[m].split(" (")[0], lw=2 if m == "fedlitep" else 1,
                 ls="--" if m == "central" else "-")
    plt.xlabel("Communication round (epoch for centralized)"); plt.ylabel("Client-test macro-F1")
    plt.legend(fontsize=6, ncol=2); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path); plt.close()


def rounds_to(R, m, target):
    out = []
    for r in R[m]:
        hit = [i + 1 for i, v in enumerate(r["curve"]) if v >= target]
        out.append(hit[0] if hit else float("nan"))
    return out


def summarize(ds, R, RA, RR, S):
    w = S.append
    w("=" * 70); w("DATASET: %s" % ds)
    n = len(R["fedlitep"]); w("runs (seeds): %d" % n)
    w("\n-- Main setting: mean +- 95% t-interval over runs --")
    for kind, key in [("client-test macro-F1", lambda r: r["client"]["f1"]), ("global-test macro-F1", lambda r: r["glob"]["f1"]),
                      ("global false-alarm rate", lambda r: r["glob"]["far"])]:
        w(kind + ":")
        for m in ORDER:
            if m in R:
                a, h = mci(get(R, m, key)); w("   %-34s %.4f +- %.4f" % (NAMES[m], a, h))
    for kind, key in [("client-test macro-F1", lambda r: r["client"]["f1"]), ("global-test macro-F1", lambda r: r["glob"]["f1"])]:
        best = max([m for m in ORDER if m in R], key=lambda m: mci(get(R, m, key))[0])
        w("Highest mean %s: %s" % (kind, NAMES[best]))
        fed = [m for m in ["local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]]
        bf = max(fed, key=lambda m: mci(get(R, m, key))[0])
        w("Highest mean %s among federated/local methods: %s" % (kind, NAMES[bf]))
    w("\n-- Paired differences, Proposed minus other (same runs), 95% t-interval --")
    for m in ORDER:
        if m == "fedlitep" or m not in R:
            continue
        for kind, key in [("client F1", lambda r: r["client"]["f1"]), ("global F1", lambda r: r["glob"]["f1"])]:
            d, h = paired(get(R, "fedlitep", key), get(R, m, key))
            flag = "n/a" if np.isnan(h) else ("SIGNIFICANT" if abs(d) > h else "not significant")
            w("   vs %-34s %-9s %+.4f +- %.4f  [%s]" % (NAMES[m], kind, d, h, flag))
    w("\n-- Communication (uplink bytes/client/round) --")
    for m in ["fedavg", "fedavg_comp", "fedper", "ditto", "fedlitep"]:
        w("   %-34s %d" % (NAMES[m], R[m][0]["uplink"]))
    u = R["fedlitep"][0]["uplink"]
    w("Uplink reduction of Proposed vs FedAvg: %.1fx; vs FedPer: %.1fx; vs FedAvg+sparsification/int8: %.2fx" % (
        R["fedavg"][0]["uplink"] / u, R["fedper"][0]["uplink"] / u, R["fedavg_comp"][0]["uplink"] / u))
    w("\n-- Ablation (client-test macro-F1) --")
    for m in ["fedavg", "fedavg_comp", "fedper", "fedlitep"]:
        a, h = mci(get(R, m, lambda r: r["client"]["f1"])); w("   %-34s %.4f +- %.4f" % (NAMES[m], a, h))
    d, h = paired(get(R, "fedper", lambda r: r["client"]["f1"]), get(R, "fedavg", lambda r: r["client"]["f1"]))
    w("   effect of personalization (FedPer - FedAvg): %+.4f +- %.4f" % (d, h))
    d, h = paired(get(R, "fedlitep", lambda r: r["client"]["f1"]), get(R, "fedper", lambda r: r["client"]["f1"]))
    w("   effect of compression on personalized model (FedLite-P - FedPer): %+.4f +- %.4f" % (d, h))
    w("\n-- Per-class recall (global test): lowest classes and Proposed minus Centralized --")
    for m in ORDER:
        if m in R:
            rec = np.nanmean(np.array([r["glob"]["recall"] for r in R[m]], float), axis=0)
            w("   %-34s %s" % (NAMES[m], "  ".join("%s=%.3f" % (c, v) for c, v in zip(CLS, rec))))
    for i, c in enumerate(CLS):
        d, h = paired(get(R, "fedlitep", lambda r: r["glob"]["recall"][i]), get(R, "central", lambda r: r["glob"]["recall"][i]))
        if not np.isnan(d):
            w("   recall %-10s Proposed - Central: %+.3f +- %.3f" % (c, d, h))
    if RA:
        w("\n-- Heterogeneity (client-test macro-F1 vs alpha) --")
        for m in ["fedavg", "fedper", "ditto", "fedlitep", "local"]:
            w("   %-34s %s" % (NAMES[m], "  ".join("a=%s:%.3f" % (k, mci(get(RA[k], m, lambda r: r["client"]["f1"]))[0]) for k in RA)))
    if RR:
        w("\n-- Sparsity (client-test macro-F1 vs rho; uplink bytes) --")
        for m in ["fedlitep", "fedavg_comp"]:
            w("   %-34s %s" % (NAMES[m], "  ".join("rho=%s:%.3f(%dB)" % (k, mci(get(RR[k], m, lambda r: r["client"]["f1"]))[0],
                                                                         RR[k][m][0]["uplink"]) for k in sorted(RR, key=float))))
    cen = np.mean(get(R, "central", lambda r: r["client"]["f1"]))
    tgt = 0.95 * cen
    w("\n-- Rounds to reach 95%% of the centralized client-test F1 (target %.3f), mean over runs that reach it --" % tgt)
    for m in ["fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]:
        rr = rounds_to(R, m, tgt); ok = [x for x in rr if not np.isnan(x)]
        w("   %-34s reached in %d/%d runs; mean round %s" % (NAMES[m], len(ok), len(rr), "%.1f" % np.mean(ok) if ok else "n/a"))


def section51(meta):
    m = meta
    cc = ", ".join("%s: %d" % (k, v) for k, v in m.get("class_counts_used", {}).items())
    return ("\\paragraph{%s.} Files: %s. After mapping labels to the six classes with keyword rules, we removed %s rows with "
            "unmapped labels (e.g.\\ labels outside the six classes), %s rows with missing or infinite values and %s duplicate rows; "
            "classes with fewer than 30 flows (%s) were dropped and at most %s flows per class were kept. "
            "The final data have %s flows, %s features and the class counts (%s).\n") % (
        m["name"], ", ".join(os.path.basename(f) for f in m["files"]), m.get("rows_removed_unmapped", "n/a"),
        m.get("rows_removed_nan_inf", "n/a"), m.get("rows_removed_duplicates", "n/a"),
        ", ".join(m.get("classes_dropped_too_rare", [])) or "none", m.get("max_per_class", "n/a"),
        m["n_used"], m["n_features"], cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json"); ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()
    res = json.load(open(a.results)); os.makedirs(a.outdir, exist_ok=True)
    S = ["Facts below are computed from %s; write the paper ONLY from these numbers." % a.results,
         "Config: %s" % json.dumps(res["config"])]
    for ds, R in res["main"].items():
        RA, RR = res["alpha"].get(ds, {}), res["rho"].get(ds, {})
        open("%s/table2_%s.tex" % (a.outdir, ds), "w").write(table_main(R, ds))
        open("%s/table3_%s.tex" % (a.outdir, ds), "w").write(table_class(R))
        if RA:
            fig_alpha(RA, R, "%s/fig1_alpha_%s.pdf" % (a.outdir, ds))
        if RR:
            fig_rho(RR, R, "%s/fig2_rho_%s.pdf" % (a.outdir, ds))
        fig_rounds(R, "%s/fig3_rounds_%s.pdf" % (a.outdir, ds))
        open("%s/section5_1_%s.tex" % (a.outdir, ds), "w").write(section51(res["meta"][ds]))
        summarize(ds, R, RA, RR, S)
    for k, v in res["cross"].items():
        R = v["results"]; tag = k.replace("->", "_to_")
        open("%s/table4_%s.tex" % (a.outdir, tag), "w").write(table_cross(R))
        S.append("=" * 70); S.append("CROSS-DATASET %s (%d common features)" % (k, v["n_common_features"]))
        for m in ORDER:
            if m in R:
                f, h = mci(get(R, m, lambda r: r["extra"]["cross"]["f1"]))
                fa, ha = mci(get(R, m, lambda r: r["extra"]["cross"]["far"]))
                S.append("   %-34s macroF1 %.4f +- %.4f   FAR %.4f +- %.4f" % (NAMES[m], f, h, fa, ha))
        for m in ["fedavg", "fedper", "central"]:
            d, h = paired(get(R, "fedlitep", lambda r: r["extra"]["cross"]["f1"]), get(R, m, lambda r: r["extra"]["cross"]["f1"]))
            S.append("   Proposed - %s (cross F1): %+.4f +- %.4f" % (NAMES[m], d, h))
    open("%s/summary.txt" % a.outdir, "w").write("\n".join(S) + "\n")
    print("\n".join(S[:60]))


if __name__ == "__main__":
    main()
