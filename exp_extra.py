"""Extra experiments: (A) FedLite-PG (additional aggregated global head), (B) label-flipping poisoning with mean / trimmed-mean aggregation.
  python exp_extra.py --data-dir /path/to/InSDN_DatasetCSV --out results/extra_results.json
"""
import argparse, copy, json, os, time
import numpy as np
import data as D, fl
from run_experiments import build_fd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--out", default="extra_results.json")
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
cfg = dict(K=20, rounds=100, E=1, lr=1e-3, bs=128, mu=0.01, ditto_lambda=0.1, cw_exp=0.5, cw_cap=10.0, tree_depth=10,
           n_client_eval=500, rho=0.1, lam=0.5)
ds = D.load_csv("InSDN", [os.path.join(a.data_dir, f) for f in ["Normal_data.csv", "OVS.csv", "metasploitable-2.csv"]])
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
t0 = time.time()

def poisoned(fd, frac, seed):
    fd2 = copy.copy(fd); fd2["ytr"] = [y.copy() for y in fd["ytr"]]
    m = int(round(frac * fd["K"]))
    mal = np.random.default_rng(seed * 31 + 5).choice(fd["K"], m, replace=False) if m else []
    fd2["honest"] = [k for k in range(fd["K"]) if k not in set(int(x) for x in mal)]
    for k in mal:
        fd2["ytr"][k] = np.where(fd2["ytr"][k] > 0, 0, fd2["ytr"][k])  # attack flows labelled benign
    return fd2

def run(key, fn):
    if key in out: return
    out[key] = fn(); json.dump(out, open(a.out, "w")); print(key, int(time.time() - t0), flush=True)

for s in range(a.seeds):
    for al in [0.5, 0.1]:
        fd, _ = build_fd(ds, s, cfg["K"], al, cfg)
        run("pg|alpha=%s|%d" % (al, s), lambda: fl.run_fed("fedlitepg", fd, cfg, np.random.default_rng(s * 7919 + 9)))
    fd, _ = build_fd(ds, s, cfg["K"], 0.5, cfg)
    for frac in [0.0, 0.1, 0.2]:
        fdp = poisoned(fd, frac, s)
        for name, meth, agg in [("fedavg", "fedavg", "mean"), ("fedavg_trim", "fedavg", "trim"),
                                ("fedlitep", "fedlitep", "mean"), ("fedlitep_trim", "fedlitep", "trim")]:
            run("poison2|%s|frac=%s|%d" % (name, frac, s),
                lambda: fl.run_fed(meth, fdp, cfg, np.random.default_rng(s * 7919 + 3), agg=agg))
print("DONE", flush=True)
