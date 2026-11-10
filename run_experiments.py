"""Run all experiments and write results.json.

Real data example:
  python run_experiments.py --data InSDN=InSDN/Normal_data.csv,InSDN/OVS.csv,InSDN/metasploitable-2.csv \
        --data TONIoT=ton_iot_network.csv --cross InSDN,TONIoT --out results.json
Smoke test (synthetic; NEVER report these numbers):
  python run_experiments.py --synthetic --seeds 2 --rounds 8 --out results_synth.json
"""
import argparse
import json
import time
import numpy as np
import data as D
import fl

MAIN_METHODS = ["local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]
ALPHAS = [0.1, 0.5, 1.0, float("inf")]
RHOS = [0.01, 0.05, 0.1, 0.25, 1.0]


def build_fd(ds, seed, K, alpha, cfg, extra_ds=None):
    """Split (70/10/20), standardise on train, Dirichlet-partition, build client eval sets."""
    rng = np.random.default_rng(seed)
    tr, va, te = D.split(ds["y"], rng)
    prep = D.Prep().fit(ds["X"][tr])
    Xtr, Xva, Xte = prep(ds["X"][tr]), prep(ds["X"][va]), prep(ds["X"][te])
    ytr, yva, yte = ds["y"][tr], ds["y"][va], ds["y"][te]
    prng = np.random.default_rng(seed * 131 + 7)
    parts = D.dirichlet_partition(ytr, K, alpha, prng)
    erng = np.random.default_rng(seed * 977 + 3)
    cval, ctest = [], []
    for p in parts:
        props = np.bincount(ytr[p], minlength=D.C).astype(float)
        props /= props.sum()
        cval.append(D.client_eval_set(yva, Xva, props, erng, cfg["n_client_eval"]))
        ctest.append(D.client_eval_set(yte, Xte, props, erng, cfg["n_client_eval"]))
    fd = dict(d=Xtr.shape[1], C=D.C, K=K, Xtr=[Xtr[p] for p in parts], ytr=[ytr[p] for p in parts],
              Xtr_all=Xtr, ytr_all=ytr, Xte=Xte, yte=yte, cval=cval, ctest=ctest,
              client_sizes=[int(len(p)) for p in parts])
    extra = {}
    if extra_ds is not None:
        extra["cross"] = (prep(extra_ds["X"]), extra_ds["y"])
    return fd, extra


def run_methods(ds, seed, alpha, rho, methods, cfg, central=False, extra_ds=None):
    c = dict(cfg, rho=rho)
    fd, extra = build_fd(ds, seed, cfg["K"], alpha, c, extra_ds)
    out = {}
    for i, m in enumerate(methods):
        rng = np.random.default_rng(seed * 7919 + i)
        out[m] = fl.run_fed(m, fd, c, rng, extra)
    if central:
        out["central"] = fl.run_central(fd, c, np.random.default_rng(seed * 7919 + 50), extra)
        out["tree"] = fl.run_tree(fd, c, seed, extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=[], help="NAME=file1.csv,file2.csv")
    ap.add_argument("--cross", action="append", default=[], help="TRAIN,TEST dataset names")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--ditto-lambda", type=float, default=0.1)
    ap.add_argument("--max-per-class", type=int, default=20000)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    cfg = dict(K=a.clients, rounds=a.rounds, E=a.epochs, lr=1e-3, bs=128, mu=a.mu, ditto_lambda=a.ditto_lambda,
               cw_exp=0.5, cw_cap=10.0, tree_depth=10, n_client_eval=500, rho=a.rho)
    dsets = {}
    if a.synthetic:
        dsets["SynthA"] = D.make_synthetic("SynthA", seed=1, shift=0.0)
        dsets["SynthB"] = D.make_synthetic("SynthB", seed=2, shift=0.35)
        if not a.cross:
            a.cross = ["SynthA,SynthB"]
    for spec in a.data:
        nm, paths = spec.split("=", 1)
        dsets[nm] = D.load_csv(nm, paths.split(","), max_per_class=a.max_per_class)
    res = dict(config={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in cfg.items()},
               seeds=list(range(a.seed_start, a.seed_start + a.seeds)), meta={n: d["meta"] for n, d in dsets.items()}, main={}, alpha={},
               rho={}, cross={})
    t0 = time.time()
    to_run = [n for n in dsets if not (a.synthetic and n == "SynthB")] if a.synthetic else list(dsets)
    for name in to_run:
        ds = dsets[name]
        R = {"main": {}, "alpha": {}, "rho": {}}
        for s in range(a.seed_start, a.seed_start + a.seeds):
            print("[%s] seed %d  (%.0fs)" % (name, s, time.time() - t0), flush=True)
            r = run_methods(ds, s, a.alpha, a.rho, MAIN_METHODS, cfg, central=True)
            for m, v in r.items():
                R["main"].setdefault(m, []).append(v)
            if a.no_sweeps:
                continue
            for al in ALPHAS:
                key = "inf" if not np.isfinite(al) else str(al)
                if al == a.alpha:
                    r2 = {m: r[m] for m in MAIN_METHODS}
                else:
                    r2 = run_methods(ds, s, al, a.rho, MAIN_METHODS, cfg)
                for m, v in r2.items():
                    R["alpha"].setdefault(key, {}).setdefault(m, []).append(v)
            for rh in RHOS:
                if rh == a.rho:
                    r3 = {m: r[m] for m in ["fedavg_comp", "fedlitep"]}
                else:
                    r3 = run_methods(ds, s, a.alpha, rh, ["fedavg_comp", "fedlitep"], cfg)
                for m, v in r3.items():
                    R["rho"].setdefault(str(rh), {}).setdefault(m, []).append(v)
        res["main"][name] = R["main"]; res["alpha"][name] = R["alpha"]; res["rho"][name] = R["rho"]
    for spec in a.cross:
        A, B = spec.split(",")
        ca, cb = D.common_features(dsets[A], dsets[B])
        print("[cross %s->%s] %d common features" % (A, B, len(ca["features"])), flush=True)
        R = {}
        for s in range(a.seed_start, a.seed_start + a.seeds):
            r = run_methods(ca, s, a.alpha, a.rho, MAIN_METHODS, cfg, central=True, extra_ds=cb)
            for m, v in r.items():
                R.setdefault(m, []).append(v)
        res["cross"]["%s->%s" % (A, B)] = {"n_common_features": len(ca["features"]), "results": R}
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("done in %.0fs -> %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()
