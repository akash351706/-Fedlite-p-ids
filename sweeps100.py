"""alpha and rho sweeps at T = 100 rounds (5 seeds). Saves/resumes results/sweeps100_raw.json.
  python sweeps100.py --data-dir /path/to/InSDN_DatasetCSV
"""
import argparse, json, os, time
import data as D
from run_experiments import run_methods, MAIN_METHODS

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True, help="folder with Normal_data.csv, OVS.csv, metasploitable-2.csv")
ap.add_argument("--out", default="results/sweeps100_raw.json")
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
cfg = dict(K=20, rounds=100, E=1, lr=1e-3, bs=128, mu=0.01, ditto_lambda=0.1, cw_exp=0.5, cw_cap=10.0,
           tree_depth=10, n_client_eval=500, rho=0.1)
ds = D.load_csv("InSDN", [os.path.join(a.data_dir, f) for f in ["Normal_data.csv", "OVS.csv", "metasploitable-2.csv"]])
out = json.load(open(a.out)) if os.path.exists(a.out) else {"alpha": {}, "rho": {}}
t0 = time.time()
for s in range(a.seeds):
    for al, key in [(0.1, "0.1"), (1.0, "1.0"), (float("inf"), "inf")]:
        if str(s) in out["alpha"].get(key, {}):
            continue
        out["alpha"].setdefault(key, {})[str(s)] = run_methods(ds, s, al, 0.1, MAIN_METHODS, cfg)
        json.dump(out, open(a.out, "w")); print("alpha", key, "seed", s, int(time.time() - t0), flush=True)
    for rh in [0.01, 0.05, 0.25, 1.0]:
        k = str(rh)
        if str(s) in out["rho"].get(k, {}):
            continue
        out["rho"].setdefault(k, {})[str(s)] = run_methods(ds, s, 0.5, rh, ["fedavg_comp", "fedlitep"], cfg)
        json.dump(out, open(a.out, "w")); print("rho", k, "seed", s, int(time.time() - t0), flush=True)
print("DONE")
