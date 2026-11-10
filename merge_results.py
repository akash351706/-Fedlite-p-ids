"""Merge the main run (results_insdn_r100.json) and the sweeps (sweeps100_raw.json) into one file for make_outputs.py.
  python merge_results.py --main results/results_insdn_r100.json --sweeps results/sweeps100_raw.json --out results/results_merged.json
"""
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True); ap.add_argument("--sweeps", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = json.load(open(a.main)); sw = json.load(open(a.sweeps))
R = res["main"]["InSDN"]; fed = ["local", "fedavg", "fedprox", "fedavg_comp", "fedper", "ditto", "fedlitep"]
RA = {"0.5": {m: R[m] for m in fed}}
for k, d in sw["alpha"].items():
    RA[k] = {}
    for s in sorted(d, key=int):
        for m, v in d[s].items():
            RA[k].setdefault(m, []).append(v)
RR = {"0.1": {m: R[m] for m in ["fedavg_comp", "fedlitep"]}}
for k, d in sw["rho"].items():
    RR[k] = {}
    for s in sorted(d, key=int):
        for m, v in d[s].items():
            RR[k].setdefault(m, []).append(v)
res["alpha"] = {"InSDN": RA}; res["rho"] = {"InSDN": RR}
json.dump(res, open(a.out, "w")); print("written", a.out)
