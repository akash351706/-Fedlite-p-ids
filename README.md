# FedLite-P: lightweight federated intrusion detection for SDN-enabled IoT

Code and raw results for the paper *"Lightweight Federated Intrusion Detection for SDN-Enabled IoT Networks with Non-IID Traffic and Communication-Efficient Personalized Updates"* (Akash, A. Asha).

The code is plain Python (NumPy, SciPy, pandas, scikit-learn, matplotlib). No GPU is needed.

## What is here
| File | Purpose |
|---|---|
| `data.py` | loading and cleaning, label unification into six classes (edit `RULES`), 70/10/20 split, Dirichlet partition |
| `fl.py` | NumPy MLP; FedAvg, FedProx, FedAvg+sparsification/int8, FedPer, Ditto, local-only, **FedLite-P**; centralized MLP and decision tree |
| `run_experiments.py` | main comparison (5 seeds); writes a results JSON |
| `sweeps100.py` | alpha and rho sweeps at 100 rounds (5 seeds) |
| `merge_results.py` | merges main run and sweeps into one file |
| `make_outputs.py` | LaTeX tables, figures and `summary.txt` (facts only) |
| `results/` | raw results used in the paper, summary, tables and figures |

`results/results_insdn_r100.json` holds the main results (T = 100 rounds); `results/sweeps100_raw.json` the sweeps (T = 100);
`results/pilot_30_rounds/` an earlier pilot with T = 30 that is mentioned in the paper but not used for its tables.

## Data (not included)
The InSDN dataset (Elsayed, Le-Khac, Jurcut, IEEE Access 8, 2020) is public; download it from its official page.
Put `Normal_data.csv`, `OVS.csv` and `metasploitable-2.csv` in one folder. Raw data are not redistributed here.

## Reproduce the paper's results
```
pip install -r requirements.txt
python run_experiments.py --rounds 100 --no-sweeps --out results/results_insdn_r100.json \
  --data InSDN=DATA/Normal_data.csv,DATA/OVS.csv,DATA/metasploitable-2.csv
python sweeps100.py --data-dir DATA
python merge_results.py --main results/results_insdn_r100.json --sweeps results/sweeps100_raw.json --out results/results_merged.json
python make_outputs.py --results results/results_merged.json --outdir outputs
```
Replace `DATA` with your folder. The main run takes about 10 minutes and the sweeps about 35 minutes on one CPU core.
Seeds 0 to 4 control the split, partition and initialization, so reruns should match the stored results up to small numerical differences between platforms.

Quick check on synthetic data (numbers are meaningless, never report them):
```
python run_experiments.py --synthetic --seeds 1 --rounds 3 --no-sweeps --out test.json
```

## Protocol summary
K = 20 clients, class-wise Dirichlet partition (alpha = 0.5 main), E = 1 local epoch, Adam (lr 1e-3, batch 128), MLP 64-32, FedProx mu = 0.01, Ditto lambda = 0.1,
top-k fraction rho = 0.1 with int8 quantization and error feedback. Split 70/10/20; the validation split picks the reported round.
Exact duplicates are removed before splitting; at most 20,000 flows per class are kept. Hyper-parameters were fixed, not tuned per method.

## Limits
One dataset, a public flow dataset rather than a live SDN testbed, honest clients and server assumed, payload bytes only. See the paper's Section 7.

## License
MIT (see `LICENSE`). Datasets keep their own licenses.

## Citation
Cite the paper (details to be added after publication).
