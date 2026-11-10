"""Data loading, label unification, cleaning, splitting (70/10/20), Dirichlet partition.

Edit RULES / ALIASES / DROP_COL if your dataset versions use other names.
Everything that is removed is counted and stored in `meta` (used by make_outputs.py).
"""
import re
import numpy as np
import pandas as pd

CLASSES = ["benign", "dos", "recon", "bruteforce", "web", "malware"]
C = len(CLASSES)

# First matching rule wins (order matters: "Web Attack - Brute Force" must be 'web').
RULES = [
    ("benign", ["benign", "normal"]),
    ("web", ["web", "xss", "sql", "inject", "upload", "browserhij"]),
    ("bruteforce", ["brute", "patator", "password", "dictionary", "bfa"]),
    ("recon", ["scan", "recon", "probe", "portsweep", "nmap", "fingerprint"]),
    ("dos", ["ddos", "dos", "flood", "hulk", "slowloris", "slowhttptest", "goldeneye"]),
    ("malware", ["bot", "malware", "ransom", "backdoor", "trojan", "worm", "shellcode",
                 "mirai", "infiltration"]),
]
LABEL_PRIORITY = ["attack_cat", "type", "attack_type", "class", "label", "attack"]
DROP_COL = re.compile(r"(^|_)(id|ip|timestamp|date|time|mac|index)$|^unnamed")

# CICFlowMeter v3 (InSDN, newer CIC sets) -> CIC-IDS2017 style names, so that
# datasets can share features. VERIFY against your CSV headers.
ALIASES = {
    "tot_fwd_pkts": "total_fwd_packets", "tot_bwd_pkts": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets", "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max", "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean", "fwd_pkt_len_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max", "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean", "bwd_pkt_len_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_s", "flow_pkts_s": "flow_packets_s",
    "fwd_iat_tot": "fwd_iat_total", "bwd_iat_tot": "bwd_iat_total",
    "fwd_header_len": "fwd_header_length", "bwd_header_len": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_s", "bwd_pkts_s": "bwd_packets_s",
    "pkt_len_min": "min_packet_length", "pkt_len_max": "max_packet_length",
    "pkt_len_mean": "packet_length_mean", "pkt_len_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "fin_flag_cnt": "fin_flag_count", "syn_flag_cnt": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count", "psh_flag_cnt": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count", "urg_flag_cnt": "urg_flag_count",
    "ece_flag_cnt": "ece_flag_count", "pkt_size_avg": "average_packet_size",
    "init_fwd_win_byts": "init_win_bytes_forward", "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd", "fwd_seg_size_min": "min_seg_size_forward",
    "fwd_seg_size_avg": "avg_fwd_segment_size", "bwd_seg_size_avg": "avg_bwd_segment_size",
    "subflow_fwd_pkts": "subflow_fwd_packets", "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes", "subflow_bwd_byts": "subflow_bwd_bytes",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def unify_label(s):
    t = str(s).lower()
    for cls, keys in RULES:
        if any(k in t for k in keys):
            return CLASSES.index(cls)
    return -1


def load_csv(name, paths, max_per_class=20000, min_class=30, dedupe=True, seed=0):
    """Return dict(X, y, features, meta). Class cap keeps rare classes intact."""
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    meta = {"name": name, "files": [str(p) for p in paths], "raw_rows": int(len(df))}
    cols = [norm(c) for c in df.columns]
    df.columns = [ALIASES.get(c, c) for c in cols]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    lab = None
    for cand in LABEL_PRIORITY:
        if cand in df.columns and (df[cand].dtype == object or pd.api.types.is_string_dtype(df[cand])):
            lab = cand
            break
    if lab is None:
        raise ValueError("No textual label column found; columns: %s" % list(df.columns)[:20])
    meta["label_column"] = lab
    raw = df[lab].astype(str)
    y = raw.map(unify_label).to_numpy()
    unm = raw[y < 0].value_counts().to_dict()
    meta["unmapped_labels_dropped"] = {str(k): int(v) for k, v in unm.items()}
    meta["raw_label_counts"] = {str(k): int(v) for k, v in raw.value_counts().items()}
    feat = [c for c in df.columns if c not in LABEL_PRIORITY and not DROP_COL.search(c)]
    X = df[feat].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.5]
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = (y >= 0) & X.notna().all(axis=1).to_numpy()
    meta["rows_removed_unmapped"] = int((y < 0).sum())
    meta["rows_removed_nan_inf"] = int(((y >= 0) & ~X.notna().all(axis=1).to_numpy()).sum())
    X, y = X[keep].reset_index(drop=True), y[keep]
    if dedupe:
        d = X.copy(); d["__y"] = y
        mask = ~d.duplicated().to_numpy()
        meta["rows_removed_duplicates"] = int((~mask).sum())
        X, y = X[mask].reset_index(drop=True), y[mask]
    cnt = np.bincount(y, minlength=C)
    meta["class_counts_after_cleaning"] = {CLASSES[c]: int(cnt[c]) for c in range(C)}
    rng = np.random.default_rng(seed)
    sel = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) < min_class:
            continue
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        sel.append(idx)
    sel = np.sort(np.concatenate(sel))
    X, y = X.iloc[sel], y[sel]
    meta["max_per_class"] = max_per_class
    meta["classes_dropped_too_rare"] = [CLASSES[c] for c in range(C) if 0 < cnt[c] < min_class]
    cnt = np.bincount(y, minlength=C)
    meta["class_counts_used"] = {CLASSES[c]: int(cnt[c]) for c in range(C)}
    meta["n_used"] = int(len(y)); meta["n_features"] = int(X.shape[1])
    return {"X": X.to_numpy(np.float64), "y": y, "features": list(X.columns), "meta": meta}


def make_synthetic(name, n=24000, d=30, seed=0, shift=0.0):
    """Synthetic flow-like data for smoke tests ONLY (never report these numbers)."""
    rng = np.random.default_rng(1234)
    mu = rng.standard_normal((C, d)) * 0.9
    scale = 1 + 0.4 * rng.random((C, d))
    rng2 = np.random.default_rng(seed + 17)
    pri = np.array([0.55, 0.16, 0.09, 0.05, 0.08, 0.07])
    y = rng2.choice(C, n, p=pri)
    mus = mu + shift * np.random.default_rng(99).standard_normal((C, d))
    Z = mus[y] + scale[y] * rng2.standard_normal((n, d))
    X = np.sign(Z) * np.expm1(np.abs(Z))
    cnt = np.bincount(y, minlength=C)
    meta = {"name": name, "files": ["synthetic"], "raw_rows": n, "n_used": n, "n_features": d,
            "class_counts_used": {CLASSES[c]: int(cnt[c]) for c in range(C)}}
    return {"X": X, "y": y, "features": ["f%d" % i for i in range(d)], "meta": meta}


def common_features(a, b):
    fs = [f for f in a["features"] if f in set(b["features"])]
    if len(fs) < 5:
        raise ValueError("Only %d common features between %s and %s" % (len(fs), a["meta"]["name"], b["meta"]["name"]))
    ia = [a["features"].index(f) for f in fs]
    ib = [b["features"].index(f) for f in fs]
    return ({**a, "X": a["X"][:, ia], "features": fs}, {**b, "X": b["X"][:, ib], "features": fs})


def split(y, rng, fr=(0.7, 0.1, 0.2)):
    tr, va, te = [], [], []
    for c in range(C):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n == 0:
            continue
        a = max(1, int(round(fr[0] * n))); b = max(1, int(round(fr[1] * n)))
        if a + b >= n:
            a, b = max(1, n - 2), 1
        tr += idx[:a].tolist(); va += idx[a:a + b].tolist(); te += idx[a + b:].tolist()
    return np.array(tr), np.array(va), np.array(te)


class Prep:
    """sign(x)log(1+|x|) then standardisation with TRAIN statistics only."""
    def fit(self, X):
        Z = self._t(X); self.mu = Z.mean(0); self.sd = Z.std(0); self.sd[self.sd < 1e-8] = 1.0
        return self

    @staticmethod
    def _t(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def __call__(self, X):
        return (self._t(X) - self.mu) / self.sd


def dirichlet_partition(y, K, alpha, rng, min_size=10):
    if not np.isfinite(alpha):
        return [np.array(p) for p in np.array_split(rng.permutation(len(y)), K)]
    for _ in range(200):
        idx = [[] for _ in range(K)]
        for c in range(C):
            ic = np.where(y == c)[0]
            if len(ic) == 0:
                continue
            rng.shuffle(ic)
            p = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
            for k, part in enumerate(np.split(ic, cuts)):
                idx[k] += part.tolist()
        if min(len(i) for i in idx) >= min_size:
            break
    return [np.array(i, dtype=int) for i in idx]


def client_eval_set(y_pool, X_pool, props, rng, n=500):
    """n flows drawn with replacement from the pool with the client's class proportions."""
    have = np.bincount(y_pool, minlength=C) > 0
    p = props * have
    p = p / p.sum() if p.sum() > 0 else have / have.sum()
    cls = rng.choice(C, n, p=p)
    out = np.empty(n, dtype=int)
    for c in np.unique(cls):
        pos = np.where(cls == c)[0]
        out[pos] = rng.choice(np.where(y_pool == c)[0], len(pos))
    return X_pool[out], y_pool[out]
