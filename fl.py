"""NumPy MLP and the federated algorithms: FedAvg, FedProx, FedAvg+sparse/int8, FedPer,
Ditto, local-only, FedLite-P; plus centralized MLP and decision tree."""
import math
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

H1, H2 = 64, 32

# ---------------------------------------------------------------- model
def sizes(d, C):
    return d * H1 + H1 + H1 * H2 + H2, H2 * C + C  # body, head


def init_theta(d, C, rng):
    nb, nh = sizes(d, C)
    th = np.zeros(nb + nh)
    W1, b1, W2, b2, W3, b3 = views(th, d, C)
    W1[:] = rng.standard_normal((d, H1)) * math.sqrt(2 / d)
    W2[:] = rng.standard_normal((H1, H2)) * math.sqrt(2 / H1)
    W3[:] = rng.standard_normal((H2, C)) * math.sqrt(1 / H2)
    return th


def views(th, d, C):
    out, o = [], 0
    for shp in [(d, H1), (H1,), (H1, H2), (H2,), (H2, C), (C,)]:
        n = int(np.prod(shp)); out.append(th[o:o + n].reshape(shp)); o += n
    return out


def logits(th, X, d, C):
    W1, b1, W2, b2, W3, b3 = views(th, d, C)
    return np.maximum(np.maximum(X @ W1 + b1, 0) @ W2 + b2, 0) @ W3 + b3


def predict(th, X, d, C):
    return logits(th, X, d, C).argmax(1)


def loss_grad(th, g, X, y, cw, d, C):
    W1, b1, W2, b2, W3, b3 = views(th, d, C)
    gW1, gb1, gW2, gb2, gW3, gb3 = views(g, d, C)
    z1 = X @ W1 + b1; a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2; a2 = np.maximum(z2, 0)
    z = a2 @ W3 + b3
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    w = cw[y]; sw = w.sum()
    p[np.arange(len(y)), y] -= 1; p *= (w / sw)[:, None]
    gW3[:] = a2.T @ p; gb3[:] = p.sum(0)
    d2 = (p @ W3.T) * (z2 > 0); gW2[:] = a1.T @ d2; gb2[:] = d2.sum(0)
    d1 = (d2 @ W2.T) * (z1 > 0); gW1[:] = X.T @ d1; gb1[:] = d1.sum(0)


def predict_dual(th, X, d, C, lam):
    nb, nh = sizes(d, C)
    W1, b1, W2, b2, W3g, b3g = views(th, d, C)
    a2 = np.maximum(np.maximum(X @ W1 + b1, 0) @ W2 + b2, 0)
    hl = th[nb + nh:]
    z = lam * (a2 @ W3g + b3g) + (1 - lam) * (a2 @ hl[:H2 * C].reshape(H2, C) + hl[H2 * C:])
    return z.argmax(1)


def loss_grad_dual(th, g, X, y, cw, d, C):
    """Shared body + global head (aggregated) + local head (never sent); loss = CE(global) + CE(local)."""
    nb, nh = sizes(d, C); o = nb + nh
    W1, b1, W2, b2, W3g, b3g = views(th, d, C)
    gW1, gb1, gW2, gb2, gW3g, gb3g = views(g, d, C)
    W3l = th[o:o + H2 * C].reshape(H2, C); b3l = th[o + H2 * C:]
    gW3l = g[o:o + H2 * C].reshape(H2, C); gb3l = g[o + H2 * C:]
    z1 = X @ W1 + b1; a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2; a2 = np.maximum(z2, 0)
    w = cw[y]; sw = w.sum(); idx = np.arange(len(y))
    def grad_p(z):
        z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        p[idx, y] -= 1; p *= (w / sw)[:, None]
        return p
    pg = grad_p(a2 @ W3g + b3g); pl = grad_p(a2 @ W3l + b3l)
    gW3g[:] = a2.T @ pg; gb3g[:] = pg.sum(0); gW3l[:] = a2.T @ pl; gb3l[:] = pl.sum(0)
    d2 = (pg @ W3g.T + pl @ W3l.T) * (z2 > 0); gW2[:] = a1.T @ d2; gb2[:] = d2.sum(0)
    d1 = (d2 @ W2.T) * (z1 > 0); gW1[:] = X.T @ d1; gb1[:] = d1.sum(0)


def agg_trim(deltas, beta):
    D_ = np.sort(np.asarray(deltas), axis=0); k = int(beta * len(D_))
    return D_[k:len(D_) - k].mean(0)


def class_weights(y, C, expo=0.5, cap=10.0):
    cnt = np.bincount(y, minlength=C); n = len(y)
    w = np.ones(C); nz = cnt > 0
    w[nz] = np.minimum((n / (C * cnt[nz])) ** expo, cap)
    return w


def train_local(th, X, y, cw, d, C, epochs, lr, bs, rng, prox_ref=None, mu=0.0, adam=None, lossfn=None):
    lossfn = lossfn or loss_grad
    """Adam on the class-weighted CE; th is updated in place. Optional proximal term."""
    n = len(y); g = np.zeros_like(th)
    st = adam if adam is not None else {}
    m = st.setdefault("m", np.zeros_like(th)); v = st.setdefault("v", np.zeros_like(th))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:
                continue
            lossfn(th, g, X[idx], y[idx], cw, d, C)
            if prox_ref is not None:
                g += mu * (th - prox_ref)
            st["t"] = t = st.get("t", 0) + 1
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            th -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)


# ---------------------------------------------------------------- compression & cost
def compress(v, rho):
    n = len(v); k = max(1, int(math.ceil(rho * n)))
    if k >= n:
        idx = np.arange(n)
    else:
        idx = np.argpartition(-np.abs(v), k - 1)[:k]
    vals = v[idx]; s = np.abs(vals).max() / 127.0
    s = s if s > 0 else 1e-12
    out = np.zeros_like(v); out[idx] = np.clip(np.round(vals / s), -127, 127) * s
    return out


def uplink_bytes(ns, rho, compressed):
    if not compressed:
        return 4 * ns
    k = int(math.ceil(rho * ns))
    if k >= ns:  # quantisation only, no indices
        return ns + 4
    return k * (1 + int(math.ceil(math.log2(ns) / 8))) + 4


# ---------------------------------------------------------------- metrics
def metrics(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); sup = cm.sum(1); pc = cm.sum(0)
    pres = sup > 0
    rec = np.full(C, np.nan); rec[pres] = tp[pres] / sup[pres]
    prec = tp / np.maximum(pc, 1)
    f1 = 2 * tp / np.maximum(sup + pc, 1)
    far = cm[0, 1:].sum() / sup[0] if sup[0] > 0 else np.nan
    att = cm[1:, :].sum()
    miss = cm[1:, 0].sum() / att if att > 0 else np.nan
    return dict(f1=float(f1[pres].mean()), prec=float(prec[pres].mean()),
                rec=float(rec[pres].mean()), acc=float(tp.sum() / cm.sum()),
                far=float(far), miss=float(miss), recall=rec.tolist())


def eval_client_sets(models, sets, d, C, pf=None):
    pf = pf or predict
    ys, ps = [], []
    for th, (X, y) in zip(models, sets):
        ys.append(y); ps.append(pf(th, X, d, C))
    return metrics(np.concatenate(ys), np.concatenate(ps), C)


def eval_global(models, X, y, d, C, n_clients=10, pf=None):
    pf = pf or predict
    seen, res = {}, []
    for th in models[:n_clients]:
        key = id(th)
        if key not in seen:
            seen[key] = metrics(y, pf(th, X, d, C), C)
        res.append(seen[key])
    out = {k: float(np.nanmean([r[k] for r in res])) for k in ["f1", "prec", "rec", "acc", "far", "miss"]}
    out["recall"] = np.nanmean(np.array([r["recall"] for r in res]), axis=0).tolist()
    return out


def latency_ms(th, x1, d, C, reps=1000, pf=None):
    predict_ = pf or predict
    for _ in range(20):
        predict_(th, x1, d, C)
    t0 = time.perf_counter()
    for _ in range(reps):
        predict_(th, x1, d, C)
    return (time.perf_counter() - t0) / reps * 1e3


# ---------------------------------------------------------------- federated runs
SPEC = {
    "fedavg": dict(personal=False, comp=False, mu=0.0),
    "fedprox": dict(personal=False, comp=False, mu=None),  # mu from cfg
    "fedavg_comp": dict(personal=False, comp=True, mu=0.0),
    "fedper": dict(personal=True, comp=False, mu=0.0),
    "fedlitep": dict(personal=True, comp=True, mu=0.0),
    "fedlitepg": dict(personal="dual", comp=True, mu=0.0),   # FedLite-P with an additional aggregated global head
    "ditto": dict(personal="ditto", comp=False, mu=0.0),
    "local": dict(personal=False, comp=False, mu=0.0),
}
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def finish(models, fd, extra, d, C, curve, best_round, uplink, val_curve, lat, pf=None):
    out = dict(client=curve["best_rec"], glob=eval_global(models, fd["Xte"], fd["yte"], d, C, pf=pf),
               best_round=best_round, uplink=uplink, curve=curve["test_f1"], val_curve=val_curve,
               latency_ms=lat, extra={})
    for nm, (Xe, ye) in (extra or {}).items():
        out["extra"][nm] = eval_global(models, Xe, ye, d, C, pf=pf)
    return out


def run_fed(name, fd, cfg, rng, extra=None, agg="mean", beta=0.2):
    """agg='trim' uses a coordinate-wise trimmed mean of the (decoded) updates instead of the weighted mean."""
    d, C, K = fd["d"], fd["C"], fd["K"]
    nb, nh = sizes(d, C)
    sp = SPEC[name]; personal = sp["personal"]
    dual = personal == "dual"; pers_head = personal is True or dual
    mu = cfg["mu"] if sp["mu"] is None else sp["mu"]
    th0 = init_theta(d, C, rng)
    cws = [class_weights(y, C, cfg["cw_exp"], cfg["cw_cap"]) for y in fd["ytr"]]
    nk = np.array([len(y) for y in fd["ytr"]], float)
    ns = nb if personal is True else nb + nh
    comp = sp["comp"]
    lam = cfg.get("lam", 0.5)
    pf = (lambda th, X, d_, C_: predict_dual(th, X, d_, C_, lam)) if dual else None
    lossfn = loss_grad_dual if dual else None
    hon = fd.get("honest")   # poisoning experiments: evaluate honest clients only
    sel = (lambda ms: [ms[k] for k in hon]) if hon is not None else (lambda ms: ms)
    cval = [fd["cval"][k] for k in hon] if hon is not None else fd["cval"]
    ctest = [fd["ctest"][k] for k in hon] if hon is not None else fd["ctest"]
    loc = None
    if name == "local":
        loc = [th0.copy() for _ in range(K)]
    elif personal == "ditto":
        g = th0.copy(); loc = [th0.copy() for _ in range(K)]
    else:
        g = th0[:ns].copy()
        heads = [th0[nb:].copy() for _ in range(K)] if pers_head else None
    resid = [np.zeros(ns) for _ in range(K)] if comp else None
    best, curve, vcurve = None, [], []
    for t in range(cfg["rounds"]):
        if name == "local":
            for k in range(K):
                train_local(loc[k], fd["Xtr"][k], fd["ytr"][k], cws[k], d, C, cfg["E"], cfg["lr"], cfg["bs"], rng)
            models = loc
        else:
            deltas = []
            for k in range(K):
                th = np.concatenate([g, heads[k]]) if pers_head else g.copy()
                ref = th.copy() if mu > 0 else None
                train_local(th, fd["Xtr"][k], fd["ytr"][k], cws[k], d, C, cfg["E"], cfg["lr"], cfg["bs"], rng,
                            prox_ref=ref, mu=mu, lossfn=lossfn)
                delta = th[:ns] - g
                if pers_head:
                    heads[k] = th[ns:].copy()
                if comp:
                    vv = delta + resid[k]; dh = compress(vv, cfg["rho"]); resid[k] = vv - dh; delta = dh
                deltas.append(delta)
                if personal == "ditto":
                    train_local(loc[k], fd["Xtr"][k], fd["ytr"][k], cws[k], d, C, cfg["E"], cfg["lr"], cfg["bs"], rng,
                                prox_ref=g.copy(), mu=cfg["ditto_lambda"])
            if agg == "trim":
                g = g + agg_trim(deltas, beta)
            else:
                g = g + (nk[:, None] * np.array(deltas)).sum(0) / nk.sum()
            if personal == "ditto":
                models = loc
            elif pers_head:
                models = [np.concatenate([g, h]) for h in heads]
            else:
                models = [g] * K
        vm = eval_client_sets(sel(models), cval, d, C, pf)["f1"]
        tm = eval_client_sets(sel(models), ctest, d, C, pf)
        vcurve.append(vm); curve.append(tm["f1"])
        if best is None or vm > best[0]:
            if models is loc or pers_head:
                snap = [m.copy() for m in models]
            else:
                c = g.copy(); snap = [c] * K
            best = (vm, t + 1, snap, tm)
    uplink = 0 if name == "local" else uplink_bytes(ns, cfg["rho"], comp)
    lat = latency_ms(best[2][0], fd["Xte"][:1], d, C, pf=pf)
    out = finish(sel(best[2]), fd, extra, d, C, dict(test_f1=curve, best_rec=best[3]), best[1], uplink, vcurve, lat, pf)
    if dual:  # sensitivity to the mixing weight, same trained models
        out["lam_sweep"] = {}
        for l_ in LAMS:
            p_ = (lambda th, X, d_, C_, l_=l_: predict_dual(th, X, d_, C_, l_))
            out["lam_sweep"][str(l_)] = dict(client=eval_client_sets(best[2], fd["ctest"], d, C, p_),
                                            glob=eval_global(best[2], fd["Xte"], fd["yte"], d, C, pf=p_))
    return out


def run_central(fd, cfg, rng, extra=None):
    d, C, K = fd["d"], fd["C"], fd["K"]
    th = init_theta(d, C, rng)
    cw = class_weights(fd["ytr_all"], C, cfg["cw_exp"], cfg["cw_cap"])
    adam, best, curve, vcurve = {}, None, [], []
    for t in range(cfg["rounds"]):
        train_local(th, fd["Xtr_all"], fd["ytr_all"], cw, d, C, 1, cfg["lr"], cfg["bs"], rng, adam=adam)
        vm = eval_client_sets([th] * K, fd["cval"], d, C)["f1"]
        tm = eval_client_sets([th] * K, fd["ctest"], d, C)
        vcurve.append(vm); curve.append(tm["f1"])
        if best is None or vm > best[0]:
            best = (vm, t + 1, th.copy(), tm)
    snap = [best[2]] * K
    lat = latency_ms(best[2], fd["Xte"][:1], d, C)
    return finish(snap, fd, extra, d, C, dict(test_f1=curve, best_rec=best[3]), best[1], None, vcurve, lat)


class _TreeModel:
    def __init__(self, clf): self.clf = clf


def run_tree(fd, cfg, seed, extra=None):
    d, C, K = fd["d"], fd["C"], fd["K"]
    cw = class_weights(fd["ytr_all"], C, cfg["cw_exp"], cfg["cw_cap"])
    clf = DecisionTreeClassifier(max_depth=cfg["tree_depth"], random_state=seed)
    clf.fit(fd["Xtr_all"], fd["ytr_all"], sample_weight=cw[fd["ytr_all"]])

    def ev(sets):
        ys = np.concatenate([y for _, y in sets]); ps = np.concatenate([clf.predict(X) for X, _ in sets])
        return metrics(ys, ps, C)
    glob = metrics(fd["yte"], clf.predict(fd["Xte"]), C)
    x1 = fd["Xte"][:1]
    for _ in range(20): clf.predict(x1)
    t0 = time.perf_counter()
    for _ in range(500): clf.predict(x1)
    lat = (time.perf_counter() - t0) / 500 * 1e3
    out = dict(client=ev(fd["ctest"]), glob={k: v for k, v in glob.items()}, best_round=None, uplink=None,
               curve=[], val_curve=[], latency_ms=lat, extra={})
    for nm, (Xe, ye) in (extra or {}).items():
        out["extra"][nm] = metrics(ye, clf.predict(Xe), C)
    return out
