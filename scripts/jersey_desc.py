import numpy as np
from matplotlib.colors import rgb_to_hsv

HUE_BINS = 12
CORE = (0.22, 0.82, 0.18, 0.82)

def _hsv_core(thumbs_bgr, core=CORE):
    rgb = thumbs_bgr[..., ::-1].astype(np.float32) / 255.0
    H, W = rgb.shape[1], rgb.shape[2]
    y0, y1 = int(core[0] * H), int(core[1] * H)
    x0, x1 = int(core[2] * W), int(core[3] * W)
    patch = rgb[:, y0:y1, x0:x1, :]
    return patch, rgb_to_hsv(patch)

def sharpness(thumbs_bgr):
    _, hsv = _hsv_core(thumbs_bgr)
    v = hsv[..., 2]
    gx = np.abs(np.diff(v, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(v, axis=1)).mean(axis=(1, 2))
    return ((gx + gy) / 2).astype(np.float32)

def _skin_mask(h, s, v):
    return (((h < 0.06) | (h > 0.96)) & (s > 0.20) & (s < 0.62) & (v > 0.35))

def descriptor(thumbs_bgr, kind="hue2", bins=None, chroma_thr=0.25):
    patch, hsv = _hsv_core(thumbs_bgr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = len(patch)

    if kind == "lab":
        f = np.concatenate([patch.mean(axis=(1, 2)), patch.std(axis=(1, 2))], 1)
        return f.astype(np.float32)

    nb = bins or HUE_BINS
    chroma = s > chroma_thr
    if kind == "hue":
        white = (~chroma) & (v > 0.55)
        black = (~chroma) & (v <= 0.55)
        wt = s * v
        wwhite, wblack = 0.5, 0.5
    else:
        white = (~chroma) & (v > 0.58)
        black = (~chroma) & (v < 0.32)
        wt = s * v
        wwhite, wblack = 1.0, 1.0

    if kind == "hue2_skin":
        skin = _skin_mask(h, s, v)
        chroma = chroma & ~skin

    out = np.zeros((n, nb + 2), np.float32)
    idx = np.clip((h * nb).astype(int), 0, nb - 1)

    wmask = wt * chroma
    mass = wmask.sum(axis=(1, 2))
    frac = chroma.mean(axis=(1, 2))
    scale = np.divide(frac, mass, out=np.zeros_like(frac), where=mass > 0)
    flat = (np.arange(n)[:, None, None] * nb + idx)
    sel = chroma
    if sel.any():
        out[:, :nb] = np.bincount(
            flat[sel].ravel(),
            weights=(wmask * scale[:, None, None])[sel].ravel(),
            minlength=n * nb).reshape(n, nb)
    out[:, nb] = white.mean(axis=(1, 2)) * wwhite
    out[:, nb + 1] = black.mean(axis=(1, 2)) * wblack
    tot = out.sum(1, keepdims=True)
    return out / np.where(tot == 0, 1, tot)

def cluster(X, method="kmeans", seed=0):
    from sklearn.decomposition import PCA
    k = min(12, X.shape[0] - 1, X.shape[1])
    Xr = PCA(n_components=k, random_state=seed).fit_transform(X) if k >= 2 else X
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(2, covariance_type="full", n_init=5, random_state=seed,
                            reg_covar=1e-4).fit(Xr)
        return g.predict(Xr)
    from sklearn.cluster import KMeans
    return KMeans(2, n_init=10, random_state=seed).fit_predict(Xr)

def order_by_lightness(labels, thumbs_bgr):
    _, hsv = _hsv_core(thumbs_bgr)
    v = hsv[..., 2].mean(axis=(1, 2))
    if v[labels == 0].mean() < v[labels == 1].mean():
        labels = 1 - labels
    return labels

def cluster_k3_drop_junk(X, thumbs_bgr, method="gmm", seed=0):
    import numpy as np
    from sklearn.decomposition import PCA
    n = len(X)
    if n < 24:
        return cluster(X, method, seed), np.ones(n, bool)
    k = min(12, n - 1, X.shape[1])
    Xr = PCA(n_components=k, random_state=seed).fit_transform(X) if k >= 2 else X
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        lab3 = GaussianMixture(3, covariance_type="full", n_init=5,
                               random_state=seed, reg_covar=1e-4).fit_predict(Xr)
    else:
        from sklearn.cluster import KMeans
        lab3 = KMeans(3, n_init=10, random_state=seed).fit_predict(Xr)

    sh = sharpness(thumbs_bgr)
    black = X[:, -1]
    score = []
    for c in range(3):
        m = lab3 == c
        if m.sum() < 4:
            score.append(-1e9)
            continue
        score.append(float(sh[m].mean() * (1 - black[m].mean()) * min(1.0, m.sum() / 20)))
    junk = int(np.argmin(score))
    keep = lab3 != junk
    remap = {c: i for i, c in enumerate([c for c in range(3) if c != junk])}
    return np.array([remap[c] for c in lab3[keep]]), keep

def cluster_fit_clean(X, thumbs_bgr, method="gmm", clean_pct=50, seed=0):
    import numpy as np
    from sklearn.decomposition import PCA
    n = len(X)
    if n < 24:
        return cluster(X, method, seed)
    sh = sharpness(thumbs_bgr)
    colour = X[:, :HUE_BINS].sum(1)
    q = (sh / (sh.max() + 1e-9)) * 0.5 + (colour / (colour.max() + 1e-9)) * 0.5
    clean = q >= np.percentile(q, 100 - clean_pct)
    if clean.sum() < 12:
        clean = np.ones(n, bool)

    k = min(12, clean.sum() - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(X[clean])
    Zc, Z = pca.transform(X[clean]), pca.transform(X)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(2, covariance_type="full", n_init=5, random_state=seed,
                            reg_covar=1e-4).fit(Zc)
        return g.predict(Z)
    from sklearn.cluster import KMeans
    km = KMeans(2, n_init=10, random_state=seed).fit(Zc)
    return km.predict(Z)
