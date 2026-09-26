import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "cache")
OUT = os.path.join(ROOT, "analysis", "predictions.csv")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "finetune"))
from clip_index import load_index
from jersey_desc import descriptor
from method_ballcolour import ring_descriptor
from score_variants_truth import colour_template, cos
from common import folds, load_items, pretty

CONF = 0.35
CLIP_ID = "openai/clip-vit-base-patch32"
PROMPT = "a basketball player wearing a {c} jersey"
_MNET = None
_CLIP = None

def decode(path, want):
    cap = cv2.VideoCapture(path)
    want = set(int(w) for w in want)
    frames, i = {}, 0
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
            want.discard(i)
        i += 1
    cap.release()
    return frames

def templates(clip):
    ta = colour_template(clip.components(clip.away))
    th = colour_template(clip.components(clip.home))
    return (None, None) if ta.sum() == 0 or th.sum() == 0 else (ta, th)

def prob(sh, sa):
    if sh == sa == 0:
        return None
    return float(0.5 + 0.5 * (sh - sa) / (abs(sh) + abs(sa) + 1e-9))

def ballcolour(clip, last_k):
    p = os.path.join(CACHE, "ball", clip.cid + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    idx = np.where(z["keep"] & (z["conf"] >= CONF))[0]
    if len(idx) == 0:
        return None
    idx = idx[-last_k:] if last_k else idx
    ta, th = templates(clip)
    if ta is None:
        return None
    frames = decode(clip.path, idx)
    sa = sh = 0.0
    for i in idx:
        f = frames.get(int(i))
        x, y = z["xy"][i]
        if f is None or not np.isfinite(x):
            continue
        d = ring_descriptor(f, x, y)
        if d is None:
            continue
        sa += cos(d, ta)
        sh += cos(d, th)
    return prob(sh, sa)

def players_colour(clip, max_frames=24):
    dp = os.path.join(CACHE, "detect", clip.cid + ".npz")
    if not os.path.exists(dp):
        return None
    d = np.load(dp)
    if not len(d["frame_idx"]):
        return None
    ta, th = templates(clip)
    if ta is None:
        return None
    use = np.unique(d["frame_idx"])
    use = use[np.linspace(0, len(use) - 1, min(max_frames, len(use))).round().astype(int)]
    frames = decode(clip.path, use)
    sa = sh = 0.0
    for fi in use:
        f = frames.get(int(fi))
        if f is None:
            continue
        for b in d["xyxy"][d["frame_idx"] == fi]:
            x0, y0, x1, y1 = [int(v) for v in b]
            x0, y0 = max(0, x0), max(0, y0)
            crop = f[y0:y1, x0:x1]
            if crop.shape[0] < 24 or crop.shape[1] < 16:
                continue
            dd = descriptor(cv2.resize(crop, (48, 72))[None], "hue2")[0]
            sa += cos(dd, ta)
            sh += cos(dd, th)
    return prob(sh, sa)

def nearest_player(clip, last_k=8):
    bp = os.path.join(CACHE, "ball", clip.cid + ".npz")
    tp = os.path.join(CACHE, "team_assign", clip.cid + ".npz")
    dp = os.path.join(CACHE, "detect", clip.cid + ".npz")
    if not all(os.path.exists(p) for p in (bp, tp, dp)):
        return None
    z, t, d = np.load(bp), np.load(tp), np.load(dp)
    team_of = {int(r): int(v) for r, v in zip(t["det_row"], t["team"])}
    idx = np.where(z["keep"] & (z["conf"] >= CONF))[0][-last_k:]
    votes = []
    for i in idx:
        bx, by = z["xy"][i]
        if not np.isfinite(bx):
            continue
        rows = [r for r in np.where(d["frame_idx"] == i)[0] if int(r) in team_of]
        if not rows:
            continue
        b = d["xyxy"][rows]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        votes.append(team_of[int(rows[int(np.argmin(np.hypot(cx - bx, cy - by)))])])
    return float(np.mean(votes)) if votes else None

def clip_model():
    global _CLIP
    if _CLIP is None:
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        proc = AutoProcessor.from_pretrained(CLIP_ID)
        model = AutoModel.from_pretrained(CLIP_ID).eval()
        _CLIP = (model, proc, {})
    return _CLIP

def _unit(out):
    v = out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            v = getattr(out, attr)
            break
    else:
        if hasattr(out, "last_hidden_state"):
            v = out.last_hidden_state.mean(1)
    return v / v.norm(dim=-1, keepdim=True)

def clip_zeroshot(clip, scope="frame", k=8, box=160):
    import torch
    model, proc, tcache = clip_model()
    if scope == "ballcrop":
        p = os.path.join(CACHE, "ball", clip.cid + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        idx = np.where(z["keep"] & (z["conf"] >= CONF))[0][-k:]
        if len(idx) == 0:
            return None
        frames = decode(clip.path, idx)
        imgs = []
        for i in idx:
            f = frames.get(int(i))
            x, y = z["xy"][i]
            if f is None or not np.isfinite(x):
                continue
            h, w = f.shape[:2]
            x0, y0 = int(max(0, x - box)), int(max(0, y - box))
            x1, y1 = int(min(w, x + box)), int(min(h, y + box))
            if x1 - x0 > 24 and y1 - y0 > 24:
                imgs.append(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(clip.path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        want = np.linspace(max(0, n - 2 * k), max(0, n - 1), k).round().astype(int)
        frames = decode(clip.path, want)
        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames.values()]
    if len(imgs) < 2:
        return None
    with torch.no_grad():
        im = _unit(model.get_image_features(**proc(images=imgs, return_tensors="pt")))

    def text_emb(colour):
        if colour not in tcache:
            t = proc(text=[PROMPT.format(c=pretty(colour))], return_tensors="pt", padding=True)
            with torch.no_grad():
                tcache[colour] = _unit(model.get_text_features(**t))[0]
        return tcache[colour]

    sh = float((im @ text_emb(clip.colours[clip.home])).mean())
    sa = float((im @ text_emb(clip.colours[clip.away])).mean())
    return prob(sh, sa)

def kinematic_features(clip):
    bp = os.path.join(CACHE, "ball", clip.cid + ".npz")
    dp = os.path.join(CACHE, "detect", clip.cid + ".npz")
    if not (os.path.exists(bp) and os.path.exists(dp)):
        return None
    z, d = np.load(bp), np.load(dp)
    ok = z["keep"] & (z["conf"] >= CONF)
    if ok.sum() < 4:
        return None
    xy = z["xy"][ok].astype(float)
    fi = z["frame_idx"][ok].astype(float)
    v = np.diff(xy, axis=0) / np.maximum(np.diff(fi)[:, None], 1)
    sp = np.hypot(v[:, 0], v[:, 1])
    acc = np.diff(sp) if len(sp) > 1 else np.array([0.0])
    ang = np.arctan2(v[:, 1], v[:, 0])
    dang = np.abs(np.diff(ang)) if len(ang) > 1 else np.array([0.0])
    per_f = np.array([np.sum(d["frame_idx"] == f) for f in np.unique(d["frame_idx"])])
    bh = d["xyxy"][:, 3] - d["xyxy"][:, 1]
    feats = [ok.mean(), ok.sum(), len(xy), sp.mean(), sp.std(), sp.max(), np.median(sp),
             acc.mean(), acc.std(), np.abs(acc).max(), dang.mean(), dang.max(),
             xy[:, 0].std(), xy[:, 1].std(), np.ptp(xy[:, 0]), np.ptp(xy[:, 1]),
             xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1], fi[-1] - fi[0],
             per_f.mean(), per_f.std(), per_f.max(), np.median(bh), bh.std(),
             np.percentile(bh, 90),
             sp[-3:].mean() if len(sp) >= 3 else sp.mean(),
             dang[-3:].mean() if len(dang) >= 3 else dang.mean()]
    return np.array(feats, float)

def mobilenet_features(clip, maxf=16):
    global _MNET
    cdir = os.path.join(CACHE, "mobilenet")
    os.makedirs(cdir, exist_ok=True)
    p = os.path.join(cdir, clip.cid + ".npy")
    if os.path.exists(p):
        return np.load(p).mean(0)
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    if _MNET is None:
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        net = mobilenet_v3_small(weights=w).eval()
        net.classifier = torch.nn.Sequential(*list(net.classifier.children())[:-1])
        _MNET = (net, w.transforms())
    net, tf = _MNET
    cap = cv2.VideoCapture(clip.path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None
    pick = np.linspace(0, len(frames) - 1, min(maxf, len(frames))).astype(int)
    batch = torch.stack([tf(torch.from_numpy(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                            .permute(2, 0, 1)) for i in pick])
    with torch.no_grad():
        out = net(batch).numpy().astype(np.float32)
    np.save(p, out)
    return out.mean(0)

def out_of_fold(items, x, name, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.array([it.y_home for it in items])
    g = np.array([it.game for it in items])
    p = np.full(len(y), np.nan)
    for tr, te in folds(items, "game", 5, seed):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)),
                          {"logisticregression__C": [0.01, 0.1, 1, 10]},
                          cv=list(inner.split(x[tr], y[tr], g[tr])), n_jobs=1)
        gs.fit(x[tr], y[tr])
        p[te] = gs.predict_proba(x[te])[:, 1]
    return [(it.cid, name, float(v)) for it, v in zip(items, p) if np.isfinite(v)]

def merge_and_write(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            old = list(csv.reader(fh))[1:]
    fresh = {(c, m) for c, m, _ in rows}
    keep = [(r[0], r[1], float(r[2])) for r in old if (r[0], r[1]) not in fresh]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "method", "p_home", "pred"])
        for c, m, v in sorted(keep + rows):
            w.writerow([c, m, f"{v:.4f}", int(v > 0.5)])
    print(f"\n{len(rows)} new predictions, {len(keep)} kept -> {OUT}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    want = set(args.methods.split(","))
    run_all = "all" in want
    clips = {c.cid: c for c in load_index(ROOT).values()}
    items = load_items()[:args.limit] if args.limit else load_items()
    rows = []

    simple = [("ballcolour_last8", lambda c: ballcolour(c, 8)),
              ("ballcolour_last24", lambda c: ballcolour(c, 24)),
              ("ballcolour_all", lambda c: ballcolour(c, 0)),
              ("players_colour", players_colour),
              ("nearest_player", nearest_player),
              ("clip_zeroshot_frame", lambda c: clip_zeroshot(c, "frame")),
              ("clip_zeroshot_ballcrop", lambda c: clip_zeroshot(c, "ballcrop"))]
    for name, fn in simple:
        if not (run_all or name in want):
            continue
        got = 0
        for n, it in enumerate(items, 1):
            v = fn(clips[it.cid])
            if v is not None:
                rows.append((it.cid, name, v))
                got += 1
            if n % 25 == 0:
                print(f"  {name}: {n}/{len(items)}", flush=True)
        print(f"{name}: {got}/{len(items)} clips", flush=True)

    if run_all or "kinematic_linear" in want:
        use = [(it, kinematic_features(clips[it.cid])) for it in items]
        use = [(it, f) for it, f in use if f is not None]
        if len(use) > 20:
            rows += out_of_fold([it for it, _ in use], np.stack([f for _, f in use]),
                                "kinematic_linear")
            print(f"kinematic_linear: {len(use)}/{len(items)} clips", flush=True)

    if run_all or "mobilenet_linear" in want:
        use = []
        for n, it in enumerate(items, 1):
            use.append((it, mobilenet_features(clips[it.cid])))
            if n % 25 == 0:
                print(f"  mobilenet features: {n}/{len(items)}", flush=True)
        use = [(it, f) for it, f in use if f is not None]
        if len(use) > 20:
            rows += out_of_fold([it for it, _ in use], np.stack([f for _, f in use]),
                                "mobilenet_linear")
            print(f"mobilenet_linear: {len(use)}/{len(items)} clips", flush=True)

    merge_and_write(rows)

if __name__ == "__main__":
    main()
