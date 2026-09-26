import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_index import load_index
from jersey_desc import descriptor, HUE_BINS
from score_variants_truth import colour_template, cos

BALL_HUE_LO, BALL_HUE_HI = 0.02, 0.11
WIN = 90

def ring_descriptor(frame, cx, cy, win=WIN, inner=0.28):
    H, W = frame.shape[:2]
    x0, y0 = int(max(0, cx - win)), int(max(0, cy - win))
    x1, y1 = int(min(W, cx + win)), int(min(H, cy + win))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    patch = frame[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (cy - y0), xx - (cx - x0))
    mask = r > inner * win
    if mask.sum() < 64:
        return None
    out = patch.copy()
    out[~mask] = 0
    d = descriptor(out[None], "hue2")[0]
    lo = int(BALL_HUE_LO * HUE_BINS); hi = int(BALL_HUE_HI * HUE_BINS) + 1
    d[lo:hi] *= 0.15
    s = d.sum()
    return d / s if s > 0 else None

def predict(base, clip, mode="late", n_frames=8):
    bp = os.path.join(base, "cache", "cache", "ball", clip.cid + ".npz")
    if not os.path.exists(bp):
        return None
    z = np.load(bp)
    conf, keep, xy = z["conf"], z["keep"], z["xy"]
    idx = np.where(keep)[0] if mode != "raw" else np.where(conf >= 0.35)[0]
    if len(idx) == 0:
        return None
    if mode in ("late", "raw"):
        idx = idx[-n_frames:]
    elif mode == "first":
        idx = idx[:n_frames]
    elif mode == "best":
        idx = idx[np.argsort(-conf[idx])[:n_frames]]

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

    Ta = colour_template(clip.components(clip.away))
    Th = colour_template(clip.components(clip.home))
    if Ta.sum() == 0 or Th.sum() == 0:
        return None

    sa = sh = 0.0
    for i in idx:
        if i >= len(frames):
            continue
        x, y = xy[i]
        if not np.isfinite(x):
            continue
        d = ring_descriptor(frames[i], x, y)
        if d is None:
            continue
        sa += cos(d, Ta)
        sh += cos(d, Th)
    if sa == sh == 0:
        return None
    return 1 if sh > sa else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="late,best,first,raw")
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx = load_index(base)
    clips = [c for c in sorted(idx.values(), key=lambda x: x.cid)
             if c.complete and c.label in ("HOME", "AWAY")]
    print(f"{len(clips)} clips with colours and a label\n")
    print(f"{'mode':<8}{'n':>5}{'acc':>8}{'guess':>8}{'lift':>8}{'95% CI on lift':>20}")
    for mode in args.modes.split(","):
        p, t = [], []
        for c in clips:
            q = predict(base, c, mode, args.n_frames)
            if q is None:
                continue
            p.append(q); t.append(1 if c.label == "HOME" else 0)
        if len(t) < 10:
            print(f"{mode:<8}{len(t):>5}  too little coverage")
            continue
        p, t = np.array(p), np.array(t)
        acc = float((p == t).mean())
        const = float(max(t.mean(), 1 - t.mean()))
        maj = 1 if t.mean() > 0.5 else 0
        d = (p == t).astype(float) - (np.full(len(t), maj) == t).astype(float)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{mode:<8}{len(t):>5}{acc:>8.1%}{const:>8.1%}{acc-const:>+8.1%}"
              f"   [{acc-const-1.96*se:>+6.1%},{acc-const+1.96*se:>+6.1%}]")
    print("\nNo player detection, no tracking, no cluster-to-team mapping -- just the")
    print("colour touching the ball, matched against the two known jerseys.")

if __name__ == "__main__":
    main()
