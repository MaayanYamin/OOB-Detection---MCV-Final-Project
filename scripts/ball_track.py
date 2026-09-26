import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_wasb_cpu import build_model, MEAN, STD

def peaks_for_clip(model, video, max_frames=30, iw=512, ih=288):
    import torch
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return None
    H, W = frames[0].shape[:2]
    sx, sy = W / iw, H / ih
    sm = [(cv2.cvtColor(cv2.resize(f, (iw, ih)), cv2.COLOR_BGR2RGB).astype(np.float32)
           / 255.0 - MEAN) / STD for f in frames]
    xy, conf = [], []
    with torch.no_grad():
        for s in range(0, len(sm) - 2, 3):
            x = np.concatenate([w.transpose(2, 0, 1) for w in sm[s:s + 3]], 0)
            o = model(torch.from_numpy(x[None]).float())
            hm = o[0] if isinstance(o, dict) else (o[0] if isinstance(o, (list, tuple)) else o)
            hm = torch.sigmoid(hm)[0].numpy()
            for j in range(min(3, hm.shape[0])):
                h = hm[j]
                iy, ix = np.unravel_index(np.argmax(h), h.shape)
                xy.append((ix * sx, iy * sy))
                conf.append(float(h.max()))
    return np.array(xy, np.float32), np.array(conf, np.float32), frames, (W, H)

def trajectory_filter(xy, conf, conf_thr=0.35, win=7, tol=3.0, iters=3):
    n = len(xy)
    keep = conf >= conf_thr
    if keep.sum() < 4:
        return keep

    for _ in range(iters):
        resid = np.full(n, np.inf, np.float32)
        for i in range(n):
            if not keep[i]:
                continue
            lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
            idx = np.where(keep[lo:hi])[0] + lo
            idx = idx[idx != i]
            if len(idx) < 3:
                resid[i] = 0.0
                continue
            A = np.stack([idx, np.ones_like(idx)], 1).astype(np.float32)
            pred = []
            for d in (0, 1):
                coef, *_ = np.linalg.lstsq(A, xy[idx, d], rcond=None)
                pred.append(coef[0] * i + coef[1])
            resid[i] = float(np.hypot(xy[i, 0] - pred[0], xy[i, 1] - pred[1]))

        acc = np.where(keep)[0]
        if len(acc) < 4:
            break
        steps = np.linalg.norm(np.diff(xy[acc], axis=0), axis=1)
        typ = max(float(np.median(steps)), 4.0)
        new = keep & (resid <= tol * typ)
        if new.sum() < 4 or np.array_equal(new, keep):
            keep = new if new.sum() >= 4 else keep
            break
        keep = new
    return keep

def smoothness(xy):
    if len(xy) < 4:
        return np.nan
    st = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    acc = np.abs(np.diff(st))
    typ = max(float(np.median(st)), 1.0)
    return float((acc < 2 * typ).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="monotrack")
    ap.add_argument("--clips-dir", default="clips/clips")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--conf-thr", type=float, default=0.35)
    ap.add_argument("--clips", default="")
    args = ap.parse_args()

    model, iw, ih = build_model(args.repo, args.weights, args.model)
    clips = args.clips.split(",") if args.clips else []

    print(f"\n{'clip':<36}{'raw>thr':>9}{'kept':>7}{'smooth raw':>12}{'smooth kept':>13}")
    for clip in clips:
        v = os.path.join(args.clips_dir, clip + ".mp4")
        if not os.path.exists(v):
            continue
        got = peaks_for_clip(model, v, args.max_frames, iw, ih)
        if got is None:
            continue
        xy, conf, frames, _ = got
        raw = conf >= args.conf_thr
        keep = trajectory_filter(xy, conf, args.conf_thr)
        print(f"{clip[:35]:<36}{int(raw.sum()):>9}{int(keep.sum()):>7}"
              f"{smoothness(xy[raw]):>12.2f}{smoothness(xy[keep]):>13.2f}")
    print("\n'smooth' = share of steps where the track does not accelerate wildly.")
    print("The filter should raise it; if it does not, it is discarding the wrong points.")

if __name__ == "__main__":
    main()
