import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_wasb_cpu import build_model
from ball_track import peaks_for_clip, smoothness

def kalman_pass(frames_idx, xy, order, q=90.0, r=36.0, gate=9.0, p0=1e4):
    n = len(xy)
    acc = np.zeros(n, bool)
    innov = np.full(n, np.nan, np.float32)

    x = None
    P = None
    t_prev = None

    for i in order:
        t = float(frames_idx[i])
        z = xy[i].astype(np.float64)

        if x is None:
            x = np.array([z[0], z[1], 0.0, 0.0])
            P = np.eye(4) * p0
            acc[i] = True
            innov[i] = 0.0
            t_prev = t
            continue

        dt = abs(t - t_prev)
        if dt == 0:
            dt = 1.0
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        Q = np.eye(4) * q * dt
        Q[2, 2] = Q[3, 3] = q * dt * 0.5
        xp = F @ x
        Pp = F @ P @ F.T + Q

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * r
        y = z - H @ xp
        S = H @ Pp @ H.T + R
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            d2 = np.inf
        innov[i] = d2

        if d2 <= gate:
            K = Pp @ H.T @ np.linalg.inv(S)
            x = xp + K @ y
            P = (np.eye(4) - K @ H) @ Pp
            acc[i] = True
            t_prev = t
        else:
            x, P, t_prev = xp, Pp, t
    return acc, innov

def kalman_filter(frames_idx, xy, conf, conf_thr=0.35, **kw):
    cand = np.where(conf >= conf_thr)[0]
    keep = np.zeros(len(xy), bool)
    innov = np.full(len(xy), np.nan, np.float32)
    if len(cand) < 3:
        keep[cand] = True
        return keep, innov
    fwd, i_f = kalman_pass(frames_idx, xy, cand, **kw)
    bwd, i_b = kalman_pass(frames_idx, xy, cand[::-1], **kw)
    keep = fwd & bwd
    innov = np.fmin(np.nan_to_num(i_f, nan=np.inf), np.nan_to_num(i_b, nan=np.inf))
    innov[~np.isfinite(innov)] = np.nan
    if keep.sum() < 3:
        keep = fwd | bwd
    return keep, innov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="monotrack")
    ap.add_argument("--clips-dir", default="clips/clips")
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--conf-thr", type=float, default=0.35)
    args = ap.parse_args()

    model, iw, ih = build_model(args.repo, args.weights, args.model)
    print(f"\n{'clip':<36}{'raw':>6}{'kept':>6}{'smooth raw':>12}{'smooth kept':>13}")
    for clip in args.clips.split(","):
        v = os.path.join(args.clips_dir, clip + ".mp4")
        if not os.path.exists(v):
            continue
        got = peaks_for_clip(model, v, args.max_frames, iw, ih)
        if got is None:
            continue
        xy, conf, frames, _ = got
        fidx = np.arange(len(xy))
        raw = conf >= args.conf_thr
        keep, innov = kalman_filter(fidx, xy, conf, args.conf_thr)
        print(f"{clip[:35]:<36}{int(raw.sum()):>6}{int(keep.sum()):>6}"
              f"{smoothness(xy[raw]):>12.2f}{smoothness(xy[keep]):>13.2f}")
    print("\nInnovation spikes on the accepted track are the touch candidates:")
    print("a contact makes the ball violate constant velocity, which is exactly what")
    print("the filter measures.")

if __name__ == "__main__":
    main()
