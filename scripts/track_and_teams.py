import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], a[:, 2, None], a[:, 3, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.where(union <= 0, 1, union)

def track(frame_idx, xyxy, iou_thr=0.25, max_missed=6):
    order = np.argsort(frame_idx, kind="stable")
    frames = defaultdict(list)
    for row in order:
        frames[int(frame_idx[row])].append(row)

    assign = np.full(len(frame_idx), -1, np.int32)
    tracks = {}
    next_tid, shift = 0, np.zeros(2, np.float32)

    for f in sorted(frames):
        rows = np.array(frames[f])
        boxes = xyxy[rows]
        tids = list(tracks)
        if tids:
            pred = np.stack([tracks[t]["box"] for t in tids])
            pred = pred + np.concatenate([shift, shift])
            M = iou_matrix(pred, boxes)
            ri, ci = linear_sum_assignment(-M)
            deltas, used_t, used_b = [], set(), set()
            for r, c in zip(ri, ci):
                if M[r, c] < iou_thr:
                    continue
                tid = tids[r]
                old = tracks[tid]["box"]
                deltas.append([(boxes[c][0] + boxes[c][2]) / 2 - (old[0] + old[2]) / 2,
                               (boxes[c][1] + boxes[c][3]) / 2 - (old[1] + old[3]) / 2])
                tracks[tid] = dict(box=boxes[c], missed=0)
                assign[rows[c]] = tid
                used_t.add(tid); used_b.add(c)
            shift = (np.median(deltas, axis=0).astype(np.float32)
                     if deltas else np.zeros(2, np.float32))
            for tid in tids:
                if tid not in used_t:
                    tracks[tid]["missed"] += 1
                    if tracks[tid]["missed"] > max_missed:
                        del tracks[tid]
        else:
            used_b = set()

        for c in range(len(boxes)):
            if c in used_b:
                continue
            tracks[next_tid] = dict(box=boxes[c], missed=0)
            assign[rows[c]] = next_tid
            next_tid += 1
    return assign, next_tid

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)

def split_two(X, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    k = min(16, X.shape[0] - 1, X.shape[1])
    Xr = PCA(n_components=k, random_state=seed).fit_transform(X) if k >= 2 else X
    return KMeans(2, n_init=10, random_state=seed).fit_predict(Xr), Xr

def analyse(cache, feature, min_track=2):
    rows = []
    for ef in sorted(glob.glob(os.path.join(cache, "embed", "*.npz"))):
        clip = os.path.basename(ef)[:-4]
        emb = np.load(ef)
        det = np.load(os.path.join(cache, "detect", clip + ".npz"))
        X = np.asarray(emb[feature], np.float32)
        if len(X) < 8:
            continue

        tid_all, n_tracks = track(det["frame_idx"], det["xyxy"])
        tid = tid_all[np.asarray(emb["det_row"], np.int64)]

        lab, _ = split_two(X)

        pure_w, pure_n, tot_crops, split_tracks, n_eval = 0.0, 0, 0, 0, 0
        for t in np.unique(tid):
            m = tid == t
            if m.sum() < min_track:
                continue
            c = np.bincount(lab[m], minlength=2)
            p = c.max() / c.sum()
            pure_w += c.max(); tot_crops += c.sum()
            pure_n += (p == 1.0); n_eval += 1
            split_tracks += (p < 1.0)

        tracks_ok = [t for t in np.unique(tid) if (tid == t).sum() >= min_track]
        if len(tracks_ok) >= 4:
            Tm = unit(np.stack([X[tid == t].mean(0) for t in tracks_ok]))
            tlab, _ = split_two(Tm)
            balance = min(np.bincount(tlab, minlength=2)) / len(tlab)
            track_lab = {t: int(l) for t, l in zip(tracks_ok, tlab)}
        else:
            balance, track_lab = np.nan, {}

        rows.append(dict(
            clip=clip, event_id=clip.split("__a")[0], n_crops=len(X),
            n_tracks=n_tracks, n_eval_tracks=n_eval,
            purity=pure_w / max(tot_crops, 1),
            frac_pure_tracks=pure_n / max(n_eval, 1),
            split_tracks=split_tracks,
            crop_balance=min(np.bincount(lab, minlength=2)) / len(lab),
            track_balance=balance, track_lab=track_lab, tid=tid, lab=lab))
    return rows

def cross_angle(rows, key):
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[r["event_id"]].append(r)
    scores = []
    for ev, rs in by_ev.items():
        if len(rs) < 2:
            continue
        a, b = rs[0][key], rs[1][key]
        if np.isnan(a) or np.isnan(b):
            continue
        scores.append(abs(a - b))
    return np.array(scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/cache")
    ap.add_argument("--features", default="siglip,dinov2,lab")
    ap.add_argument("--write-tracks", action="store_true")
    ap.add_argument("--out", default="cache/cache/track_report.csv")
    args = ap.parse_args()

    all_rows = {}
    print(f"{'feature':<9}{'clips':>7}{'crop purity':>14}{'fully-pure tracks':>20}"
          f"{'split tracks':>15}{'balance':>10}")
    for feat in args.features.split(","):
        rows = analyse(args.cache, feat)
        all_rows[feat] = rows
        pur = np.array([r["purity"] for r in rows])
        fp = np.array([r["frac_pure_tracks"] for r in rows])
        sp = sum(r["split_tracks"] for r in rows)
        bal = np.array([r["track_balance"] for r in rows], float)
        print(f"{feat:<9}{len(rows):>7}{np.mean(pur):>13.1%}{np.mean(fp):>19.1%}"
              f"{sp:>15}{np.nanmean(bal):>10.2f}")

    rows = all_rows[args.features.split(",")[0]]
    nt = np.array([r["n_tracks"] for r in rows])
    ne = np.array([r["n_eval_tracks"] for r in rows])
    print(f"\ntracks per clip: median {np.median(nt):.0f} "
          f"(range {nt.min()}-{nt.max()}); scoreable (>=2 crops) median {np.median(ne):.0f}")
    print("A clip has 10 players plus officials, so a median far above ~15 means "
          "the tracker is fragmenting or picking up crowd.")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "feature", "n_crops", "n_tracks", "n_eval_tracks",
                    "crop_purity", "frac_pure_tracks", "split_tracks",
                    "crop_balance", "track_balance"])
        for feat, rs in all_rows.items():
            for r in rs:
                w.writerow([r["clip"], feat, r["n_crops"], r["n_tracks"],
                            r["n_eval_tracks"], round(r["purity"], 4),
                            round(r["frac_pure_tracks"], 4), r["split_tracks"],
                            round(r["crop_balance"], 3),
                            "" if np.isnan(r["track_balance"]) else round(r["track_balance"], 3)])
    print(f"\nper-clip detail -> {args.out}")

    if args.write_tracks:
        d = os.path.join(args.cache, "tracks")
        os.makedirs(d, exist_ok=True)
        for r in all_rows[args.features.split(",")[0]]:
            np.savez_compressed(os.path.join(d, r["clip"] + ".npz"),
                                crop_track=r["tid"], crop_cluster=r["lab"])
        print(f"tracks -> {d}")

if __name__ == "__main__":
    main()
