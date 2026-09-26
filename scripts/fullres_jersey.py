import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_and_teams import track
from oncourt_filter import oncourt_mask
from jersey_desc import (descriptor, cluster, sharpness, order_by_lightness,
                         cluster_k3_drop_junk)

TORSO_X, TORSO_Y = (0.20, 0.80), (0.15, 0.55)
CROP = (48, 72)

def crops_for_clip(video, det, keep, max_frames=None):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None, None

    out, rows = [], []
    H, W = frames[0].shape[:2]
    for i in np.where(keep)[0]:
        fi = int(det["frame_idx"][i])
        if fi >= len(frames):
            continue
        x1, y1, x2, y2 = det["xyxy"][i]
        w, h = x2 - x1, y2 - y1
        cx1 = int(max(0, x1 + TORSO_X[0] * w)); cx2 = int(min(W, x1 + TORSO_X[1] * w))
        cy1 = int(max(0, y1 + TORSO_Y[0] * h)); cy2 = int(min(H, y1 + TORSO_Y[1] * h))
        if cx2 - cx1 < 10 or cy2 - cy1 < 14:
            continue
        patch = frames[fi][cy1:cy2, cx1:cx2]
        out.append(cv2.resize(patch, CROP, interpolation=cv2.INTER_AREA))
        rows.append(i)
    del frames
    if not out:
        return None, None
    return np.stack(out), np.array(rows)

def sheet(thumbs, labels, title, sub, per_row=13, tw=44, th=66, pad=4, head=32):
    W = per_row * (tw + pad)
    img = Image.new("RGB", (W, head + th * 2 + pad * 3), (22, 22, 22))
    d = ImageDraw.Draw(img)
    d.text((6, 6), title, fill=(235, 235, 235))
    d.text((6, 19), sub, fill=(150, 190, 230))
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(per_row, len(idx))).round().astype(int)]
        y = head + c * (th + pad)
        for j, i in enumerate(pick):
            img.paste(Image.fromarray(thumbs[i][..., ::-1]).resize((tw, th)),
                      (j * (tw + pad), y))
        d.text((W - 48, y + 3), "LIGHT" if c == 0 else "DARK", fill=(255, 255, 255))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/cache")
    ap.add_argument("--clips-dir", default="clips/clips")
    ap.add_argument("--outdir", default="cache/cache/review_final")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sharp-pct", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--k3", action="store_true",
                    help="fit three groups and discard the non-jersey one")
    args = ap.parse_args()

    names = [os.path.basename(f)[:-4]
             for f in sorted(glob.glob(os.path.join(args.cache, "detect", "*.npz")))]
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]
    elif not args.all:
        names = names[:args.limit or 8]

    os.makedirs(args.outdir, exist_ok=True)
    rows_out, t0 = [], time.time()
    for k, clip in enumerate(names, 1):
        video = os.path.join(args.clips_dir, clip + ".mp4")
        if not os.path.exists(video):
            continue
        det = np.load(os.path.join(args.cache, "detect", clip + ".npz"))
        pose = np.load(os.path.join(args.cache, "pose", clip + ".npz"))
        tid_all, _ = track(det["frame_idx"], det["xyxy"])
        keep, _ = oncourt_mask(det, pose, tid_all)
        if keep.sum() < 12:
            keep = np.ones(len(det["frame_idx"]), bool)

        th, rows = crops_for_clip(video, det, keep)
        if th is None or len(th) < 12:
            continue
        sh = sharpness(th)
        good = sh >= np.percentile(sh, args.sharp_pct)
        if good.sum() >= 12:
            th, rows = th[good], rows[good]

        X = descriptor(th, "hue2")
        if args.k3:
            lab, keep3 = cluster_k3_drop_junk(X, th)
            th, rows, X = th[keep3], rows[keep3], X[keep3]
        else:
            lab = cluster(X, "gmm")
        lab = order_by_lightness(lab, th)

        c0, c1 = X[lab == 0].mean(0), X[lab == 1].mean(0)
        sep = float(1 - (c0 @ c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))

        eid = clip.split("__a")[0]
        sheet(th, lab, clip,
              f"{len(th)} full-res crops from {len(np.unique(det['frame_idx']))} frames"
              f"   separation {sep:.3f}"
              ).save(os.path.join(args.outdir, clip + ".jpg"), quality=92)
        rows_out.append(dict(clip=clip, event_id=eid, n_crops=len(th),
                             n_boxes=int(keep.sum()), separation=round(sep, 4),
                             n_light=int((lab == 0).sum()), n_dark=int((lab == 1).sum())))
        np.savez_compressed(os.path.join(args.cache, "fullres", clip + ".npz")
                            if os.path.isdir(os.path.join(args.cache, "fullres"))
                            else _mk(args.cache, clip), labels=lab, det_row=rows, desc=X)
        print(f"[{k}/{len(names)}] {clip:<44} {len(th):>5} crops  sep {sep:.3f}")

    if rows_out:
        p = os.path.join(args.cache, "fullres_report.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"\n{len(rows_out)} clips in {(time.time()-t0)/60:.1f} min -> {args.outdir}")
        print(f"report -> {p}")

def _mk(cache, clip):
    d = os.path.join(cache, "fullres")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, clip + ".npz")

if __name__ == "__main__":
    main()
