import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_and_teams import track, split_two, iou_matrix, unit

LEGS = [13, 14, 15, 16]

def pose_legs_for_dets(det, pose):
    score = np.zeros(len(det["frame_idx"]), np.float32)
    by_frame = defaultdict(list)
    for i, f in enumerate(pose["frame_idx"]):
        by_frame[int(f)].append(i)
    for i, f in enumerate(det["frame_idx"]):
        cand = by_frame.get(int(f), [])
        if not cand:
            continue
        M = iou_matrix(det["xyxy"][i:i + 1], pose["xyxy"][cand])[0]
        j = int(np.argmax(M))
        if M[j] < 0.5:
            continue
        score[i] = float(np.mean(pose["kconf"][cand[j]][LEGS]))
    return score

def oncourt_mask(det, pose, tid, leg_thr=0.5, move_thr=6.0, height_pct=35):
    legs = pose_legs_for_dets(det, pose)

    disp = {}
    for t in np.unique(tid):
        m = tid == t
        if m.sum() < 2:
            disp[t] = 0.0
            continue
        b = det["xyxy"][m]
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        d = np.hypot(np.diff(cx), np.diff(cy))
        disp[t] = float(np.median(d))
    dmed = np.median([v for v in disp.values()]) or 1.0
    moves = np.array([disp[t] > move_thr * 0 + dmed * 0.6 for t in tid])

    h = det["xyxy"][:, 3] - det["xyxy"][:, 1]
    tall = h >= np.percentile(h, height_pct)

    return (legs >= leg_thr) & tall, dict(legs=legs >= leg_thr, tall=tall, moves=moves)

def sheet(thumbs, labels, title, sub, per_row=12, tw=48, th=72, pad=6, head=44):
    W = per_row * (tw + pad)
    img = Image.new("RGB", (W, head + th * 2 + pad * 3), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((6, 8), title, fill=(235, 235, 235))
    d.text((6, 24), sub, fill=(150, 190, 230))
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(per_row, len(idx))).round().astype(int)]
        y = head + c * (th + pad)
        for j, i in enumerate(pick):
            img.paste(Image.fromarray(thumbs[i][..., ::-1]), (j * (tw + pad), y))
        d.text((W - 52, y + 4), "BRIGHT" if c == 0 else "DARK", fill=(255, 255, 255))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/cache")
    ap.add_argument("--feature", default="siglip")
    ap.add_argument("--render", type=int, default=0, help="re-render N sheets")
    ap.add_argument("--outdir", default="cache/cache/review_oncourt")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "embed", "*.npz")))
    stats, rendered = [], 0
    for ef in files:
        clip = os.path.basename(ef)[:-4]
        emb = np.load(ef)
        det = np.load(os.path.join(args.cache, "detect", clip + ".npz"))
        pose = np.load(os.path.join(args.cache, "pose", clip + ".npz"))
        X = np.asarray(emb[args.feature], np.float32)
        if len(X) < 10:
            continue
        tid_all, _ = track(det["frame_idx"], det["xyxy"])
        keep_det, parts = oncourt_mask(det, pose, tid_all)
        rows = np.asarray(emb["det_row"], np.int64)
        keep = keep_det[rows]
        if keep.sum() < 10:
            continue

        lab_before, _ = split_two(X)
        lab_after, _ = split_two(X[keep])
        b0 = min(np.bincount(lab_before, minlength=2)) / len(lab_before)
        b1 = min(np.bincount(lab_after, minlength=2)) / len(lab_after)
        stats.append((clip, len(X), int(keep.sum()), b0, b1))

        if rendered < args.render and "thumbs" in emb.files:
            os.makedirs(args.outdir, exist_ok=True)
            th = emb["thumbs"]
            im = sheet(th[keep], lab_after, clip,
                       f"on-court only: {keep.sum()}/{len(X)} crops kept")
            im.save(os.path.join(args.outdir, clip + ".jpg"), quality=90)
            rendered += 1

    kept = np.array([s[2] / s[1] for s in stats])
    b0 = np.array([s[3] for s in stats]); b1 = np.array([s[4] for s in stats])
    print(f"{len(stats)} clips")
    print(f"crops kept        {kept.mean():.0%} (median {np.median(kept):.0%})")
    print(f"cluster balance   {b0.mean():.3f} -> {b1.mean():.3f}   "
          f"(0.50 is a perfect 5-on-5 split)")
    print(f"clips improving   {(b1 > b0).sum()}/{len(stats)}")
    if args.render:
        print(f"\nre-rendered {rendered} sheets -> {args.outdir}")

if __name__ == "__main__":
    main()
