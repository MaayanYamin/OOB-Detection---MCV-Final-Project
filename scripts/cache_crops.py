import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_index import load_index, cached
from track_and_teams import track
from oncourt_filter import oncourt_mask
from jersey_desc import sharpness
from fullres_jersey import crops_for_clip

SHARP_DROP = 20

def build(base, clip, max_crops=400, sharp_pct=SHARP_DROP):
    d, p = cached(base, "detect", clip.cid), cached(base, "pose", clip.cid)
    if not (d and p and os.path.exists(clip.path)):
        return None
    det, pose = np.load(d), np.load(p)
    tid_all, _ = track(det["frame_idx"], det["xyxy"])
    keep, _ = oncourt_mask(det, pose, tid_all)
    if keep.sum() < 12:
        keep = np.ones(len(det["frame_idx"]), bool)

    th, rows = crops_for_clip(clip.path, det, keep)
    if th is None or len(th) < 12:
        return None
    sh = sharpness(th)
    g = sh >= np.percentile(sh, sharp_pct)
    if g.sum() >= 12:
        th, rows = th[g], rows[g]

    if len(th) > max_crops:
        sel = np.linspace(0, len(th) - 1, max_crops).round().astype(int)
        th, rows = th[sel], rows[sel]

    return dict(crops=th.astype(np.uint8), det_row=rows.astype(np.int32),
                track=tid_all[rows].astype(np.int32),
                frame_idx=det["frame_idx"][rows].astype(np.int32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-crops", type=int, default=400)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(base, "cache", "cache", "crops")
    os.makedirs(outdir, exist_ok=True)
    idx = load_index(base)
    clips = sorted([c for c in idx.values()
                    if cached(base, "detect", c.cid) and cached(base, "pose", c.cid)],
                   key=lambda c: c.cid)
    if not args.all:
        clips = clips[:args.limit]

    t0, done, skipped, failed = time.time(), 0, 0, []
    for i, c in enumerate(clips, 1):
        out = os.path.join(outdir, c.cid + ".npz")
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue
        r = build(base, c, args.max_crops)
        if r is None:
            failed.append(c.cid)
            continue
        np.savez_compressed(out, **r)
        done += 1
        if done % 10 == 0 or i == len(clips):
            print(f"[{i}/{len(clips)}] {done} written, {skipped} cached, "
                  f"{len(failed)} failed, {(time.time()-t0)/60:.1f} min")
    print(f"\n{done} written, {skipped} already cached, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed[:8]))
    tot = sum(os.path.getsize(os.path.join(outdir, f))
              for f in os.listdir(outdir)) / 1048576
    print(f"crop cache: {len(os.listdir(outdir))} clips, {tot:.0f} MB")

if __name__ == "__main__":
    main()
