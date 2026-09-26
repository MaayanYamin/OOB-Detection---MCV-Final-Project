import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_index import load_index
from ball_wasb_cpu import build_model
from ball_track import peaks_for_clip
from ball_kalman import kalman_filter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="monotrack")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="cap per clip; clips are 45-180 frames so 200 means all")
    ap.add_argument("--conf-thr", type=float, default=0.35)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(base, "cache", "cache", "ball")
    os.makedirs(outdir, exist_ok=True)

    idx = load_index(base)
    clips = sorted(idx.values(), key=lambda c: c.cid)
    if args.limit:
        clips = clips[:args.limit]

    model, iw, ih = build_model(args.repo, args.weights, args.model)
    t0, done, skipped, failed = time.time(), 0, 0, []

    for i, c in enumerate(clips, 1):
        out = os.path.join(outdir, c.cid + ".npz")
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue
        if not os.path.exists(c.path):
            failed.append(c.cid)
            continue
        try:
            got = peaks_for_clip(model, c.path, args.max_frames, iw, ih)
            if got is None:
                failed.append(c.cid)
                continue
            xy, conf, frames, _ = got
            fidx = np.arange(len(xy), dtype=np.int32)
            keep, innov = kalman_filter(fidx, xy, conf, args.conf_thr)
            np.savez_compressed(out, frame_idx=fidx, xy=xy.astype(np.float32),
                                conf=conf.astype(np.float32), keep=keep,
                                innov=np.asarray(innov, np.float32))
            done += 1
            hit = (conf >= args.conf_thr).mean()
            print(f"[{i}/{len(clips)}] {c.cid[:38]:<40} {len(xy):>4}f  "
                  f"fires {hit:>4.0%}  kept {int(keep.sum()):>4}  "
                  f"{(time.time()-t0)/60:>5.1f}m", flush=True)
        except Exception as e:
            failed.append(f"{c.cid}: {type(e).__name__}")

    print(f"\n{done} written, {skipped} cached, {len(failed)} failed "
          f"in {(time.time()-t0)/60:.1f} min")
    if failed:
        print("failed:", ", ".join(failed[:8]))

if __name__ == "__main__":
    main()
