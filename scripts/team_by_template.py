import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_index import load_index
from jersey_desc import descriptor
from score_variants_truth import colour_template, cos

def assign(crops, clip, tracks=None, per_track=True):
    X = descriptor(crops, "hue2")
    Ta = colour_template(clip.components(clip.away))
    Th = colour_template(clip.components(clip.home))
    if Ta.sum() == 0 or Th.sum() == 0:
        return None, None
    sa = np.array([cos(x, Ta) for x in X], np.float32)
    sh = np.array([cos(x, Th) for x in X], np.float32)

    if per_track and tracks is not None and len(np.unique(tracks)) >= 2:
        out = np.zeros(len(X), np.int8)
        marg = np.zeros(len(X), np.float32)
        for t in np.unique(tracks):
            m = tracks == t
            d = float(sa[m].mean() - sh[m].mean())
            out[m] = 0 if d > 0 else 1
            marg[m] = abs(d)
        return out, marg
    return (sa <= sh).astype(np.int8), np.abs(sa - sh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-track", action="store_true", default=True)
    ap.add_argument("--out", default="cache/cache/team_template.npz")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cropdir = os.path.join(base, "cache", "cache", "crops")
    idx = load_index(base)
    clips = sorted([c for c in idx.values() if c.complete
                    and os.path.exists(os.path.join(cropdir, c.cid + ".npz"))],
                   key=lambda c: c.cid)

    outdir = os.path.join(base, "cache", "cache", "team_assign")
    os.makedirs(outdir, exist_ok=True)

    balances, margins, degenerate = [], [], []
    for c in clips:
        z = np.load(os.path.join(cropdir, c.cid + ".npz"))
        team, marg = assign(z["crops"], c, z["track"], args.per_track)
        if team is None:
            continue
        np.savez_compressed(os.path.join(outdir, c.cid + ".npz"),
                            team=team, margin=marg, det_row=z["det_row"],
                            track=z["track"], frame_idx=z["frame_idx"],
                            away=c.away, home=c.home)
        frac = float((team == 0).mean())
        balances.append(frac)
        margins.append(float(marg.mean()))
        if frac < 0.02 or frac > 0.98:
            degenerate.append((c.cid, frac, c.colours[c.away], c.colours[c.home]))

    b = np.array(balances)
    print(f"{len(balances)} clips assigned by colour template\n")
    print(f"share of crops given to the away team: median {np.median(b):.2f}, "
          f"mean {b.mean():.2f}")
    print(f"clips where one team took >98% of crops: {len(degenerate)}")
    for cid, f, ca, ch in degenerate[:10]:
        print(f"   {cid[:34]:<36}{f:>5.2f}   {ca} vs {ch}")
    print(f"\nmean per-crop margin: {np.mean(margins):.3f}")
    print("\nA degenerate clip means the templates could not tell the uniforms apart")
    print("on that footage -- the equivalent of a collapsed cluster, and the number")
    print("to compare against clustering's 35 failures.")
    print(f"\nwritten -> cache/cache/team_assign/")

if __name__ == "__main__":
    main()
