import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_index import load_index, cached, HUE, ACHROMATIC
from jersey_desc import descriptor, cluster, HUE_BINS

VARIANTS = [("hue2", "gmm"), ("hue2", "kmeans"),
            ("hue2_skin", "gmm"), ("hue", "gmm"), ("lab", "gmm")]

def colour_template(words):
    v = np.zeros(HUE_BINS + 2, np.float32)
    for w in words:
        if w == "white":
            v[HUE_BINS] += 1.0
        elif w in ACHROMATIC:
            v[HUE_BINS + 1] += 1.0
        elif w in HUE:
            b = int(HUE[w] * HUE_BINS) % HUE_BINS
            v[b] += 1.0
            v[(b - 1) % HUE_BINS] += 0.35
            v[(b + 1) % HUE_BINS] += 0.35
    s = v.sum()
    return v / s if s > 0 else v

def cos(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0

def eval_clip(crops, clip, kind, method, per_track=False, tracks=None):
    X = descriptor(crops, kind)
    if per_track and tracks is not None and len(np.unique(tracks)) >= 4:
        uniq = np.unique(tracks)
        tl = cluster(np.stack([X[tracks == t].mean(0) for t in uniq]), method)
        m = {t: int(l) for t, l in zip(uniq, tl)}
        lab = np.array([m[t] for t in tracks])
    else:
        lab = cluster(X, method)
    if len(set(lab.tolist())) < 2:
        return None
    C = descriptor(crops, "hue2")
    G = np.stack([C[lab == k].mean(0) for k in (0, 1)])

    Ta = colour_template(clip.components(clip.away))
    Th = colour_template(clip.components(clip.home))
    if Ta.sum() == 0 or Th.sum() == 0:
        return None
    p0 = cos(G[0], Ta) > cos(G[0], Th)
    p1 = cos(G[1], Ta) > cos(G[1], Th)
    correct = p0 != p1
    straight = cos(G[0], Ta) + cos(G[1], Th)
    swapped = cos(G[0], Th) + cos(G[1], Ta)
    margin = abs(straight - swapped) / 2
    mapping = (clip.away, clip.home) if straight >= swapped else (clip.home, clip.away)
    return correct, margin, mapping, lab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-track", action="store_true")
    ap.add_argument("--out", default="cache/cache/day1_team_accuracy.csv")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cropdir = os.path.join(base, "cache", "cache", "crops")
    idx = load_index(base)
    clips = sorted([c for c in idx.values() if c.complete
                    and os.path.exists(os.path.join(cropdir, c.cid + ".npz"))],
                   key=lambda c: c.cid)
    print(f"{len(clips)} clips with colours and cached crops\n")
    if not clips:
        sys.exit("no crop cache yet -- run scripts/cache_crops.py --all")

    data = {}
    for c in clips:
        z = np.load(os.path.join(cropdir, c.cid + ".npz"))
        data[c.cid] = (z["crops"], z["track"])

    print(f"{'descriptor':<12}{'clust':<8}{'correct':>10}{'accuracy':>11}"
          f"{'95% CI':>16}{'margin':>9}")
    best_rows = None
    for kind, method in VARIANTS:
        ok = tot = 0
        margins, rows = [], []
        for c in clips:
            crops, tracks = data[c.cid]
            r = eval_clip(crops, c, kind, method, args.per_track, tracks)
            if r is None:
                continue
            correct, margin, mapping, lab = r
            tot += 1
            ok += correct
            margins.append(margin)
            rows.append(dict(clip=c.cid, away=c.away, home=c.home,
                             away_colour=c.colours[c.away], home_colour=c.colours[c.home],
                             correct=int(correct), margin=round(margin, 4),
                             group0_team=mapping[0], n_crops=len(crops)))
        acc = ok / max(tot, 1)
        se = (acc * (1 - acc) / max(tot, 1)) ** 0.5
        print(f"{kind:<12}{method:<8}{ok:>4}/{tot:<5}{acc:>10.1%}"
              f"   [{max(0,acc-1.96*se):>5.0%},{min(1,acc+1.96*se):>5.0%}]"
              f"{np.mean(margins):>9.3f}")
        if best_rows is None:
            best_rows = rows

    if best_rows:
        out = os.path.join(base, args.out)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(best_rows[0]))
            w.writeheader(); w.writerows(best_rows)
        wrong = [r for r in best_rows if not r["correct"]]
        print(f"\nper-clip -> {args.out}")
        print(f"{len(wrong)} failures under {VARIANTS[0][0]}/{VARIANTS[0][1]}; "
              f"most common colour pairs among them:")
        import collections
        cc = collections.Counter(tuple(sorted((r["away_colour"], r["home_colour"])))
                                 for r in wrong)
        for pair, n in cc.most_common(6):
            print(f"   {pair[0]:<20} vs {pair[1]:<20}{n}")

if __name__ == "__main__":
    main()
