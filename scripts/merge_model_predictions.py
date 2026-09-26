import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "predictions.csv")

def main():
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "finetune", "results", "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("protocol") not in (None, "game"):
            continue
        runs[f"{r['model']}_{r['input']}"].append(r)

    fresh = []
    for name, rs in sorted(runs.items()):
        per_clip = defaultdict(list)
        for r in rs:
            for c, p in zip(r["clips"], r["p_home"]):
                if p is not None:
                    per_clip[c].append(float(p))
        for c, vals in per_clip.items():
            fresh.append((c, name, float(np.mean(vals))))
        print(f"  {name:<22}{len(rs)} seeds, {len(per_clip)} clips")

    old = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            old = list(csv.reader(fh))[1:]
    names = {n for _, n, _ in fresh}
    keep = [(r[0], r[1], float(r[2])) for r in old if r[1] not in names]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "method", "p_home", "pred"])
        for c, m, v in sorted(keep + fresh):
            w.writerow([c, m, f"{v:.4f}", int(v > 0.5)])
    print(f"\n{len(fresh)} model predictions merged, {len(keep)} kept -> {OUT}")

if __name__ == "__main__":
    main()
