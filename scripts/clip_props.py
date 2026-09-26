import csv
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "cache")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "finetune"))
from clip_index import ACHROMATIC, load_index
from common import load_items, parts

CONF = 0.35

def video_meta(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n

def npz(stage, cid):
    p = os.path.join(CACHE, stage, cid + ".npz")
    return np.load(p) if os.path.exists(p) else None

def main():
    items = {i.cid: i for i in load_items()}
    idx = load_index(ROOT)
    man = {r["event_id"]: r for r in
           csv.DictReader(open(os.path.join(ROOT, "manifest.csv"), encoding="utf-8-sig"))}
    per_event = Counter(i.event for i in items.values())
    rows = []
    for cid, it in items.items():
        w, h, fps, nf = video_meta(it.path)
        m = man.get(it.event, {})
        cols = [c for t in (it.col_home, it.col_away) for c in parts(t)]
        r = dict(clip=cid, arena=it.arena, event=it.event, game=it.game,
                 label=("HOME" if it.y_home else "AWAY"),
                 width=w, height=h, fps=round(fps, 2), n_frames=nf,
                 duration_s=round(nf / fps, 2) if fps else "",
                 size_mb=round(os.path.getsize(it.path) / 2**20, 1),
                 period=m.get("period", ""), phase=m.get("phase", ""),
                 angles_for_event=per_event[it.event],
                 jersey_contrast=("achromatic_pair" if any(c in ACHROMATIC for c in cols)
                                  else "both_chromatic"),
                 col_home=it.col_home, col_away=it.col_away)
        d = npz("detect", cid)
        if d is not None and len(d["frame_idx"]):
            box_h = d["xyxy"][:, 3] - d["xyxy"][:, 1]
            cx = (d["xyxy"][:, 0] + d["xyxy"][:, 2]) / 2
            cy = (d["xyxy"][:, 1] + d["xyxy"][:, 3]) / 2
            per_f = Counter(d["frame_idx"].tolist())
            mids = np.array([[np.median(cx[d["frame_idx"] == f]), np.median(cy[d["frame_idx"] == f])]
                             for f in sorted(per_f)])
            r["players_per_frame"] = round(float(np.mean(list(per_f.values()))), 2)
            r["player_height_frac"] = round(float(np.median(box_h) / h), 4) if h else ""
            r["camera_drift_px"] = (round(float(np.median(np.hypot(*np.diff(mids, axis=0).T))), 2)
                                    if len(mids) > 1 else "")
            r["frames_with_people"] = len(per_f)
        b = npz("ball", cid)
        if b is not None:
            good = b["keep"] & (b["conf"] >= CONF)
            r["ball_rate"] = round(float(good.mean()), 3)
            r["ball_hits"] = int(good.sum())
        t = npz("team_assign", cid)
        if t is not None and "margin" in t.files and len(t["margin"]):
            r["team_margin"] = round(float(np.mean(t["margin"])), 4)
            r["team_balance"] = round(float(np.mean(t["team"] == 1)), 3)
        rows.append(r)
    cols = ["clip", "arena", "event", "game", "label", "width", "height", "fps", "n_frames",
            "duration_s", "size_mb", "period", "phase", "angles_for_event", "jersey_contrast",
            "col_home", "col_away", "players_per_frame", "player_height_frac",
            "camera_drift_px", "frames_with_people", "ball_rate", "ball_hits",
            "team_margin", "team_balance"]
    out = os.path.join(CACHE, "clip_props.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in sorted(rows, key=lambda r: r["clip"]):
            wr.writerow({c: r.get(c, "") for c in cols})
    have = sum(1 for r in rows if "players_per_frame" in r)
    print(f"{len(rows)} clips -> {out}")
    print(f"  with detect cache: {have}   with ball: {sum(1 for r in rows if 'ball_rate' in r)}"
          f"   with team: {sum(1 for r in rows if 'team_margin' in r)}")
    d = [r["duration_s"] for r in rows if r["duration_s"]]
    print(f"  duration: min {min(d):.1f}s  median {sorted(d)[len(d)//2]:.1f}s  max {max(d):.1f}s")
    print(f"  resolutions: {dict(Counter((r['width'], r['height']) for r in rows))}")

if __name__ == "__main__":
    main()
