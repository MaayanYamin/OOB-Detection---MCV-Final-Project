import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, ROOT, load_items, preflight, score

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from track_and_teams import track

CONF = 0.35
KCONF = 0.30
NEAR = 0.40
FAR = 0.80
STORE = 64
OUT = os.path.join(CACHE, "ribbon")

def pick(n, k=STORE):
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))

def player_height(det, fallback=400.0):
    hh = det["xyxy"][:, 3] - det["xyxy"][:, 1]
    if len(hh) < 5:
        return fallback
    return float(np.median(hh[hh >= np.percentile(hh, 70)]))

def det_teams(cid, det):
    sign = np.zeros(len(det["frame_idx"]), np.int8)
    p = os.path.join(CACHE, "team_assign", cid + ".npz")
    if not os.path.exists(p):
        return sign
    z = np.load(p)
    tid, _ = track(det["frame_idx"], det["xyxy"])
    team_of = {int(t): (1 if z["team"][z["track"] == t].mean() > 0.5 else -1)
               for t in np.unique(z["track"])}
    for i, t in enumerate(tid):
        sign[i] = team_of.get(int(t), 0)
    sign[z["det_row"]] = np.where(z["team"] == 1, 1, -1)
    return sign

def iou(a, b):
    x0 = np.maximum(a[0], b[:, 0])
    y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2])
    y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda q: np.clip(q[..., 2] - q[..., 0], 0, None) * np.clip(q[..., 3] - q[..., 1], 0, None)
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)

def build(cid, n_frames=None):
    dp = os.path.join(CACHE, "detect", cid + ".npz")
    pp = os.path.join(CACHE, "pose", cid + ".npz")
    bp = os.path.join(CACHE, "ball", cid + ".npz")
    for p, stage in ((dp, "detect"), (pp, "pose")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"no {stage} cache for {cid}")
    det, pose = np.load(dp), np.load(pp)
    n = int(n_frames or max(int(det["frame_idx"].max()), int(pose["frame_idx"].max())) + 1)

    att = np.zeros(n, np.float32)
    m_att = np.zeros(n, np.uint8)
    team = np.zeros(n, np.float32)
    m_team = np.zeros(n, np.uint8)
    meta = {"n": n, "ball_frames": 0, "player_height": player_height(det)}

    if not os.path.exists(bp):
        return att, m_att, team, m_team, meta
    ball = np.load(bp)
    conf = np.nan_to_num(ball["conf"], nan=0.0)
    good = ball["keep"] & (conf >= CONF)
    balls = {int(f): ball["xy"][i] for i, f in enumerate(ball["frame_idx"])
             if good[i] and 0 <= f < n and np.all(np.isfinite(ball["xy"][i]))}
    meta["ball_frames"] = len(balls)
    if not balls:
        return att, m_att, team, m_team, meta

    ph = meta["player_height"]
    sign = det_teams(cid, det)
    kp, kc = pose["kpts"][..., :2], pose["kconf"]

    for f, xy in balls.items():
        rows = np.where(pose["frame_idx"] == f)[0]
        if not len(rows):
            continue
        best_d, best_row = np.inf, -1
        for r in rows:
            for k in (9, 10):
                if kc[r, k] <= KCONF or not np.all(np.isfinite(kp[r, k])):
                    continue
                d = float(np.hypot(*(kp[r, k] - xy)))
                if d < best_d:
                    best_d, best_row = d, int(r)
        if best_row < 0:
            continue
        d = best_d / max(ph, 1.0)
        if d <= NEAR:
            att[f], m_att[f] = 1.0, 1
        elif d >= FAR:
            att[f], m_att[f] = 0.0, 1
        else:
            continue
        if att[f] < 0.5:
            continue
        drows = np.where(det["frame_idx"] == f)[0]
        if not len(drows):
            continue
        j = int(np.argmax(iou(pose["xyxy"][best_row], det["xyxy"][drows])))
        if iou(pose["xyxy"][best_row], det["xyxy"][drows])[j] < 0.3:
            continue
        s = int(sign[drows[j]])
        if s == 0:
            continue
        team[f], m_team[f] = float(s > 0), 1
    return att, m_att, team, m_team, meta

def store(cid, n_frames=None):
    att, m_att, team, m_team, meta = build(cid, n_frames)
    idx = pick(meta["n"])
    return dict(frames=idx.astype(np.int32), att=att[idx], m_att=m_att[idx],
                team=team[idx], m_team=m_team[idx], n=np.int32(meta["n"]))

def path(cid):
    return os.path.join(OUT, cid + ".npz")

def load(cid, n_store):
    z0 = np.zeros(n_store, np.float32)
    m0 = np.zeros(n_store, np.float32)
    p = path(cid)
    if not os.path.exists(p):
        return z0, m0, z0.copy(), m0.copy()
    with open(p, "rb") as fh:
        z = np.load(fh)
        out = []
        for k in ("att", "m_att", "team", "m_team"):
            a = z[k].astype(np.float32)
            if len(a) != n_store:
                a = np.interp(np.linspace(0, 1, n_store),
                              np.linspace(0, 1, len(a)), a) if len(a) else z0.copy()
                if k.startswith("m_"):
                    a = (a > 0.99).astype(np.float32)
            out.append(a)
    return tuple(out)

def prep(force=False, report_only=False, limit=None):
    os.makedirs(OUT, exist_ok=True)
    items = load_items()[:limit] if limit else load_items()
    rows, done, failed = [], 0, 0
    for i, it in enumerate(items, 1):
        try:
            if report_only or force or not os.path.exists(path(it.cid)):
                d = store(it.cid)
                if not report_only:
                    with open(path(it.cid), "wb") as fh:
                        np.savez(fh, **d)
                    done += 1
            else:
                with open(path(it.cid), "rb") as fh:
                    d = {k: v for k, v in np.load(fh).items()}
            rows.append((it.cid, float(d["m_att"].mean()), float(d["m_team"].mean()),
                         float(d["att"][d["m_att"] > 0].mean()) if d["m_att"].sum() else np.nan))
        except Exception as e:
            failed += 1
            print(f"  FAILED {it.cid}: {type(e).__name__}: {e}", flush=True)
        if i % 25 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] ribbon: {done} written, {failed} failed", flush=True)
    if rows:
        a = np.array([r[1] for r in rows])
        b = np.array([r[2] for r in rows])
        h = np.array([r[3] for r in rows])
        print(f"\nribbon coverage over {len(rows)} clips")
        print(f"  frames with an 'attached / in flight' call : "
              f"median {np.median(a):.0%}, mean {a.mean():.0%}, "
              f"{int((a == 0).sum())} clips with none")
        print(f"  frames with a team as well                 : "
              f"median {np.median(b):.0%}, mean {b.mean():.0%}, "
              f"{int((b == 0).sum())} clips with none")
        print(f"  of the called frames, share 'attached'     : "
              f"median {np.nanmedian(h):.0%}")
        print(f"\nfor comparison, the clip label supplies exactly 1 value per clip; "
              f"the ribbon supplies {b.mean() * STORE:.0f} on average.")
    return failed == 0

def baseline(limit=None):
    items = preflight(load_items())[:limit] if limit else preflight(load_items())
    p = np.full(len(items), np.nan)
    have = 0
    for i, it in enumerate(items):
        pth = path(it.cid)
        if not os.path.exists(pth):
            continue
        with open(pth, "rb") as fh:
            z = np.load(fh)
            team, m = z["team"], z["m_team"]
        j = np.where(m > 0)[0]
        if not len(j):
            continue
        p[i] = float(team[j[-1]])
        have += 1
    if not have:
        print("no clip has a team-resolved frame - nothing to score")
        return False
    s = score(items, p)
    for level in ("clip", "event"):
        c = s[level]
        print(f"ribbon last-touch baseline, {level} level: acc {c['acc']:.1%} (n={c['n']}, "
              f"decided on {have}/{len(items)} clips)  lift vs constant "
              f"{c['constant']['lift']:+.1%} [{c['constant']['ci'][0]:+.1%}, "
              f"{c['constant']['ci'][1]:+.1%}]")
    print("\nA lift whose interval does not clear zero is not a result (RESULTS.md 4d).")
    return True

def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = bool(np.allclose(got, want))
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: got {got}, want {want}")

    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    check("iou self / half-overlap / disjoint", np.round(iou(a, b), 4),
          [1.0, round(25 / 175, 4), 0.0])
    check("pick(8, 4) is evenly spaced", pick(8, 4), [0, 2, 5, 7])
    check("pick keeps the last frame", pick(78)[-1], 77)
    check("pick returns at most STORE rows", len(pick(500)), STORE)
    check("NEAR below FAR", NEAR < FAR, True)
    d = np.array([0.20, 0.40, 0.60, 0.80, 1.20])
    called = (d <= NEAR) | (d >= FAR)
    check("0.6 player-heights is refused, not guessed", called, [True, True, False, True, True])
    print("\nself-test:", "passed" if ok else "FAILED")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="coverage table only, write nothing")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="score 'the team holding the ball last' - no training")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.baseline:
        sys.exit(0 if baseline(a.limit) else 1)
    sys.exit(0 if prep(a.force, a.report, a.limit) else 1)
