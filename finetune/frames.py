import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, ROOT, colour_code, load_items, question

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from track_and_teams import track

STORE = 64
RAW_S = 224
BALL_S = 256
SIDE_MULT = 2.5
CONF = 0.35
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LIME, MAGENTA, CYAN = (0, 255, 0), (255, 0, 255), (0, 255, 255)

def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames

def pick(n, k=STORE):
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))

def letterbox(img, s):
    h, w = img.shape[:2]
    r = s / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.zeros((s, s, 3), np.uint8)
    y0, x0 = (s - nh) // 2, (s - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def blob(ch, u, v, sigma):
    s = ch.shape[0]
    if not (0 <= u < s and 0 <= v < s):
        return
    yy, xx = np.ogrid[:s, :s]
    g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
    np.maximum(ch, (g * 255).astype(np.uint8), out=ch)

def smooth(x, w=9):
    if len(x) < w:
        return x
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")

def ball_track(cid, n):
    p = os.path.join(CACHE, "ball", cid + ".npz")
    if not os.path.exists(p):
        return {}
    z = np.load(p)
    good = z["keep"] & (z["conf"] >= CONF)
    return {int(f): z["xy"][i] for i, f in enumerate(z["frame_idx"]) if good[i] and f < n}

def centres(balls, det, n, w, h):
    if len(balls) >= 2:
        f = np.array(sorted(balls))
        xy = np.array([balls[k] for k in f], float)
    else:
        fr, cx, cy = [], [], []
        for k in np.unique(det["frame_idx"]):
            b = det["xyxy"][det["frame_idx"] == k]
            fr.append(k)
            cx.append(np.median((b[:, 0] + b[:, 2]) / 2))
            cy.append(np.median((b[:, 1] + b[:, 3]) / 2))
        if len(fr) < 2:
            return np.tile([w / 2, h / 2], (n, 1))
        f, xy = np.array(fr), np.stack([cx, cy], 1)
    t = np.arange(n)
    return np.stack([smooth(np.interp(t, f, xy[:, 0])), smooth(np.interp(t, f, xy[:, 1]))], 1)

def player_height(det, h):
    hh = det["xyxy"][:, 3] - det["xyxy"][:, 1]
    if len(hh) < 5:
        return h / 4
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

def build_raw(it):
    frames = decode(it.path)
    idx = pick(len(frames))
    x = np.stack([letterbox(frames[f], RAW_S)[..., ::-1] for f in idx])
    return x, {"frames": idx.tolist(), "n": len(frames)}

def build_ball(it):
    for stage in ("detect", "pose"):
        if not os.path.exists(os.path.join(CACHE, stage, it.cid + ".npz")):
            raise FileNotFoundError(f"no {stage} cache for this clip - run the caching "
                                    f"steps in finetune/README.md section 2 first")
    frames = decode(it.path)
    n, (h, w) = len(frames), frames[0].shape[:2]
    det = np.load(os.path.join(CACHE, "detect", it.cid + ".npz"))
    pose = np.load(os.path.join(CACHE, "pose", it.cid + ".npz"))
    kp = pose["kpts"][..., :2]
    kc = pose["kconf"] if "kconf" in pose.files else np.ones(kp.shape[:2], np.float32)
    balls = ball_track(it.cid, n)
    cen = centres(balls, det, n, w, h)
    side = float(np.clip(SIDE_MULT * player_height(det, h), 160, min(h, w)))
    sign = det_teams(it.cid, det)
    s = BALL_S / side
    idx = pick(n)
    x = np.zeros((len(idx), BALL_S, BALL_S, 6), np.uint8)
    for j, f in enumerate(idx):
        x0 = float(np.clip(cen[f, 0] - side / 2, 0, w - side))
        y0 = float(np.clip(cen[f, 1] - side / 2, 0, h - side))
        crop = frames[f][int(y0):int(y0 + side), int(x0):int(x0 + side)]
        x[j, :, :, :3] = cv2.resize(crop, (BALL_S, BALL_S),
                                    interpolation=cv2.INTER_AREA)[..., ::-1]
        ball = np.zeros((BALL_S, BALL_S), np.uint8)
        if f in balls:
            blob(ball, (balls[f][0] - x0) * s, (balls[f][1] - y0) * s, 5)
        paint = np.full((BALL_S, BALL_S), 128, np.uint8)
        m = det["frame_idx"] == f
        boxes, sg = det["xyxy"][m], sign[m]
        for k in np.argsort(-(boxes[:, 3] - boxes[:, 1])):
            if sg[k] == 0:
                continue
            u0, v0, u1, v1 = ((boxes[k] - [x0, y0, x0, y0]) * s).astype(int)
            cv2.rectangle(paint, (u0, v0), (u1, v1), 255 if sg[k] > 0 else 0, -1)
        wrist = np.zeros((BALL_S, BALL_S), np.uint8)
        for r in np.where(pose["frame_idx"] == f)[0]:
            for k in (9, 10):
                if kc[r, k] > 0.3:
                    blob(wrist, (kp[r, k, 0] - x0) * s, (kp[r, k, 1] - y0) * s, 3)
        x[j, :, :, 3], x[j, :, :, 4], x[j, :, :, 5] = ball, paint, wrist
    return x, {"frames": idx.tolist(), "n": n, "side": side, "ball_frames": len(balls)}

def prep(kind, force=False, limit=None):
    out = os.path.join(CACHE, f"ftin_{kind}")
    os.makedirs(out, exist_ok=True)
    items = load_items()[:limit] if limit else load_items()
    build = build_raw if kind == "raw" else build_ball
    done = failed = 0
    for i, it in enumerate(items, 1):
        npy = os.path.join(out, it.cid + ".npy")
        if os.path.exists(npy) and not force:
            continue
        try:
            x, meta = build(it)
            with open(npy, "wb") as fh:
                np.save(fh, x)
            with open(npy[:-4] + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED {it.cid}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] {kind}: {done} written, {failed} failed", flush=True)
    have = sum(os.path.exists(os.path.join(out, it.cid + ".npy")) for it in items)
    print(f"{kind}: {have}/{len(items)} clips ready in {out}")
    return have == len(items)

def draw_overlays(rgb, hints, swap):
    out = np.ascontiguousarray(rgb.copy())
    k = np.ones((3, 3), np.uint8)
    for t in range(len(out)):
        paint = -hints[t, :, :, 1] if swap else hints[t, :, :, 1]
        for sign, col in ((1, LIME), (-1, MAGENTA)):
            m = (paint * sign > 0.5).astype(np.uint8)
            out[t][cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k).astype(bool)] = col
        b = hints[t, :, :, 0]
        if b.max() > 0.5:
            v, u = np.unravel_index(int(np.argmax(b)), b.shape)
            cv2.circle(out[t], (int(u), int(v)), max(6, out.shape[1] // 28), CYAN, 2)
    return out

class ClipSet:

    def __init__(self, items, kind, train=False, augment=False, t=16, seed=0,
                 ribbon=False):
        self.items, self.kind, self.train, self.aug, self.t = items, kind, train, augment, t
        self.ribbon = ribbon
        self.dir = os.path.join(CACHE, f"ftin_{kind}")
        self.rng = np.random.default_rng(seed)
        missing = [it.cid for it in items
                   if not os.path.exists(os.path.join(self.dir, it.cid + ".npy"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips not prepared for '{kind}' "
                                    f"(e.g. {missing[0]}) - run: python finetune/frames.py "
                                    f"--kind {kind}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        swap = bool(self.rng.random() < 0.5) if self.train else False
        return self.get(i, swap)

    def _times(self, n):
        if self.aug:
            span = max(2, int(round(n * self.rng.uniform(0.75, 1.0))))
            start = int(self.rng.integers(0, n - span + 1))
            return np.linspace(start, start + span - 1, self.t).round().astype(int)
        return np.linspace(0, n - 1, self.t).round().astype(int)

    def _frames(self, i):
        with open(os.path.join(self.dir, self.items[i].cid + ".npy"), "rb") as fh:
            x = np.load(fh)
        n_store = len(x)
        times = self._times(n_store)
        x = x[times].astype(np.float32)
        if self.kind == "ball":
            s = x.shape[1]
            if self.aug:
                c = int(round(s * self.rng.uniform(0.8, 1.0)))
                y0, x0 = self.rng.integers(0, s - c + 1, size=2)
                x = x[:, y0:y0 + c, x0:x0 + c]
            rs = lambda a: cv2.resize(a, (RAW_S, RAW_S), interpolation=cv2.INTER_AREA)
            x = np.stack([np.concatenate([rs(f[..., :3]), rs(f[..., 3:])], -1)
                          for f in x])
        if self.aug and self.rng.random() < 0.5:
            x = x[:, :, ::-1]
        rgb = x[..., :3] / 255.0
        if self.aug:
            rgb = rgb * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05)
            if self.rng.random() < 0.3:
                a = int(self.rng.integers(0, self.t))
                b = int(self.rng.integers(a + 1, self.t + 1))
                rgb[a:b] *= self.rng.uniform(0.92, 1.08, size=3)
            rgb = np.clip(rgb, 0, 1)
        hints = None
        if self.kind == "ball":
            hints = np.stack([x[..., 3] / 255.0, (x[..., 4] - 128.0) / 127.0,
                              x[..., 5] / 255.0], -1)
        return np.ascontiguousarray(rgb), hints, (times, n_store)

    def target(self, i, swap):
        ca, cb, t = question(self.items[i], swap)
        return np.concatenate([colour_code(ca), colour_code(cb)]), np.float32(t)

    def ribbon_of(self, i, swap, draw):
        from ribbon import load as load_ribbon
        times, n_store = draw
        att, m_att, team, m_team = load_ribbon(self.items[i].cid, n_store)
        att, m_att = att[times], m_att[times]
        team, m_team = team[times], m_team[times]
        if swap:
            team = 1.0 - team
        return np.stack([att, m_att, team, m_team]).astype(np.float32)

    def get(self, i, swap=False):
        rgb, hints, draw = self._frames(i)
        chans = [(rgb - MEAN) / STD]
        if hints is not None:
            chans += [hints[..., 0:1], -hints[..., 1:2] if swap else hints[..., 1:2],
                      hints[..., 2:3]]
        video = np.ascontiguousarray(np.concatenate(chans, -1).transpose(0, 3, 1, 2),
                                     dtype=np.float32)
        code, t = self.target(i, swap)
        if self.ribbon:
            return video, code, t, self.ribbon_of(i, swap, draw), i
        return video, code, t, i

    def get_rgb(self, i, swap=False, size=None):
        rgb, hints, _ = self._frames(i)
        u8 = (rgb * 255).round().astype(np.uint8)
        if hints is not None:
            u8 = draw_overlays(u8, hints, swap)
        if size and size != u8.shape[1]:
            u8 = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
                           for f in u8])
        return u8

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["raw", "ball"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    sys.exit(0 if prep(a.kind, a.force, a.limit) else 1)
