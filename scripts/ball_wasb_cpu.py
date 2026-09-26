import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCORE_THR = 0.5
FRAMES_IN = 3

def build_model(repo, weights, model_name="wasb"):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, os.path.join(repo, "src"))
    from models import build_model as wasb_build

    cfg_path = os.path.join(repo, "src", "configs", "model", f"{model_name}.yaml")
    model_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.create({"model": model_cfg})
    model = wasb_build(cfg)

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  note: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        print("  weights loaded strictly, no missing or unexpected keys")
    model.eval()
    return model, int(model_cfg["inp_width"]), int(model_cfg["inp_height"])

def heatmap_to_xy(hm, thr=SCORE_THR):
    if hm.max() <= thr:
        return None, float(hm.max())
    _, binar = cv2.threshold(hm, thr, 1, cv2.THRESH_BINARY)
    n, labels = cv2.connectedComponents(binar.astype(np.uint8))
    best, best_score = None, -1.0
    for m in range(1, n):
        ys, xs = np.where(labels == m)
        w = hm[ys, xs]
        score = float(w.sum())
        if score > best_score:
            best_score = score
            best = (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()))
    return best, best_score

def run_clip(model, video, inp_w, inp_h, max_frames=None):
    import torch
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) < FRAMES_IN:
        return None
    H, W = frames[0].shape[:2]
    sx, sy = W / inp_w, H / inp_h

    small = [cv2.resize(f, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
             for f in frames]
    small = [(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
             for f in small]

    fidx, xy, conf = [], [], []
    with torch.no_grad():
        for start in range(0, len(small) - FRAMES_IN + 1, FRAMES_IN):
            win = small[start:start + FRAMES_IN]
            x = np.concatenate([w.transpose(2, 0, 1) for w in win], axis=0)
            out = model(torch.from_numpy(x[None]).float())
            hm = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(hm, dict):
                hm = hm[sorted(hm.keys())[0]]
            hm = torch.sigmoid(hm)[0].cpu().numpy()
            for j in range(min(FRAMES_IN, hm.shape[0])):
                pt, sc = heatmap_to_xy(hm[j])
                fidx.append(start + j)
                conf.append(sc)
                xy.append((pt[0] * sx, pt[1] * sy) if pt else (np.nan, np.nan))
    return dict(frame_idx=np.array(fidx, np.int32),
                xy=np.array(xy, np.float32),
                conf=np.array(conf, np.float32),
                n_frames=len(frames), width=W, height=H)

def diagnostics(r):
    ok = ~np.isnan(r["xy"][:, 0])
    rate = ok.mean()
    plaus = np.nan
    if ok.sum() >= 4:
        p = r["xy"][ok]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        accel = np.abs(np.diff(step))
        typ = np.median(step) if np.median(step) > 0 else 1.0
        plaus = float((accel < 2 * typ).mean())
    return rate, plaus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips-dir", default="clips/clips")
    ap.add_argument("--cache", default="cache/cache")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=60,
                    help="cap frames per clip; CPU inference is slow")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    print("building WASB (hrnet) with basketball weights")
    model, iw, ih = build_model(args.repo, args.weights)

    import glob
    clips = sorted(glob.glob(os.path.join(args.clips_dir, "*.mp4")))
    if args.only:
        want = set(args.only.split(","))
        clips = [c for c in clips if os.path.basename(c)[:-4] in want]
    else:
        clips = clips[:args.limit]

    outdir = os.path.join(args.cache, "ball")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'clip':<44}{'frames':>8}{'fires':>8}{'smooth':>9}{'sec':>7}")
    for c in clips:
        name = os.path.basename(c)[:-4]
        t0 = time.time()
        r = run_clip(model, c, iw, ih, args.max_frames)
        if r is None:
            print(f"{name:<44}  too short")
            continue
        rate, plaus = diagnostics(r)
        np.savez_compressed(os.path.join(outdir, f"{name}__wasb__s{iw}.npz"),
                            frame_idx=r["frame_idx"], xy=r["xy"], conf=r["conf"],
                            method="wasb", scale=iw)
        print(f"{name:<44}{len(r['frame_idx']):>8}{rate:>8.0%}"
              f"{plaus:>9.2f}{time.time()-t0:>7.0f}")

    print("\n'fires' = share of frames with a detection above 0.5.")
    print("'smooth' = share of steps where acceleration stays small; a real ball")
    print("follows an arc, so near 1.0 is good and low means it is jumping around.")

if __name__ == "__main__":
    main()
