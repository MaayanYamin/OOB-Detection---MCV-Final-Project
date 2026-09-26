import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

TORSO_X = (0.20, 0.80)
TORSO_Y = (0.15, 0.55)

DEFAULT_SIGLIP = "google/siglip-base-patch16-224"
DEFAULT_DINO = "facebook/dinov2-base"

def find_clips(clips_dir, limit=None):
    clips = sorted(Path(clips_dir).glob("*.mp4"))
    if not clips:
        sys.exit(f"no .mp4 files found in {clips_dir}")
    return clips[:limit] if limit else clips

def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames

def clip_key(clip):
    stem = clip.stem
    if "__a" not in stem:
        return stem
    head, tail = stem.split("__a", 1)
    return f"{head}__a{tail.split('_')[0]}"

def out_path(cache, stage, clip, ext):
    d = Path(cache) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{clip_key(clip)}.{ext}"

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]

def pick_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  device: cuda -- {name}")
        return "cuda"
    print("  device: CPU ONLY. This will work but is roughly 20-40x slower;")
    print("          budget an overnight run rather than an hour.")
    return "cpu"

class Models:

    def __init__(self, device, siglip_id, dino_id, det_weights, pose_weights):
        self.device = device
        self.siglip_id, self.dino_id = siglip_id, dino_id
        self.det_weights, self.pose_weights = det_weights, pose_weights
        self._det = self._pose = None
        self._siglip = self._siglip_proc = None
        self._dino = self._dino_proc = None

    @property
    def det(self):
        if self._det is None:
            from ultralytics import YOLO
            print(f"  loading detector {self.det_weights}")
            self._det = YOLO(self.det_weights)
        return self._det

    @property
    def pose(self):
        if self._pose is None:
            from ultralytics import YOLO
            print(f"  loading pose {self.pose_weights}")
            self._pose = YOLO(self.pose_weights)
        return self._pose

    @property
    def siglip(self):
        if self._siglip is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            print(f"  loading {self.siglip_id}")
            self._siglip_proc = AutoImageProcessor.from_pretrained(self.siglip_id)
            self._siglip = AutoModel.from_pretrained(self.siglip_id).to(self.device).eval()
        return self._siglip, self._siglip_proc

    @property
    def dino(self):
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel
            print(f"  loading {self.dino_id}")
            self._dino_proc = AutoImageProcessor.from_pretrained(self.dino_id)
            self._dino = AutoModel.from_pretrained(self.dino_id).to(self.device).eval()
        return self._dino, self._dino_proc

def stage_detect(frames, models, imgsz, conf, batch):
    fi, boxes, confs = [], [], []
    for start, chunk in batched(frames, batch):
        res = models.det.predict(chunk, imgsz=imgsz, conf=conf, classes=[0],
                                 device=models.device, verbose=False)
        for j, r in enumerate(res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            c = b.conf.cpu().numpy().astype(np.float32)
            fi.append(np.full(len(xyxy), start + j, dtype=np.int32))
            boxes.append(xyxy)
            confs.append(c)
    if not boxes:
        return dict(frame_idx=np.zeros(0, np.int32), xyxy=np.zeros((0, 4), np.float32),
                    conf=np.zeros(0, np.float32))
    return dict(frame_idx=np.concatenate(fi), xyxy=np.concatenate(boxes),
                conf=np.concatenate(confs))

def stage_pose(frames, models, imgsz, conf, batch):
    fi, boxes, confs, kpts, kconf = [], [], [], [], []
    for start, chunk in batched(frames, batch):
        res = models.pose.predict(chunk, imgsz=imgsz, conf=conf,
                                  device=models.device, verbose=False)
        for j, r in enumerate(res):
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            c = r.boxes.conf.cpu().numpy().astype(np.float32)
            k = r.keypoints.xy.cpu().numpy().astype(np.float32)
            kc = (r.keypoints.conf.cpu().numpy().astype(np.float32)
                  if r.keypoints.conf is not None
                  else np.ones(k.shape[:2], np.float32))
            fi.append(np.full(len(xyxy), start + j, dtype=np.int32))
            boxes.append(xyxy); confs.append(c); kpts.append(k); kconf.append(kc)
    if not boxes:
        return dict(frame_idx=np.zeros(0, np.int32), xyxy=np.zeros((0, 4), np.float32),
                    conf=np.zeros(0, np.float32), kpts=np.zeros((0, 17, 2), np.float32),
                    kconf=np.zeros((0, 17), np.float32))
    return dict(frame_idx=np.concatenate(fi), xyxy=np.concatenate(boxes),
                conf=np.concatenate(confs), kpts=np.concatenate(kpts),
                kconf=np.concatenate(kconf))

def torso_crop(frame, box, min_side=12):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1 = int(round(x1 + TORSO_X[0] * w)); cx2 = int(round(x1 + TORSO_X[1] * w))
    cy1 = int(round(y1 + TORSO_Y[0] * h)); cy2 = int(round(y1 + TORSO_Y[1] * h))
    H, W = frame.shape[:2]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    if cx2 - cx1 < min_side or cy2 - cy1 < min_side:
        return None
    return frame[cy1:cy2, cx1:cx2]

def lab_histogram(crop, bins=(8, 4, 4)):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, bins, [0, 256] * 3)
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist

@np.errstate(all="ignore")
def stage_embed(frames, det, models, n_frames, batch, thumb_frames):
    import torch
    from PIL import Image

    if len(det["frame_idx"]) == 0:
        return None

    have = np.unique(det["frame_idx"])

    def sample(n):
        if n <= 0 or len(have) <= n:
            return set(int(t) for t in have)
        return set(int(t) for t in
                   have[np.linspace(0, len(have) - 1, n).round().astype(int)])

    take = sample(n_frames)
    thumb_on = sample(thumb_frames)

    crops, meta, thumbs, thumb_row = [], [], [], []
    for i, f in enumerate(det["frame_idx"]):
        f = int(f)
        if f not in take:
            continue
        c = torso_crop(frames[f], det["xyxy"][i])
        if c is None:
            continue
        if f in thumb_on and len(thumbs) < 400:
            thumb_row.append(len(crops))
            thumbs.append(cv2.resize(c, (48, 72), interpolation=cv2.INTER_AREA))
        crops.append(c)
        meta.append((f, i))

    if not crops:
        return None

    lab = np.stack([lab_histogram(c) for c in crops])
    pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]

    def encode(model, proc, kind):
        outs = []
        for _, chunk in batched(pil, batch):
            inputs = proc(images=chunk, return_tensors="pt").to(models.device)
            with torch.no_grad():
                if kind == "siglip":
                    v = model.get_image_features(**inputs)
                    if not torch.is_tensor(v):
                        v = v.pooler_output
                else:
                    v = model(**inputs).last_hidden_state[:, 0]
            outs.append(v.float().cpu().numpy())
        v = np.concatenate(outs).astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return (v / np.where(n == 0, 1, n)).astype(np.float16)

    sig_m, sig_p = models.siglip
    din_m, din_p = models.dino
    out = dict(
        frame_idx=np.array([m[0] for m in meta], np.int32),
        det_row=np.array([m[1] for m in meta], np.int32),
        xyxy=np.stack([det["xyxy"][m[1]] for m in meta]).astype(np.float32),
        lab=lab.astype(np.float16),
        siglip=encode(sig_m, sig_p, "siglip"),
        dinov2=encode(din_m, din_p, "dino"),
    )
    if thumbs:
        out["thumbs"] = np.stack(thumbs).astype(np.uint8)
        out["thumb_row"] = np.array(thumb_row, np.int32)
    return out

def cluster_clip(emb, feature, n_pca=16, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = np.asarray(emb[feature], dtype=np.float32)
    if len(X) < 4:
        return None

    k = min(n_pca, X.shape[0] - 1, X.shape[1])
    Xr = PCA(n_components=k, random_state=seed).fit_transform(X) if k >= 2 else X
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xr)
    labels = km.labels_

    L_centres = (np.arange(8) + 0.5) * (256 / 8)
    lab_L = np.asarray(emb["lab"], np.float32).reshape(len(X), 8, 4, 4).sum(axis=(2, 3))
    mean_L = np.array([float((lab_L[labels == c] * L_centres).sum(1).mean())
                       for c in (0, 1)])

    order = np.argsort(-mean_L)
    remap = {int(order[0]): 0, int(order[1]): 1}
    labels = np.array([remap[int(l)] for l in labels], np.int32)
    mean_L = mean_L[order]

    sil = None
    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(Xr, labels))
    except Exception:
        pass

    return dict(labels=labels, mean_L=mean_L, silhouette=sil,
                sizes=[int((labels == 0).sum()), int((labels == 1).sum())])

def contact_sheet(emb, cl, title, subtitle, per_row=12):
    if "thumbs" not in emb or "thumb_row" not in emb:
        return None
    th, tw = 72, 48
    pad, head = 6, 46
    thumb_label = cl["labels"][np.asarray(emb["thumb_row"], np.int64)]
    rows = []
    for c in (0, 1):
        idx = np.where(thumb_label == c)[0]
        if len(idx) == 0:
            rows.append(np.full((th, per_row * (tw + pad), 3), 30, np.uint8))
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(per_row, len(idx))).round().astype(int)]
        strip = np.full((th, per_row * (tw + pad), 3), 30, np.uint8)
        for j, i in enumerate(pick):
            strip[:, j * (tw + pad):j * (tw + pad) + tw] = emb["thumbs"][i]
        rows.append(strip)

    W = rows[0].shape[1]
    sheet = np.full((head + th * 2 + pad * 3, W, 3), 24, np.uint8)
    sheet[head:head + th] = rows[0]
    sheet[head + th + pad:head + th * 2 + pad] = rows[1]
    f, s = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    cv2.putText(sheet, title, (6, 18), f, s, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(sheet, subtitle, (6, 36), f, 0.4, (150, 190, 230), 1, cv2.LINE_AA)
    cv2.putText(sheet, "BRIGHT", (W - 66, head + 14), f, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, "DARK", (W - 66, head + th + pad + 14), f, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
    return sheet

def load_manifest(path):
    if not Path(path).exists():
        print(f"  note: {path} not found -- contact sheets will omit team names")
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        return {r["event_id"]: r for r in csv.DictReader(fh)}

def run_cluster(cache, manifest_path, feature, force):
    emb_dir = Path(cache) / "embed"
    files = sorted(emb_dir.glob("*.npz"))
    if not files:
        sys.exit(f"no embeddings in {emb_dir} -- run --stage embed first")

    man = load_manifest(manifest_path)
    (Path(cache) / "cluster").mkdir(parents=True, exist_ok=True)
    (Path(cache) / "review").mkdir(parents=True, exist_ok=True)

    todo, done, skipped = [], 0, 0
    for f in files:
        clip = f.stem
        jpath = Path(cache) / "cluster" / f"{clip}.json"
        if jpath.exists() and not force:
            skipped += 1
        emb = np.load(f)
        cl = cluster_clip(emb, feature)
        if cl is None:
            print(f"  {clip}: too few crops, skipped")
            continue

        event_id = clip.split("__a")[0]
        row = man.get(event_id, {})
        away, home = row.get("away_team", "?"), row.get("home_team", "?")

        if not jpath.exists() or force:
            json.dump(dict(clip=clip, event_id=event_id, feature=feature,
                           labels=cl["labels"].tolist(),
                           frame_idx=emb["frame_idx"].tolist(),
                           det_row=emb["det_row"].tolist(),
                           mean_L=cl["mean_L"].tolist(),
                           silhouette=cl["silhouette"], sizes=cl["sizes"]),
                      open(jpath, "w", encoding="utf-8"), indent=1)
            sheet = contact_sheet(
                emb, cl, clip,
                f"which team is the BRIGHT row?   {away} (away)  vs  {home} (home)")
            if sheet is not None:
                ok, buf = cv2.imencode(".jpg", sheet,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    (Path(cache) / "review" / f"{clip}.jpg").write_bytes(buf.tobytes())
            done += 1

        todo.append(dict(
            event_id=event_id, clip=clip, away_team=away, home_team=home,
            bright_mean_L=round(float(cl["mean_L"][0]), 1),
            dark_mean_L=round(float(cl["mean_L"][1]), 1),
            n_bright=cl["sizes"][0], n_dark=cl["sizes"][1],
            silhouette=round(cl["silhouette"], 3) if cl["silhouette"] is not None else "",
            light_team="", notes=""))

    csv_path = Path(cache) / "labels_todo.csv"
    if not csv_path.exists() or force:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(todo[0].keys()))
            w.writeheader()
            w.writerows(todo)

    print(f"\nclustered {done} clips ({skipped} already done)")
    print(f"  contact sheets : {Path(cache) / 'review'}")
    print(f"  labelling sheet: {csv_path}")
    print("\nNext, by hand: open each contact sheet, and in labels_todo.csv put the")
    print("team code of whichever team is wearing the BRIGHT jerseys (top row) into")
    print("the light_team column. One choice of two per clip.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", default="clips/clips")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "pose", "embed", "cluster"])
    ap.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    ap.add_argument("--force", action="store_true", help="redo clips already cached")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8, help="frames per model call")
    ap.add_argument("--embed-batch", type=int, default=64, help="crops per model call")
    ap.add_argument("--embed-frames", type=int, default=0,
                    help="frames per clip to embed; 0 = every frame (default). "
                         "Dense costs minutes of GPU and ~216 MB at float16, and "
                         "re-running this pass costs days of calendar time.")
    ap.add_argument("--thumb-frames", type=int, default=16,
                    help="frames per clip that also keep a thumbnail, for the "
                         "contact sheets. Dense thumbnails would cost 674 MB.")
    ap.add_argument("--det-weights", default="yolo11l.pt")
    ap.add_argument("--pose-weights", default="yolo11l-pose.pt")
    ap.add_argument("--siglip", default=DEFAULT_SIGLIP)
    ap.add_argument("--dino", default=DEFAULT_DINO)
    ap.add_argument("--cluster-feature", default="siglip",
                    choices=["siglip", "dinov2", "lab"])
    ap.add_argument("--no-thumbs", action="store_true",
                    help="skip storing crop thumbnails (disables contact sheets)")
    args = ap.parse_args()

    if args.stage == "cluster":
        run_cluster(args.cache, args.manifest, args.cluster_feature, args.force)
        return

    stages = ["detect", "pose", "embed"] if args.stage == "all" else [args.stage]
    if "embed" in stages and "detect" not in stages:
        stages.insert(0, "detect")

    clips = find_clips(args.clips_dir, args.limit)
    print(f"{len(clips)} clips from {args.clips_dir}")
    print(f"stages: {', '.join(stages)}")
    device = pick_device()
    models = Models(device, args.siglip, args.dino, args.det_weights, args.pose_weights)

    t0, failures = time.time(), []
    for n, clip in enumerate(clips, 1):
        want = [s for s in stages
                if args.force or not out_path(args.cache, s, clip, "npz").exists()]
        if not want:
            print(f"[{n}/{len(clips)}] {clip.stem} -- cached, skipping")
            continue

        print(f"[{n}/{len(clips)}] {clip.stem}")
        try:
            ts = time.time()
            frames = decode(clip)
            print(f"  {len(frames)} frames  {frames[0].shape[1]}x{frames[0].shape[0]}")

            det = None
            if "detect" in want:
                det = stage_detect(frames, models, args.imgsz, args.conf, args.batch)
                np.savez_compressed(out_path(args.cache, "detect", clip, "npz"), **det)
                print(f"  detect: {len(det['frame_idx'])} boxes")
            if "pose" in want:
                pose = stage_pose(frames, models, args.imgsz, args.conf, args.batch)
                np.savez_compressed(out_path(args.cache, "pose", clip, "npz"), **pose)
                print(f"  pose:   {len(pose['frame_idx'])} people")
            if "embed" in want:
                if det is None:
                    det = dict(np.load(out_path(args.cache, "detect", clip, "npz")))
                emb = stage_embed(frames, det, models, args.embed_frames,
                                  args.embed_batch,
                                  0 if args.no_thumbs else args.thumb_frames)
                if emb is None:
                    print("  embed:  no usable torso crops")
                else:
                    np.savez_compressed(out_path(args.cache, "embed", clip, "npz"), **emb)
                    print(f"  embed:  {len(emb['frame_idx'])} crops"
                          f"  siglip{emb['siglip'].shape}  dinov2{emb['dinov2'].shape}")
            del frames
            print(f"  {time.time() - ts:.1f}s")
        except Exception:
            failures.append(clip.stem)
            print(f"  FAILED: {clip.stem}")
            traceback.print_exc()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {args.cache}/")
    if failures:
        print(f"{len(failures)} clips failed: {', '.join(failures)}")
    if "embed" in stages:
        print("\nNow run:  python scripts/cache_gpu.py --stage cluster")

if __name__ == "__main__":
    main()
