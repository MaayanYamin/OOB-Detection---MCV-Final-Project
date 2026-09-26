import argparse
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POC = os.path.join(ROOT, "poc", "Basketball_OOB_Detection-main")
sys.path.insert(0, POC)
sys.path.insert(0, os.path.join(ROOT, "finetune"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import RGB, folds, load_items, parts, save_run
import config
from dataset import OOBDatasetWithColor
from model import OOBModelWithColor
from torch.utils.data import DataLoader

THEIR_COLOURS = {"white": (.95, .95, .95), "blue": (.10, .30, .75), "black": (.08, .08, .08)}
_FRAMES = {}

def cached_extract(self, video_path, aug_version=0, seed=0):
    import cv2
    key = (video_path, aug_version)
    if key not in _FRAMES:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = self.num_frames
        if total < n:
            idx = np.array([i % max(total, 1) for i in range(n)])
        elif self.augment and aug_version == 1:
            idx = np.linspace(0, int(total * 0.66), n, dtype=int)
        elif self.augment and aug_version == 2:
            idx = np.linspace(int(total * 0.33), total - 1, n, dtype=int)
        else:
            idx = np.linspace(0, total - 1, n, dtype=int)
        want, got, i = set(int(v) for v in idx), {}, 0
        while want:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (224, 224))
                want.discard(i)
            i += 1
        cap.release()
        _FRAMES[key] = [got[int(v)] for v in idx if int(v) in got]
    frames = [self.transform(f) for f in _FRAMES[key]]
    return (torch.stack(frames) if frames
            else torch.zeros(self.num_frames, 3, 224, 224))

def nearest_colour(word):
    rgb = np.array(RGB[parts(word)[0]])
    return min(THEIR_COLOURS, key=lambda k: np.linalg.norm(rgb - np.array(THEIR_COLOURS[k])))

def corpus():
    items = load_items()
    paths = [it.path for it in items]
    labels = [1 - it.y_home for it in items]
    colours = [nearest_colour(it.col_home) for it in items]
    return items, paths, labels, colours

def train_fold_final_epoch(train_loader, val_loader, fold, epochs):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OOBModelWithColor().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    crit = torch.nn.CrossEntropyLoss()
    preds = []
    for ep in range(epochs):
        model.train()
        tot = corr = 0
        losses = []
        for b in train_loader:
            out = model(b["frames"].to(dev), b["color"].to(dev))
            loss = crit(out, b["label"].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            corr += int(out.argmax(1).cpu().eq(b["label"]).sum())
            tot += len(b["label"])
        model.eval()
        preds = []
        with torch.no_grad():
            for b in val_loader:
                p = torch.softmax(model(b["frames"].to(dev), b["color"].to(dev)), 1)[:, 1]
                for name, prob, lab in zip(b["video_name"], p.cpu().numpy(),
                                           b["label"].numpy()):
                    preds.append((name, float(prob), int(lab)))
        acc = float(np.mean([(p > 0.5) == lab for _, p, lab in preds]))
        print(f"  fold {fold} epoch {ep+1}/{epochs}: train {corr/tot:.0%}  val {acc:.0%}",
              flush=True)
        sched.step(float(np.mean(losses)))
    return preds

def honest(items, paths, labels, colours, epochs):
    p_home = np.full(len(items), np.nan)
    for k, (tr, te) in enumerate(folds(items, "game", 5, 0), 1):
        tr_ds = OOBDatasetWithColor([paths[i] for i in tr], [labels[i] for i in tr],
                                    [colours[i] for i in tr], is_train=True, augment=True)
        te_ds = OOBDatasetWithColor([paths[i] for i in te], [labels[i] for i in te],
                                    [colours[i] for i in te], is_train=False, augment=False)
        preds = train_fold_final_epoch(
            DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True),
            DataLoader(te_ds, batch_size=1, shuffle=False), k, epochs)
        for j, (_, prob, _) in zip(te, preds):
            p_home[j] = 1.0 - prob
        print(f"fold {k}: {len(te)} test clips done", flush=True)
    path = save_run("poc_mobilenet_lstm_game_s0", items, p_home,
                    dict(model="poc_mobilenet_lstm", input="raw", protocol="game", seed=0,
                         note="original PoC architecture, grouped folds, final epoch"))
    print("saved", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    OOBDatasetWithColor.extract_frames = cached_extract
    items, paths, labels, colours = corpus()
    print(f"{len(paths)} clips | home-ball {sum(labels)} away-ball {len(labels)-sum(labels)}"
          f" | colours " + ", ".join(f"{c}:{colours.count(c)}" for c in THEIR_COLOURS))
    print("device: " + ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    honest(items, paths, labels, colours, args.epochs)
    print("run took %.0f min" % ((time.time() - t0) / 60))

if __name__ == "__main__":
    main()
