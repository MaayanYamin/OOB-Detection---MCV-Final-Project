import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, folds, load_items, save_run, score
from frames import MEAN, STD, ClipSet
from train_videomae import ColourHead

BACKBONES = {
    "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", 256, 16),
    "videomae": ("MCG-NJU/videomae-base-finetuned-ssv2", 224, 16),
}
BACKBONE = "vjepa2"
MODEL_ID, SIZE, PATCH = BACKBONES[BACKBONE]
N_AUG = 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_backbone():
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16 if DEV == "cuda"
                                  else torch.float32)
    return m.to(DEV).eval()

@torch.no_grad()
def encode(model, frames_u8):
    x = (frames_u8.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).unsqueeze(0).to(DEV, next(model.parameters()).dtype)
    try:
        tok = model.get_vision_features(pixel_values_videos=x)
    except (AttributeError, TypeError):
        try:
            tok = model(pixel_values_videos=x)
        except (TypeError, ValueError):
            tok = model(pixel_values=x)
    if isinstance(tok, (tuple, list)):
        tok = tok[0]
    if hasattr(tok, "last_hidden_state"):
        tok = tok.last_hidden_state
    tok = tok[0].float()
    t, g = x.shape[1] // 2, SIZE // PATCH
    if tok.shape[0] != t * g * g:
        raise RuntimeError(f"unexpected token count {tok.shape[0]} (expected {t * g * g}); "
                           "check the V-JEPA 2 output layout")
    tok = tok.view(t, g, g, -1).permute(0, 3, 1, 2)
    tok = F.avg_pool2d(tok, 2).permute(0, 2, 3, 1).reshape(-1, tok.shape[1])
    return tok.half().cpu().numpy()

def feat_dir(inp):
    return os.path.join(CACHE, f"{BACKBONE}_feats", inp)

def feat_path(inp, cid, order, k):
    return os.path.join(feat_dir(inp), f"{cid}__o{order}__k{k}.npy")

def copies(inp):
    return ((0,) if inp == "raw" else (0, 1)), (N_AUG if inp.endswith("_aug") else 1)

def extract(items, inp):
    kind = inp.split("_")[0]
    orders, n_k = copies(inp)
    os.makedirs(feat_dir(inp), exist_ok=True)
    todo = [(i, o, k) for i, it in enumerate(items) for o in orders for k in range(n_k)
            if not os.path.exists(feat_path(inp, it.cid, o, k))]
    if not todo:
        return
    print(f"extracting V-JEPA 2 features: {len(todo)} to do for '{inp}'", flush=True)
    model = load_backbone()
    plain = ClipSet(items, kind)
    aug = ClipSet(items, kind, augment=True, seed=1234)
    t0 = time.time()
    for n, (i, o, k) in enumerate(todo, 1):
        ds = aug if k > 0 else plain
        f = encode(model, ds.get_rgb(i, swap=bool(o), size=SIZE))
        with open(feat_path(inp, items[i].cid, o, k), "wb") as fh:
            np.save(fh, f)
        if n % 25 == 0 or n == len(todo):
            print(f"   {n}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

def load_feat(inp, cid, order, k):
    orders, _ = copies(inp)
    with open(feat_path(inp, cid, order if order in orders else 0, k), "rb") as fh:
        return np.load(fh).astype(np.float32)

def run_fold(items, tr, te, inp, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    orders, n_k = copies(inp)
    ds = ClipSet(items, inp.split("_")[0])
    d = load_feat(inp, items[tr[0]].cid, 0, 0).shape[1]
    head = ColourHead(d).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    for ep in range(args.epochs):
        head.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            tok, code, tgt = [], [], []
            for i in idx:
                swap = bool(rng.random() < 0.5)
                k = int(rng.integers(0, n_k))
                tok.append(load_feat(inp, items[i].cid, int(swap), k))
                c, t = ds.target(i, swap)
                code.append(c)
                tgt.append(t)
            tok = torch.from_numpy(np.stack(tok)).to(DEV)
            code = torch.from_numpy(np.stack(code)).to(DEV)
            tgt = torch.from_numpy(np.array(tgt, np.float32)).to(DEV)
            loss = F.binary_cross_entropy_with_logits(head(tok, code), tgt * 0.9 + 0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(idx):
        head.eval()
        out = []
        for i in idx:
            pa = []
            for swap in (False, True):
                tok = torch.from_numpy(load_feat(inp, items[i].cid, int(swap), 0))[None].to(DEV)
                c, _ = ds.target(i, swap)
                pa.append(torch.sigmoid(head(tok, torch.from_numpy(c)[None].to(DEV))).item())
            out.append(0.5 * (pa[0] + 1 - pa[1]))
        return np.array(out)

    fit = score([items[i] for i in tr], predict(tr))["clip"]["acc"]
    return predict(te), fit

def main():
    global MODEL_ID, SIZE, PATCH, BACKBONE, N_AUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, choices=["raw", "ball", "ball_aug"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--protocol", default="game", choices=["game", "arena"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone", default="vjepa2", choices=sorted(BACKBONES),
                    help="vjepa2 (ViT-L, video-pretrained) or videomae (base, SSv2)")
    ap.add_argument("--aug-copies", type=int, default=N_AUG,
                    help="augmented copies per clip for *_aug inputs; lower is cheaper")
    ap.add_argument("--limit", type=int, help="smoke test on the first N clips; not saved")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    BACKBONE = args.backbone
    MODEL_ID, SIZE, PATCH = BACKBONES[BACKBONE]
    N_AUG = args.aug_copies

    print(f"GPU: {torch.cuda.get_device_name(0)}" if DEV == "cuda"
          else "WARNING: no CUDA GPU found - feature extraction will be very slow on CPU")
    items = load_items()
    if args.limit:
        items = items[:args.limit]
    extract(items, args.input)

    for seed in args.seeds:
        name = f"{BACKBONE}_{args.input}_{args.protocol}_s{seed}"
        if not args.limit and os.path.exists(os.path.join(RESULTS, name + ".json")) \
                and not args.force:
            print(f"{name}: already done, skipping (--force to redo)")
            continue
        t0 = time.time()
        p = np.full(len(items), np.nan)
        fits = []
        for f, (tr, te) in enumerate(folds(items, args.protocol, args.folds, seed), 1):
            p[te], fit = run_fold(items, tr, te, args.input, args, seed)
            fits.append(fit)
            print(f"   seed {seed} fold {f}: train fit {fit:.0%}, {len(te)} test clips",
                  flush=True)
        c = score(items, p)["clip"]
        print(f"{name}: acc {c['acc']:.1%} (n={c['n']})  lift vs constant "
              f"{c['constant']['lift']:+.1%}, vs arena {c['arena']['lift']:+.1%}, "
              f"vs lighter {c['lighter']['lift']:+.1%}   [{(time.time() - t0) / 60:.1f} min]")
        if args.limit:
            print("   (smoke test - not saved)")
            continue
        path = save_run(name, items, p, dict(
            model=BACKBONE, input=args.input, protocol=args.protocol, seed=seed,
            backbone=MODEL_ID, epochs=args.epochs, lr=args.lr, train_fit=fits,
            minutes=round((time.time() - t0) / 60, 1)))
        print(f"   saved {path}\n")

if __name__ == "__main__":
    main()
