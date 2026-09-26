import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, clear_partial, folds, load_items, plateau_report,
                    preflight, resume, save_partial, save_run, score)
from frames import ClipSet

MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def restore_qkv_bias(backbone, model_id):
    try:
        from transformers.utils import cached_file

        def grab(name):
            try:
                return cached_file(model_id, name,
                                   _raise_exceptions_for_missing_entries=False)
            except Exception:
                return None

        sd, used = None, None
        for name in ("model.safetensors", "pytorch_model.bin"):
            path = grab(name)
            if not path:
                continue
            if name.endswith(".safetensors"):
                import safetensors.torch as sft
                sd, used = sft.load_file(path), name
            else:
                sd, used = torch.load(path, map_location="cpu", weights_only=True), name
            break
        if sd is None:
            import json as _json
            for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                idx = grab(idx_name)
                if not idx:
                    continue
                with open(idx, encoding="utf-8") as fh:
                    shards = sorted(set(_json.load(fh)["weight_map"].values()))
                sd, used = {}, idx_name
                for sh in shards:
                    q = grab(sh)
                    if not q:
                        continue
                    if sh.endswith(".safetensors"):
                        import safetensors.torch as sft
                        sd.update(sft.load_file(q))
                    else:
                        sd.update(torch.load(q, map_location="cpu", weights_only=True))
                break
        if not sd:
            return ("qkv bias: no weights file found in the cache "
                    "(tried safetensors, bin and both shard indexes); left as loaded")
        done, mags = 0, []
        for i, layer in enumerate(backbone.encoder.layer):
            att = layer.attention.attention
            for src, dst in (("q_bias", "query"), ("v_bias", "value")):
                key = next((k for k in
                            (f"videomae.encoder.layer.{i}.attention.attention.{src}",
                             f"encoder.layer.{i}.attention.attention.{src}") if k in sd), None)
                lin = getattr(att, dst, None)
                if key is None or lin is None or getattr(lin, "bias", None) is None:
                    continue
                v = sd[key]
                if tuple(v.shape) != tuple(lin.bias.shape):
                    continue
                with torch.no_grad():
                    lin.bias.copy_(v.to(lin.bias.dtype))
                mags.append(float(v.abs().mean()))
                done += 1
        if not done:
            hits = sum(1 for k in sd if k.endswith(("q_bias", "v_bias")))
            return (f"qkv bias: nothing restored from {used} "
                    f"({len(sd)} tensors, {hits} named q_bias/v_bias) - either this "
                    f"transformers maps them itself, or the naming changed again")
        return (f"qkv bias: restored {done} pretrained tensors that transformers dropped, "
                f"from {used} (mean |bias| {np.mean(mags):.4f}, "
                f"max {np.max(mags):.4f})")
    except Exception as e:
        return f"qkv bias: NOT restored ({type(e).__name__}: {e}) - continuing as loaded"

def widen(backbone, channels):
    pe = backbone.embeddings.patch_embeddings
    old = pe.projection
    new = nn.Conv3d(channels, old.out_channels, old.kernel_size, old.stride,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    pe.projection = new
    pe.num_channels = channels
    backbone.config.num_channels = channels

class ColourHead(nn.Module):

    def __init__(self, d, heads=8, dropout=0.3):
        super().__init__()
        self.q0 = nn.Parameter(torch.zeros(1, 1, d))
        self.qc = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d + 12, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 1))

    def forward(self, tokens, code):
        q = self.q0.expand(len(tokens), -1, -1) + self.qc(code).unsqueeze(1)
        v, _ = self.attn(q, tokens, tokens)
        return self.out(torch.cat([self.norm(v[:, 0]), code], 1)).squeeze(1)

class RibbonHead(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.att = nn.Linear(d, 1)
        self.team = nn.Sequential(nn.Linear(d + 12, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, tokens, code, t):
        b, n, d = tokens.shape
        x = self.norm(tokens.view(b, t, n // t, d).mean(2))
        c = code.unsqueeze(1).expand(-1, t, -1)
        return self.att(x).squeeze(-1), self.team(torch.cat([x, c], -1)).squeeze(-1)

def pool_ribbon(rib, t):
    b, _, f = rib.shape
    if f % t:
        raise ValueError(f"cannot pool {f} ribbon frames onto {t} model time steps")
    r = rib.view(b, 4, t, f // t)

    def avg(v, m):
        s = m.sum(-1)
        return (v * m).sum(-1) / s.clamp(min=1.0), (s > 0).float()

    ya, ma = avg(r[:, 0], r[:, 1])
    yt, mt = avg(r[:, 2], r[:, 3])
    return ya, ma, yt, mt

def masked_bce(logit, y, m):
    if float(m.sum()) == 0.0:
        return logit.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logit, y * 0.9 + 0.05, reduction="none")
    return (loss * m).sum() / m.sum()

class Net(nn.Module):
    _said = False

    def __init__(self, channels, dropout=0.3, ribbon=False):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(MODEL_ID)
        report = restore_qkv_bias(self.backbone, MODEL_ID)
        if not Net._said:
            print("   " + report, flush=True)
            Net._said = True
        if channels != 3:
            widen(self.backbone, channels)
        self.backbone.gradient_checkpointing_enable()
        d = self.backbone.config.hidden_size
        self.head = ColourHead(d, dropout=dropout)
        self.rib = RibbonHead(d) if ribbon else None

    def steps(self, n_tokens):
        cfg = self.backbone.config
        g = cfg.image_size // cfg.patch_size
        t, rem = divmod(n_tokens, g * g)
        if rem or t < 1:
            raise RuntimeError(f"{n_tokens} tokens is not a whole number of {g}x{g} grids")
        return t

    def forward(self, video, code, ribbon=False):
        tok = self.backbone(pixel_values=video).last_hidden_state
        y = self.head(tok, code)
        if not ribbon or self.rib is None:
            return y
        return y, self.rib(tok, code, self.steps(tok.shape[1]))

def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda")

@torch.no_grad()
def predict(net, ds):
    net.eval()
    out = []
    for j in range(len(ds)):
        v0, c0, _, _ = ds.get(j, swap=False)
        v1, c1, _, _ = ds.get(j, swap=True)
        video = torch.from_numpy(np.stack([v0, v1])).to(DEV)
        code = torch.from_numpy(np.stack([c0, c1])).to(DEV)
        with amp():
            pa = torch.sigmoid(net(video, code).float()).cpu().numpy()
        out.append(0.5 * (pa[0] + 1 - pa[1]))
    return np.array(out)

def run_fold(items, tr, te, kind, channels, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_rib = args.aux_weight > 0
    net = Net(channels, dropout=args.dropout, ribbon=use_rib).to(DEV)
    train = ClipSet([items[i] for i in tr], kind, train=True, augment=args.aug, seed=seed,
                    ribbon=use_rib)
    dl = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(seed))
    bb = [p for n, p in net.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr},
                             {"params": hd, "lr": args.lr_head}], weight_decay=0.05)
    total = max(1, args.epochs * len(dl))
    warm = max(len(dl), int(round(args.warmup * total)))
    frozen = args.freeze_epochs * len(dl)

    def shape(step):
        return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda st: 0.0 if st < frozen else shape(st), shape])

    last = float("nan")
    for ep in range(args.epochs):
        net.train()
        seen = right = 0
        losses, aux_losses = [], []
        for batch in dl:
            if use_rib:
                video, code, target, rib, _ = batch
                rib = rib.to(DEV)
            else:
                video, code, target, _ = batch
                rib = None
            video, code, target = video.to(DEV), code.to(DEV), target.to(DEV)
            with amp():
                out = net(video, code, ribbon=use_rib)
            if use_rib:
                logit, (att_logit, team_logit) = out
                logit = logit.float()
                ya, ma, yt, mt = pool_ribbon(rib, att_logit.shape[1])
                aux = masked_bce(att_logit.float(), ya, ma) + \
                    masked_bce(team_logit.float(), yt, mt)
                aux_losses.append(float(aux))
            else:
                logit, aux = out.float(), None
            main = F.binary_cross_entropy_with_logits(logit, target * 0.9 + 0.05)
            loss = main if aux is None else main + args.aux_weight * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(main.item())
            right += int(((logit > 0).float() == target).sum())
            seen += len(target)
        last = float(np.mean(losses))
        if args.verbose or ep == args.epochs - 1 or ep == args.freeze_epochs - 1:
            tail = f"  aux {np.mean(aux_losses):.3f}" if aux_losses else ""
            phase = " [head only]" if ep < args.freeze_epochs else ""
            print(f"      epoch {ep + 1:>2}/{args.epochs}  loss {last:.3f}{tail}  "
                  f"train acc {right / seen:.0%}{phase}", flush=True)

    fit = score([items[i] for i in tr],
                predict(net, ClipSet([items[i] for i in tr], kind)))["clip"]["acc"]
    p = predict(net, ClipSet([items[i] for i in te], kind))
    del net
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return p, fit, last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, choices=["raw", "raw_aug", "ball", "ball_aug"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--protocol", default="game", choices=["game", "arena"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="epochs with the backbone held still while the head learns")
    ap.add_argument("--warmup", type=float, default=0.05, help="share of steps warming up")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the possession-ribbon auxiliary loss (0 = off); "
                         "any value > 0 adds '_rib' to the run name")
    ap.add_argument("--tag", default="", help="suffix for the run name, so a new "
                                              "configuration never overwrites an old run")
    ap.add_argument("--limit", type=int, help="smoke test on the first N clips; not saved")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.aug = args.input.endswith("_aug")
    kind = args.input.split("_")[0]
    channels = 3 if kind == "raw" else 6

    if DEV != "cuda":
        print("WARNING: no CUDA GPU found - this will run on CPU, roughly 30x slower.\n"
              "Check: python -c \"import torch; print(torch.cuda.is_available())\"")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    items = preflight(load_items())
    if args.limit:
        items = items[:args.limit]
    tag = args.input + ("_rib" if args.aux_weight > 0 else "") \
        + (("_" + args.tag) if args.tag else "")
    per_fold = args.epochs * math.ceil(len(items) * (args.folds - 1) / args.folds / args.batch)
    print(f"{len(items)} clips, input={tag} ({channels} channels), "
          f"protocol={args.protocol}, {args.epochs} epochs "
          f"(~{per_fold} optimiser steps per fold, {args.freeze_epochs} head-only)\n")

    cfg = (f"ep{args.epochs}_b{args.batch}_lr{args.lr}_lh{args.lr_head}"
           f"_fz{args.freeze_epochs}_wu{args.warmup}_do{args.dropout}"
           f"_aux{args.aux_weight}_n{len(items)}")
    for seed in args.seeds:
        name = f"videomae_{tag}_{args.protocol}_s{seed}"
        if not args.limit and os.path.exists(os.path.join(RESULTS, name + ".json")) \
                and not args.force:
            print(f"{name}: already done, skipping (--force to redo)")
            continue
        t0 = time.time()
        p = np.full(len(items), np.nan)
        fits, last_losses = [], []
        pairs = folds(items, args.protocol, args.folds, seed)
        for f, (tr, te) in enumerate(pairs, 1):
            tf = time.time()
            cids = [items[i].cid for i in te]
            done = resume(name, f, cids, cfg, args.force)
            if done is not None:
                p[te], fit, last = done
                fits.append(fit)
                last_losses.append(last)
                print(f"   seed {seed} fold {f}/{len(pairs)}: resumed from checkpoint "
                      f"(train fit {fit:.0%}, final loss {last:.3f})", flush=True)
                continue
            p[te], fit, last = run_fold(items, tr, te, kind, channels, args, seed)
            fits.append(fit)
            last_losses.append(last)
            save_partial(name, f, cids, p[te], fit, last, cfg)
            mem = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
            print(f"   seed {seed} fold {f}/{len(pairs)}: train fit {fit:.0%}, "
                  f"final loss {last:.3f}, {len(te)} test clips, "
                  f"{(time.time() - tf) / 60:.1f} min, peak GPU {mem:.1f} GB "
                  f"[checkpointed]", flush=True)
        flags = plateau_report(name, last_losses)
        s = score(items, p)
        c = s["clip"]
        print(f"{name}: acc {c['acc']:.1%} (n={c['n']})  lift vs constant "
              f"{c['constant']['lift']:+.1%}, vs arena {c['arena']['lift']:+.1%}, "
              f"vs lighter {c['lighter']['lift']:+.1%}   [{(time.time() - t0) / 60:.0f} min]")
        if args.limit:
            print("   (smoke test - not saved)")
            continue
        path = save_run(name, items, p, dict(
            model="videomae", input=tag, protocol=args.protocol, seed=seed,
            epochs=args.epochs, batch=args.batch, lr=args.lr, lr_head=args.lr_head,
            freeze_epochs=args.freeze_epochs, warmup=args.warmup, dropout=args.dropout,
            aux_weight=args.aux_weight, train_fit=fits, final_loss=last_losses,
            plateau=flags, minutes=round((time.time() - t0) / 60, 1),
            gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"))
        clear_partial(name)
        print(f"   saved {path}\n")

if __name__ == "__main__":
    main()
