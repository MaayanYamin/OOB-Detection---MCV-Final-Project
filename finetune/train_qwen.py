import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, clear_partial, folds, load_items, parts,
                    plateau_report, preflight, pretty, question, resume,
                    save_partial, save_run, score)
from frames import ClipSet

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_FRAMES = 8
PIX = 224
DECOYS = ["red", "green", "yellow", "purple", "orange", "black", "white", "blue"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def dtype_kw(dt):
    import transformers
    try:
        v5 = int(transformers.__version__.split(".")[0]) >= 5
    except (AttributeError, ValueError):
        v5 = False
    return {"dtype": dt} if v5 else {"torch_dtype": dt}

def load(model_id, lora, bits):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
    try:
        proc = AutoProcessor.from_pretrained(model_id, min_pixels=PIX * PIX,
                                             max_pixels=PIX * PIX)
    except (TypeError, ValueError):
        proc = AutoProcessor.from_pretrained(
            model_id, size={"shortest_edge": PIX * PIX, "longest_edge": PIX * PIX})
    kw = dict(device_map={"": 0} if DEV == "cuda" else None, **dtype_kw(torch.bfloat16))
    if bits == 4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = Model.from_pretrained(model_id, **kw)
    if lora:
        from peft import LoraConfig, get_peft_model
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    ids = [proc.tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B")]
    if not all(len(i) == 1 for i in ids):
        raise RuntimeError(f"'A'/'B' are not single tokens for this tokenizer: {ids}")
    return model, proc, ids[0][0], ids[1][0]

def question_text(ca, cb, drawn):
    a, b = pretty(ca), pretty(cb)
    note = (f" Players outlined in bright green are the {a} team, players outlined in "
            f"magenta are the {b} team, and the ball is circled in cyan." if drawn else "")
    return (f"These are frames, in order, from an NBA replay just before the ball went out "
            f"of bounds. One team wears {a}, the other wears {b}.{note} Which team touched "
            f"the ball last? Answer with one letter: A for the {a} team, B for the {b} team.")

def diff(model, proc, id_a, id_b, frames, text):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs]
             + [{"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=imgs, return_tensors="pt").to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        last = model(**enc).logits[0, -1].float()
    return last[id_a] - last[id_b]

@torch.no_grad()
def predict(model, proc, id_a, id_b, items, idx, kind):
    model.eval()
    ds = ClipSet(items, kind, t=N_FRAMES)
    out = []
    for i in idx:
        pa = []
        for swap in (False, True):
            ca, cb, _ = question(items[i], swap)
            d = diff(model, proc, id_a, id_b, ds.get_rgb(i, swap), question_text(ca, cb, kind == "ball"))
            pa.append(torch.sigmoid(d).item())
        out.append(0.5 * (pa[0] + 1 - pa[1]))
    return np.array(out)

@torch.no_grad()
def control(model, proc, id_a, id_b, items, kind):
    ds = ClipSet(items, kind, t=N_FRAMES)
    right = 0
    for i, it in enumerate(items):
        worn = set(parts(it.col_home)) | set(parts(it.col_away))
        decoy = next(c for c in DECOYS if c not in worn)
        p = []
        for first_true in (True, False):
            a, b = (it.col_home, decoy) if first_true else (decoy, it.col_home)
            text = (f"These are frames from an NBA replay. Which of these two jersey colours "
                    f"is worn by players in these frames? Answer with one letter: A for "
                    f"{pretty(a)}, B for {pretty(b)}.")
            d = diff(model, proc, id_a, id_b, ds.get_rgb(i, False), text)
            pa = torch.sigmoid(d).item()
            p.append(pa if first_true else 1 - pa)
        right += int(np.mean(p) > 0.5)
    return right / len(items)

def run_fold(items, tr, te, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kind = args.input.split("_")[0]
    model, proc, id_a, id_b = load(args.model_id, True, args.bits)
    train = ClipSet(items, kind, augment=args.input.endswith("_aug"), t=N_FRAMES, seed=seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    total = max(1, args.epochs * len(tr) // max(1, args.accum))
    warm = max(1, int(round(args.warmup * total)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm)
        * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))

    n = 0
    last = float("nan")
    for ep in range(args.epochs):
        model.train()
        losses = []
        for i in rng.permutation(tr):
            swap = bool(rng.random() < 0.5)
            ca, cb, t = question(items[i], swap)
            d = diff(model, proc, id_a, id_b, train.get_rgb(i, swap),
                     question_text(ca, cb, kind == "ball"))
            loss = F.binary_cross_entropy_with_logits(d, torch.tensor(t * 0.9 + 0.05, device=DEV))
            (loss / args.accum).backward()
            losses.append(loss.item())
            n += 1
            if n % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        last = float(np.mean(losses))
        print(f"      epoch {ep + 1}/{args.epochs}  loss {last:.3f}  "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    opt.zero_grad(set_to_none=True)
    fit = score([items[i] for i in tr], predict(model, proc, id_a, id_b, items, tr, kind))["clip"]["acc"]
    p = predict(model, proc, id_a, id_b, items, te, kind)
    del model, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return p, fit, last

def report(name, items, p, t0):
    c = score(items, p)["clip"]
    print(f"{name}: acc {c['acc']:.1%} (n={c['n']})  lift vs constant "
          f"{c['constant']['lift']:+.1%}, vs arena {c['arena']['lift']:+.1%}, "
          f"vs lighter {c['lighter']['lift']:+.1%}   [{(time.time() - t0) / 60:.1f} min]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zeroshot", "lora"])
    ap.add_argument("--input", required=True, choices=["raw", "ball", "ball_aug"])
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--protocol", default="game", choices=["game", "arena"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.1, help="share of steps warming up")
    ap.add_argument("--tag", default="", help="suffix for the run name, so a new "
                                              "configuration never overwrites an old run")
    ap.add_argument("--limit", type=int, help="smoke test on the first N clips; not saved")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}" if DEV == "cuda"
          else "WARNING: no CUDA GPU found - Qwen needs the GPU")
    items = preflight(load_items())
    if args.limit:
        items = items[:args.limit]
    kind = args.input.split("_")[0]
    tag = args.input + (("_" + args.tag) if args.tag else "")

    if args.mode == "zeroshot":
        if args.input.endswith("_aug"):
            sys.exit("zero-shot has no training, so augmentation does not apply")
        name = f"qwen_zs_{tag}"
        if not args.limit and os.path.exists(os.path.join(RESULTS, name + ".json")) \
                and not args.force:
            print(f"{name}: already done, skipping (--force to redo)")
            return
        t0 = time.time()
        model, proc, id_a, id_b = load(args.model_id, False, args.bits)
        ctrl = control(model, proc, id_a, id_b, items, kind)
        print(f"control - picks the colour actually worn: {ctrl:.0%} of clips "
              f"(near 50% would mean it cannot see the jerseys)", flush=True)
        p = predict(model, proc, id_a, id_b, items, range(len(items)), kind)
        report(name, items, p, t0)
        if not args.limit:
            print("   saved", save_run(name, items, p, dict(
                model="qwen_zs", input=tag, protocol="zeroshot", seed=0,
                backbone=args.model_id, control_colour_acc=ctrl,
                minutes=round((time.time() - t0) / 60, 1))))
        return

    cfg = (f"ep{args.epochs}_ac{args.accum}_lr{args.lr}_wu{args.warmup}"
           f"_b{args.bits}_n{len(items)}")
    steps = args.epochs * int(len(items) * (args.folds - 1) / args.folds) // max(1, args.accum)
    print(f"{len(items)} clips, input={tag}, {args.epochs} epochs at accum {args.accum} "
          f"-> ~{steps} optimiser steps per fold (September ran 37)\n")
    for seed in args.seeds:
        name = f"qwen_lora_{tag}_{args.protocol}_s{seed}"
        if not args.limit and os.path.exists(os.path.join(RESULTS, name + ".json")) \
                and not args.force:
            print(f"{name}: already done, skipping (--force to redo)")
            continue
        t0 = time.time()
        p = np.full(len(items), np.nan)
        fits, last_losses = [], []
        for f, (tr, te) in enumerate(folds(items, args.protocol, args.folds, seed), 1):
            tf = time.time()
            cids = [items[i].cid for i in te]
            done = resume(name, f, cids, cfg, args.force)
            if done is not None:
                p[te], fit, last = done
                fits.append(fit)
                last_losses.append(last)
                print(f"   seed {seed} fold {f}: resumed from checkpoint "
                      f"(train fit {fit:.0%}, final loss {last:.3f})", flush=True)
                continue
            p[te], fit, last = run_fold(items, tr, te, args, seed)
            fits.append(fit)
            last_losses.append(last)
            save_partial(name, f, cids, p[te], fit, last, cfg)
            mem = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
            print(f"   seed {seed} fold {f}: train fit {fit:.0%}, final loss {last:.3f}, "
                  f"{len(te)} test clips, {(time.time() - tf) / 60:.1f} min, "
                  f"peak GPU {mem:.1f} GB [checkpointed]", flush=True)
        flags = plateau_report(name, last_losses)
        report(name, items, p, t0)
        if args.limit:
            print("   (smoke test - not saved)")
            continue
        print("   saved", save_run(name, items, p, dict(
            model="qwen_lora", input=tag, protocol=args.protocol, seed=seed,
            backbone=args.model_id, epochs=args.epochs, lr=args.lr, accum=args.accum,
            warmup=args.warmup, train_fit=fits, final_loss=last_losses, plateau=flags,
            minutes=round((time.time() - t0) / 60, 1))), "\n")
        clear_partial(name)

if __name__ == "__main__":
    main()
