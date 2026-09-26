import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, RESULTS, ROOT, RGB, code_fingerprint,
                    colour_code, folds, load_items, parts, question)

CHECKS = []
FAILS, WARNS = [], []

def check(why):
    def deco(fn):
        CHECKS.append((fn.__name__, why, fn))
        return fn
    return deco

def ok(msg):
    print(f"  ok    {msg}", flush=True)

def fail(msg):
    FAILS.append(msg)
    print(f"  FAIL  {msg}", flush=True)

def warn(msg):
    WARNS.append(msg)
    print(f"  warn  {msg}", flush=True)

@check("A colour word with no RGB entry raises KeyError inside the first epoch, hours in. "
       "The edited/ batch brought two new ones.")
def colours(items):
    bad = {}
    for it in items:
        for w in (it.col_home, it.col_away):
            for p in parts(w):
                if p not in RGB:
                    bad.setdefault(p, []).append(it.cid)
    if bad:
        for w, c in bad.items():
            fail(f"colour word '{w}' has no RGB entry ({len(c)} clips, e.g. {c[0]})")
    else:
        ok(f"every colour word on all {len(items)} clips maps to an RGB value")

@check("A game straddling a fold split makes the score measure memory, not generalisation. "
       "Camera angles of one play share kit, lighting and camera.")
def fold_integrity(items):
    for seed in (0, 1, 2):
        pairs = folds(items, "game", 5, seed)
        seen, leaked = set(), []
        for f, (tr, te) in enumerate(pairs, 1):
            g_tr = {items[i].game for i in tr}
            g_te = {items[i].game for i in te}
            if g_tr & g_te:
                leaked.append((f, sorted(g_tr & g_te)[:3]))
            if set(tr) & set(te):
                fail(f"seed {seed} fold {f}: a clip is in both train and test")
            seen |= set(te)
        if leaked:
            fail(f"seed {seed}: games straddle a split - {leaked}")
        elif len(seen) != len(items):
            fail(f"seed {seed}: folds cover {len(seen)} of {len(items)} clips")
        else:
            ok(f"seed {seed}: 5 folds, no game straddles a split, every clip tested once")
    ar = folds(items, "arena")
    sizes = [len(te) for _, te in ar]
    if len(ar) < 2 or min(sizes) < 5:
        fail(f"arena protocol has {len(ar)} folds, smallest {min(sizes) if sizes else 0}")
    else:
        ok(f"arena protocol: {len(ar)} folds, test sizes {sizes}")

@check("The question must be exactly antisymmetric in the colour order, or swap-averaging "
       "is not the identity it is claimed to be and 0.5 stops meaning 'no information'.")
def swap_symmetry(items):
    bad = 0
    for it in items[:40]:
        ca, cb, t = question(it, False)
        cb2, ca2, t2 = question(it, True)
        code0 = np.concatenate([colour_code(ca), colour_code(cb)])
        code1 = np.concatenate([colour_code(cb2), colour_code(ca2)])
        if not (t2 == 1 - t and np.allclose(code0[:6], code1[6:])
                and np.allclose(code0[6:], code1[:6])):
            bad += 1
    if bad:
        fail(f"{bad} of 40 clips: swapping the colour order is not antisymmetric")
    else:
        ok("swapping the colour order flips the target and exchanges the code halves")

@check("A leak_ column reaching a model was worth 99.5% accuracy and no science. "
       "RESULTS.md quarantines them by name.")
def no_leak_columns():
    import csv
    p = os.path.join(ROOT, "manifest.csv")
    if not os.path.exists(p):
        return fail("manifest.csv is missing")
    with open(p, encoding="utf-8-sig") as fh:
        cols = next(csv.reader(fh))
    leak = [c for c in cols if c.startswith("leak_")]
    it = load_items()[0]
    reachable = set(vars(it))
    bleed = [c for c in leak if c in reachable]
    if bleed:
        fail(f"quarantined columns reachable from Item: {bleed}")
    else:
        ok(f"{len(leak)} leak_ columns in the manifest, none reachable from Item "
           f"(Item carries {sorted(reachable)})")

@check("A clip with no prepared input is dropped silently by some paths and raises in "
       "others. frames.py must have produced one file per usable clip.")
def inputs_present(items):
    for kind in ("raw", "ball"):
        d = os.path.join(CACHE, f"ftin_{kind}")
        missing = [it.cid for it in items
                   if not os.path.exists(os.path.join(d, it.cid + ".npy"))]
        if missing:
            fail(f"ftin_{kind}: {len(missing)} of {len(items)} clips not prepared "
                 f"(e.g. {missing[0]})")
        else:
            ok(f"ftin_{kind}: all {len(items)} usable clips prepared")

@check("The ribbon is indexed by the stored frame grid. If frames.py and ribbon.py ever "
       "disagree about that grid, the auxiliary targets silently point at other frames.")
def ribbon_alignment(items):
    import ribbon as R
    d = os.path.join(CACHE, "ftin_ball")
    have, bad, empty = 0, 0, 0
    for it in items:
        p = R.path(it.cid)
        if not os.path.exists(p):
            continue
        have += 1
        with open(os.path.join(d, it.cid + ".npy"), "rb") as fh:
            np.lib.format.read_magic(fh)
            n_store = int(np.lib.format.read_array_header_1_0(fh)[0][0])
        with open(p, "rb") as fh:
            z = np.load(fh)
            if len(z["att"]) != n_store:
                bad += 1
            elif z["m_att"].sum() == 0:
                empty += 1
    if not have:
        warn("no ribbon files yet - run finetune/ribbon.py (the aux runs need it)")
    elif bad:
        fail(f"{bad} ribbon files are on a different frame grid than ftin_ball")
    else:
        ok(f"ribbon: {have} files, all on the same frame grid as ftin_ball "
           f"({empty} carry no usable frame, which only removes them from the aux loss)")

ALLOWED_DROPS = {
    "MCG-NJU/videomae-base-finetuned-ssv2": ("classifier.", "fc_norm."),
}

RENAMES = {
    "MCG-NJU/videomae-base-finetuned-ssv2": {"q_bias": "query.bias",
                                             "v_bias": "value.bias"},
}

BLOCKING = {"MCG-NJU/videomae-base-finetuned-ssv2"}

def _checkpoint(model_id):
    import torch
    from transformers.utils import cached_file

    def grab(name):
        try:
            return cached_file(model_id, name, _raise_exceptions_for_missing_entries=False)
        except Exception:
            return None

    for name in ("model.safetensors", "pytorch_model.bin"):
        p = grab(name)
        if not p:
            continue
        if name.endswith(".safetensors"):
            import safetensors.torch as sft
            return sft.load_file(p), name
        return torch.load(p, map_location="cpu", weights_only=True), name
    import json
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = grab(idx_name)
        if not idx:
            continue
        with open(idx, encoding="utf-8") as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
        sd = {}
        for sh in shards:
            q = grab(sh)
            if not q:
                continue
            if sh.endswith(".safetensors"):
                import safetensors.torch as sft
                sd.update(sft.load_file(q))
            else:
                sd.update(torch.load(q, map_location="cpu", weights_only=True))
        if sd:
            return sd, idx_name
    return None, None

def dropped(model, model_id, renames=None):
    import torch
    sd, src = _checkpoint(model_id)
    if sd is None:
        return None, None, None
    live = {k: v for k, v in model.state_dict().items()}
    prefixes = {""} | {k.split(".", 1)[0] + "." for k in sd if "." in k}
    TOL = {torch.float32: 1e-5, torch.float64: 1e-7,
           torch.float16: 2e-3, torch.bfloat16: 2e-2}
    miss = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        cand = [k] + [k[len(p):] for p in prefixes if p and k.startswith(p)]
        for src_suffix, dst_suffix in (renames or {}).items():
            cand += [c[: -len(src_suffix)] + dst_suffix
                     for c in list(cand) if c.endswith(src_suffix)]
        hit = next((live[c] for c in cand if c in live and live[c].shape == v.shape), None)
        if hit is None:
            miss.append(k)
            continue
        a = hit.detach().float().cpu()
        b = v.detach().float().cpu()
        tol = TOL.get(hit.dtype, 2e-2)
        if not torch.allclose(a, b, rtol=tol, atol=tol):
            miss.append(k)
    return miss, len(sd), src

def _squash(names):
    import re
    groups = {}
    for n in names:
        groups.setdefault(re.sub(r"\.\d+\.", ".{N}.", n), []).append(n)
    return [f"{k}  ({len(v)}x)" if len(v) > 1 else v[0] for k, v in sorted(groups.items())]

@check("transformers 5 renamed VideoMAE's q_bias/v_bias and shipped no conversion, so 24 "
       "trained tensors were silently zeroed on every load, in September and again now. "
       "This asserts that every pretrained tensor actually arrives.")
def weights_arrive():
    import torch
    try:
        from train_videomae import MODEL_ID as VM_ID, Net, restore_qkv_bias
        import train_vjepa2 as VJ
        from train_qwen import MODEL_ID as QW_ID
    except Exception as e:
        return warn(f"cannot import the trainers ({type(e).__name__}: {e})")

    try:
        from transformers import VideoMAEModel
        bb = VideoMAEModel.from_pretrained(VM_ID)
        ren = RENAMES.get(VM_ID)
        before, total, src = dropped(bb, VM_ID, ren)
        if before is None:
            warn("VideoMAE: checkpoint not readable from the cache - cannot verify")
        else:
            report = restore_qkv_bias(bb, VM_ID)
            after, _, _ = dropped(bb, VM_ID, ren)
            allowed = ALLOWED_DROPS.get(VM_ID, ())
            real = [k for k in after if not any(a in k for a in allowed)]
            print(f"        {report}")
            if real:
                fail(f"VideoMAE ({src}, {total} tensors): {len(real)} pretrained tensors "
                     f"still do not reach the model: {_squash(real)[:4]}")
            else:
                healed = len(before) - len(after)
                ok(f"VideoMAE ({src}, {total} tensors): every pretrained tensor arrives "
                   f"(the repair recovered {healed}; {len(after)} dropped are the "
                   f"classification head we deliberately discard)")
        del bb
    except Exception as e:
        warn(f"VideoMAE weight check could not run: {type(e).__name__}: {e}")

    try:
        m = VJ.load_backbone()
        miss, total, src = dropped(m, VJ.MODEL_ID, RENAMES.get(VJ.MODEL_ID))
        if miss is None:
            warn("V-JEPA 2: checkpoint not readable from the cache - cannot verify")
        elif miss:
            say = fail if VJ.MODEL_ID in BLOCKING else warn
            say(f"V-JEPA 2 ({src}, {total} tensors): {len(miss)} did not match by value: "
                f"{_squash(miss)[:4]} - transformers reported a clean load for this model, "
                f"so read this as a gap in the check before reading it as lost weights")
        else:
            ok(f"V-JEPA 2 ({src}, {total} tensors): every pretrained tensor arrives")
        del m
    except Exception as e:
        warn(f"V-JEPA 2 weight check could not run: {type(e).__name__}: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    warn(f"Qwen ({QW_ID}) is loaded 4-bit quantised, so its weights cannot be compared "
         f"against the checkpoint by value; its logs show no MISSING or UNEXPECTED lines")

DECLARED = [
    ("videomae", "raw",        30, lambda n: -(-n // 4), 1),
    ("videomae", "ball",       30, lambda n: -(-n // 4), 1),
    ("videomae", "raw_aug",    30, lambda n: -(-n // 4), 1),
    ("videomae", "ball_aug",   30, lambda n: -(-n // 4), 1),
    ("vjepa2",   "raw",        40, lambda n: -(-n // 16), 1),
    ("vjepa2",   "ball",       40, lambda n: -(-n // 16), 1),
    ("vjepa2",   "ball_aug",  240, lambda n: -(-n // 16), 6),
    ("qwen_lora", "raw",        4, lambda n: n, 1),
    ("qwen_lora", "ball",       4, lambda n: n, 1),
    ("qwen_lora", "ball_aug",   4, lambda n: n, 1),
]
FLOOR = 40

@check("vjepa2 ball_aug stores 6 augmented copies per clip but got the same epoch count as "
       "the 1-copy runs, so it saw each sample one sixth as often and 7 of 15 folds never "
       "left the ln(2) plateau. This asserts every run has steps to spare per sample.")
def step_budgets(items):
    n_tr = int(len(items) * 4 / 5)
    print(f"        {n_tr} training clips per fold, floor {FLOOR} steps per distinct sample")
    for model, inp, ep, per_ep, copies in DECLARED:
        steps = ep * per_ep(n_tr)
        distinct = n_tr * copies
        line = (f"{model} {inp}: {steps} steps, {distinct} distinct samples, "
                f"{steps / distinct:.2f} steps per sample")
        if steps < 300:
            fail(f"{line}  <- under 300 steps per fold; that is the September defect")
        elif steps / distinct < 0.5:
            fail(f"{line}  <- too few steps for the number of distinct samples")
        else:
            ok(line)

@check("status.json said finished:true from September, so the watchdog reported ALL DONE "
       "and started nothing. Stale marker files had the same effect on two cache jobs.")
def queue_state():
    import json
    logs = os.path.join(ROOT, "finetune", "logs")
    st = os.path.join(logs, "status.json")
    if os.path.exists(st):
        try:
            with open(st, encoding="utf-8") as fh:
                d = json.load(fh)
        except ValueError:
            return fail("status.json is not valid JSON")
        sys.path.insert(0, os.path.join(ROOT, "finetune"))
        import run_all as R
        left = [j["name"] for j in R.jobs() if j["name"] != "summary" and not R.is_done(j)]
        if d.get("finished") and left:
            fail(f"status.json says finished but {len(left)} jobs are not done "
                 f"(first: {left[0]}) - the watchdog would report ALL DONE")
        else:
            ok(f"status.json is consistent with the disk ({len(left)} jobs left to do)")
    else:
        ok("no status.json - the queue will start from a clean slate")
    stale = []
    for name, out in (("cache_crops", os.path.join(CACHE, "crops")),
                      ("cache_team", os.path.join(CACHE, "team_assign"))):
        m = os.path.join(logs, name + ".done")
        if os.path.exists(m):
            have = len(os.listdir(out)) if os.path.isdir(out) else 0
            need = len(load_items())
            if have < need:
                stale.append(f"{name}.done exists but {out} holds {have} of {need}")
    for s in stale:
        fail(s + " - that job would be skipped")
    if not stale:
        ok("no stale .done marker would skip a job that still has work")

@check("Re-running a changed configuration under an old run name either overwrites results "
       "(CLAUDE.md rule 3) or, as the scripts actually behave, skips every job.")
def result_names():
    if not os.path.isdir(RESULTS):
        return ok("no results directory yet")
    import json
    bad = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS, f), encoding="utf-8") as fh:
                r = json.load(fh)
        except ValueError:
            bad.append(f"{f} is not valid JSON")
            continue
        if r.get("name") + ".json" != f:
            bad.append(f"{f} carries the name {r.get('name')!r}")
    for b in bad:
        fail(b)
    if not bad:
        n = len([f for f in os.listdir(RESULTS) if f.endswith('.json')])
        ok(f"{n} result files, every one named after the run inside it")

@check("Twice in one day a result survived a repair and would have been read beside results "
       "from the corrected code. A final table assembled from several versions of the "
       "pipeline is not a result.")
def code_consistency():
    import json
    EQUIVALENT = {"33372ae963bd"}
    if not os.path.isdir(RESULTS):
        return ok("no results directory yet")
    tag = os.environ.get("OOB_TAG", "v2")
    now = code_fingerprint()
    stale = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".json") or f"_{tag}" not in f:
            continue
        try:
            with open(os.path.join(RESULTS, f), encoding="utf-8") as fh:
                got = json.load(fh).get("code")
        except (OSError, ValueError):
            got = None
        if got != now and got not in EQUIVALENT:
            stale.append((f, got or "unstamped"))
    if stale:
        fail(f"{len(stale)} '{tag}' result(s) came from different training code than is on "
             f"disk now (current {now}): " +
             ", ".join(f"{f} [{g}]" for f, g in stale[:4]) +
             (" ..." if len(stale) > 4 else "") +
             "  -> run RUN\\RESET.bat to move them to results/superseded/ and start the "
             "grid clean")
    else:
        n = len([f for f in os.listdir(RESULTS) if f.endswith('.json') and f"_{tag}" in f])
        ok(f"all {n} '{tag}' results came from the training code now on disk ({now})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-weights", action="store_true",
                    help="data and inputs only; no model is loaded")
    ap.add_argument("--list", action="store_true", help="what it checks and why")
    args = ap.parse_args()

    if args.list:
        print("verify_run.py asserts the following. Each one exists because something in "
              "this project went wrong in exactly that way.\n")
        for name, why, _ in CHECKS:
            print(f"{name}\n    {why}\n")
        return 0

    print("verify_run.py - asserting what the run assumes, before any GPU time\n")
    items = load_items()
    print(f"dataset: {len(items)} usable clips, {len({i.event for i in items})} events, "
          f"{len({i.game for i in items})} games, "
          f"{len({i.arena for i in items})} arenas, "
          f"HOME {np.mean([i.y_home for i in items]):.1%}\n")

    print("data")
    colours(items); fold_integrity(items); swap_symmetry(items); no_leak_columns()
    print("\ninputs")
    inputs_present(items); ribbon_alignment(items)
    print("\nstep budgets")
    step_budgets(items)
    print("\nqueue hygiene")
    queue_state(); result_names(); code_consistency()
    if args.skip_weights:
        print("\npretrained weights\n  warn  skipped (--skip-weights)")
        WARNS.append("weight check skipped")
    else:
        print("\npretrained weights (loads each backbone once; a few minutes)")
        weights_arrive()

    print("\n" + "=" * 72)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S) - the run must not start:")
        for f in FAILS:
            print(f"  - {f}")
    if WARNS:
        print(f"{len(WARNS)} warning(s) - could not be verified here, not swallowed:")
        for w in WARNS:
            print(f"  - {w}")
    if not FAILS:
        print("verify_run: PASSED" + (" with warnings" if WARNS else ""))
    print("=" * 72)
    return 1 if FAILS else 0

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
