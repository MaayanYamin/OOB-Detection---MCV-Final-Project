import argparse
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LN2, load_items, preflight

TOL = 0.05
FIT = 0.90
STEPS = 300

def _args(**kw):
    return types.SimpleNamespace(**kw)

def videomae(items, idx, steps, aux_weight):
    import train_videomae as T
    batch = 4
    per_ep = max(1, int(np.ceil(len(idx) / batch)))
    ep = max(2, int(round(steps / per_ep)))
    a = _args(epochs=ep, batch=batch, lr=3e-5, lr_head=1e-3, freeze_epochs=max(1, ep // 10),
              warmup=0.05, dropout=0.3, aux_weight=aux_weight, aug=False, verbose=False)
    p, fit, last = T.run_fold(items, idx, idx, "ball", 6, a, 0)
    return p, fit, last, ep * per_ep

def vjepa2(items, idx, steps, aux_weight):
    import train_vjepa2 as T
    T.extract([items[i] for i in idx], "ball")
    batch = 16
    per_ep = max(1, int(np.ceil(len(idx) / batch)))
    ep = max(2, int(round(steps / per_ep)))
    a = _args(epochs=ep, batch=batch, lr=5e-4, dropout=0.3, wd=0.05)
    p, fit, last = T.run_fold(items, idx, idx, "ball", a, 0)
    return p, fit, last, ep * per_ep

def qwen(items, idx, steps, aux_weight):
    import train_qwen as T
    ep = max(2, int(round(steps / max(1, len(idx)))))
    a = _args(input="ball", model_id=T.MODEL_ID, bits=4, epochs=ep, accum=1, lr=2e-4,
              warmup=0.1)
    p, fit, last = T.run_fold(items, idx, idx, a, 0)
    return p, fit, last, ep * len(idx)

def videomae_rib(items, idx, steps, aux_weight):
    return videomae(items, idx, steps, aux_weight or 0.3)

MODELS = {"videomae": videomae, "videomae_rib": videomae_rib,
          "vjepa2": vjepa2, "qwen": qwen}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n", type=int, default=16, help="clips the model may memorise")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="override the ribbon weight; videomae_rib uses 0.3 by default")
    args = ap.parse_args()

    items = preflight(load_items())
    if len(items) < args.n:
        sys.exit(f"only {len(items)} clips available, need {args.n}")
    idx = np.arange(args.n)
    print(f"\nescape check: {args.n} clips, train == test, ~{args.steps} steps per model.\n"
          f"pass = final loss < {LN2 - TOL:.3f} (ln2 - {TOL}) and train fit >= {FIT:.0%}\n")

    rows, ok, crashed = [], True, []
    for m in args.model:
        t0 = time.time()
        try:
            p, fit, last, steps = MODELS[m](items, idx, args.steps, args.aux_weight)
        except Exception as e:
            print(f"  {m:<10} CRASHED  {type(e).__name__}: {e}", flush=True)
            rows.append((m, np.nan, np.nan, 0, "crashed"))
            crashed.append(m)
            continue
        spread = float(np.std(p))
        escaped = np.isfinite(last) and last < LN2 - TOL
        fitted = fit >= FIT
        verdict = "pass" if (escaped and fitted) else "FAIL"
        ok &= escaped and fitted
        why = "" if verdict == "pass" else (
            "  <- stuck at ln(2): the optimiser never started" if not escaped
            else "  <- left the plateau but did not memorise: try more steps or a higher lr")
        print(f"  {m:<10} {verdict}  final loss {last:.3f}  train fit {fit:.0%}  "
              f"prediction sd {spread:.3f}  ({steps} steps, {(time.time()-t0)/60:.1f} min)"
              f"{why}", flush=True)
        rows.append((m, last, fit, steps, verdict))

    print("\nprediction sd near 0 means the model answers 0.5 for every clip, which is what"
          "\nswap-averaged evaluation returns for a model that ignores the colour code.")
    if crashed:
        print(f"\nescape check: COULD NOT RUN for {', '.join(crashed)}. That is a fault in"
              "\nthis check or its environment, not proof that a model cannot learn, so the"
              "\nqueue is allowed to continue. Read the traceback above before trusting the"
              "\nresults it produces.", flush=True)
        return 2
    print("\nescape check:", "PASSED" if ok else "FAILED - do not start the queue", flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
