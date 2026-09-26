import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "cache")
RESULTS = os.path.join(ROOT, "finetune", "results")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from clip_index import load_index

RGB = {"white": (.95, .95, .95), "black": (.08, .08, .08), "red": (.80, .10, .15),
       "blue": (.10, .30, .75), "darkblue": (.05, .12, .35), "lightblue": (.45, .70, .90),
       "green": (.00, .50, .25), "purple": (.40, .20, .55), "yellow": (.98, .78, .10),
       "orange": (.95, .45, .10),
       "darkred": (.45, .08, .16), "lightgreen": (.55, .85, .50)}
COMPOUND = {"whiteandlightblue": ("white", "lightblue"),
            "blackandpurple": ("black", "purple"), "blueandred": ("blue", "red")}
PRETTY = {"whiteandlightblue": "white and light blue", "blackandpurple": "black and purple",
          "blueandred": "blue and red", "lightblue": "light blue", "darkblue": "dark blue"}

def parts(word):
    return COMPOUND.get(word, (word, word))

def colour_code(word):
    a, b = parts(word)
    for w in (a, b):
        if w not in RGB:
            raise KeyError(f"colour word '{w}' has no RGB entry in finetune/common.py")
    return np.array(RGB[a] + RGB[b], np.float32)

def pretty(word):
    return PRETTY.get(word, word)

def lightness(word):
    return float(np.mean(RGB[parts(word)[0]]))

@dataclass
class Item:
    cid: str
    path: str
    event: str
    game: str
    arena: str
    home: str
    away: str
    y_home: int
    col_home: str
    col_away: str

def load_items():
    out = []
    for c in sorted(load_index(ROOT).values(), key=lambda c: c.cid):
        if c.label not in ("HOME", "AWAY") or not (c.home in c.colours and c.away in c.colours):
            continue
        out.append(Item(c.cid, c.path, c.event_id, c.event_id.rsplit("_P", 1)[0], c.home,
                        c.home, c.away, int(c.label == "HOME"),
                        c.colours[c.home], c.colours[c.away]))
    return out

def question(it, swap):
    if swap:
        return it.col_away, it.col_home, 1 - it.y_home
    return it.col_home, it.col_away, it.y_home

def folds(items, scheme="game", k=5, seed=0):
    y = np.array([it.y_home for it in items])
    if scheme == "arena":
        ar = np.array([it.arena for it in items])
        return [(np.where(ar != a)[0], np.where(ar == a)[0]) for a in sorted(set(ar))]
    from sklearn.model_selection import StratifiedGroupKFold
    g = np.array([it.game for it in items])
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, g))

LN2 = float(np.log(2))

def plateau(final_loss, tol=0.03):
    return bool(np.isfinite(final_loss) and abs(float(final_loss) - LN2) <= tol)

def plateau_report(name, final_losses):
    flags = [plateau(v) for v in final_losses]
    if any(flags):
        bad = [i + 1 for i, f in enumerate(flags) if f]
        print(f"   !! {name}: {sum(flags)}/{len(flags)} folds ended at ln(2)={LN2:.4f} "
              f"(folds {bad}) - these folds DID NOT TRAIN. Their test numbers measure the "
              f"optimiser, not the task; see TRAINING_FAILURE.md.", flush=True)
    else:
        print(f"   ok {name}: every fold left the ln(2) plateau "
              f"(final losses {', '.join(f'{v:.3f}' for v in final_losses)})", flush=True)
    return flags

def preflight(items):
    bad = {}
    for it in items:
        for w in (it.col_home, it.col_away):
            for part in parts(w):
                if part not in RGB:
                    bad.setdefault(part, []).append(it.cid)
    if bad:
        lines = "\n".join(f"  '{w}' on {len(c)} clips, e.g. {c[0]}" for w, c in bad.items())
        raise KeyError(f"colour words with no RGB entry in finetune/common.py:\n{lines}")
    n_ev = len({it.event for it in items})
    n_ar = len({it.arena for it in items})
    print(f"preflight ok: {len(items)} clips, {n_ev} events, "
          f"{len({it.game for it in items})} games, {n_ar} arenas, "
          f"HOME {np.mean([it.y_home for it in items]):.1%}")
    return items

def _bars(y, arena, light):
    const = np.full(len(y), int(y.mean() >= 0.5))
    ar = np.array([int(y[arena == a].mean() >= 0.5) for a in arena])
    return {"constant": const, "arena": ar, "lighter": light}

def _report(y, pred, bars):
    ok = (pred == y).astype(float)
    out = {"n": int(len(y)), "acc": float(ok.mean())}
    for name, b in bars.items():
        bo = (b == y).astype(float)
        d = ok - bo
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        out[name] = {"bar": float(bo.mean()), "lift": float(d.mean()),
                     "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
    return out

def score(items, p_home):
    p = np.asarray(p_home, float)
    ok = np.isfinite(p)
    its = [it for it, k in zip(items, ok) if k]
    p = p[ok]
    y = np.array([it.y_home for it in its])
    arena = np.array([it.arena for it in its])
    light = np.array([int(lightness(it.col_home) >= lightness(it.col_away)) for it in its])
    clip = _report(y, (p > 0.5).astype(int), _bars(y, arena, light))

    ev = defaultdict(list)
    for i, it in enumerate(its):
        ev[it.event].append(i)
    rows = list(ev.values())
    ye = np.array([y[r[0]] for r in rows])
    pe = np.array([p[r].mean() for r in rows])
    ae = np.array([arena[r[0]] for r in rows])
    le = np.array([light[r[0]] for r in rows])
    event = _report(ye, (pe > 0.5).astype(int), _bars(ye, ae, le))
    return {"clip": clip, "event": event}

CODE_FILES = ("common.py", "frames.py", "ribbon.py",
              "train_videomae.py", "train_vjepa2.py", "train_qwen.py")

def code_fingerprint():
    import hashlib
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in CODE_FILES:
        try:
            with open(os.path.join(here, name), "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"missing:" + name.encode())
    return h.hexdigest()[:12]

PARTIAL = os.path.join(RESULTS, "partial")

def _partial_path(name):
    return os.path.join(PARTIAL, name + ".json")

def load_partial(name):
    try:
        with open(_partial_path(name), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}

def resume(name, fold, cids, cfg, force=False):
    got = load_partial(name).get(str(fold))
    if force or not got or got.get("cids") != list(cids) or got.get("cfg") != cfg:
        return None
    try:
        return (np.array(got["p"], float), float(got["train_fit"]),
                float(got["final_loss"]))
    except (KeyError, TypeError, ValueError):
        return None

def save_partial(name, fold, cids, p_te, fit, loss, cfg):
    os.makedirs(PARTIAL, exist_ok=True)
    d = load_partial(name)
    d[str(fold)] = {"cids": list(cids), "cfg": cfg,
                    "p": [None if not np.isfinite(v) else float(v) for v in p_te],
                    "train_fit": float(fit), "final_loss": float(loss),
                    "saved": __import__("datetime").datetime.now().isoformat(
                        timespec="seconds")}
    tmp = _partial_path(name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, _partial_path(name))
    return len(d)

def clear_partial(name):
    try:
        os.remove(_partial_path(name))
    except OSError:
        pass

def save_run(name, items, p_home, info):
    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(info, name=name, code=code_fingerprint(), clips=[it.cid for it in items],
               p_home=[None if not np.isfinite(v) else float(v) for v in p_home],
               score=score(items, p_home))
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return path

def summarize():
    runs = defaultdict(list)
    flat = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        key = (r["model"], r["input"], r["protocol"])
        runs[key].append(r["score"])
        flat[key] += list(r.get("plateau") or [])
    if not runs:
        print("no finished runs in", RESULTS)
        return
    print(f"{'model':<10}{'input':<11}{'split':<7}{'seeds':>6}{'acc':>8}{'sd':>6}"
          f"{'n':>5}  lift vs: constant   arena  lighter   (clip level; event level below)")
    for level in ("clip", "event"):
        if level == "event":
            print("\nevent level (camera angles of one play averaged):")
        for (m, inp, prot), sc in sorted(runs.items()):
            acc = np.array([s[level]["acc"] for s in sc])
            lifts = [np.mean([s[level][b]["lift"] for s in sc])
                     for b in ("constant", "arena", "lighter")]
            spread = f"{acc.std():.1%}" if len(sc) > 1 else "-"
            print(f"{m:<10}{inp:<11}{prot:<7}{len(sc):>6}{acc.mean():>8.1%}"
                  f"{spread:>6}{sc[0][level]['n']:>5}  "
                  + "  ".join(f"{v:>+7.1%}" for v in lifts))
    dead = {k: v for k, v in flat.items() if any(v)}
    if dead:
        print("\nfolds that ended at ln(2) and therefore never trained "
              "(TRAINING_FAILURE.md) - do not read their accuracy as a result:")
        for (m, inp, prot), v in sorted(dead.items()):
            print(f"  {m:<10}{inp:<11}{prot:<7}{sum(v):>3}/{len(v)} folds")
    elif any(flat.values()) or flat:
        print("\nno fold ended at ln(2): every run that recorded its loss did train.")
    print("\nA lift counts only if it holds across seeds AND its interval (in each JSON)"
          " clears zero. Report every row, not the best one (RESULTS.md 4d).")

if __name__ == "__main__":
    items = preflight(load_items())
    print(f"{len(items)} clips, {len({i.event for i in items})} events, "
          f"{len({i.game for i in items})} games\n")
    summarize()
