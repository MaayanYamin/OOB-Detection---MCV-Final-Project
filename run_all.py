import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
CACHE = os.path.join(ROOT, "cache", "cache")
RESULTS = os.path.join(ROOT, "finetune", "results")
ANALYSIS = os.path.join(ROOT, "analysis")
PREDICTIONS = os.path.join(ANALYSIS, "predictions.csv")
STATUS = os.path.join(LOGS, "status.json")
PY = sys.executable
ATTEMPTS = 2

NAMES = {
    "majority": "majority class (the bar)",
    "ballcolour_last24": "jersey colour around the ball",
    "ballcolour_last8": "jersey colour around the ball, last 8 frames",
    "ballcolour_all": "jersey colour around the ball, whole clip",
    "nearest_player": "team of the player nearest the ball",
    "players_colour": "colour of the players near the ball",
    "clip_zeroshot_frame": "CLIP zero-shot, whole frame",
    "clip_zeroshot_ballcrop": "CLIP zero-shot, ball window",
    "mobilenet_linear": "MobileNet features + linear head",
    "kinematic_linear": "ball and player kinematics + linear head",
    "ribbon_baseline": "last player attached to the ball, no training",
    "videomae_raw_game": "VideoMAE frozen + head, whole frame",
    "videomae_ball_game": "VideoMAE frozen + head, ball window",
    "vjepa2_raw_game": "V-JEPA 2 frozen + head, whole frame",
    "vjepa2_ball_game": "V-JEPA 2 frozen + head, ball window",
    "poc_mobilenet_lstm_game": "MobileNetV2 + LSTM",
    "qwen_zs_raw_v2": "Qwen2.5-VL zero-shot, whole frame",
    "qwen_zs_ball_v2": "Qwen2.5-VL zero-shot, ball window",
    "qwen_lora_ball_v2_game": "Qwen2.5-VL QLoRA, ball window",
    "vjepa2_raw_v2_game": "V-JEPA 2 + head, whole frame (GPU grid)",
    "vjepa2_ball_v2_game": "V-JEPA 2 + head, ball window (GPU grid)",
    "vjepa2_ball_aug_v2_game": "V-JEPA 2 + head, ball window, augmented (GPU grid)",
    "vjepa2_ball_v2_arena": "V-JEPA 2 + head, ball window, unseen arenas",
    "videomae_ball_v2_game": "VideoMAE fine-tuned, ball window (GPU grid)",
    "videomae_ball_rib_v2_game": "VideoMAE fine-tuned + possession ribbon (GPU grid)",
    "videomae_ball_rib_v2_arena": "VideoMAE + ribbon, unseen arenas",
}
GPU_GRID_RUNS = 22


def n_clips():
    return len(glob.glob(os.path.join(ROOT, "clips", "clips", "*.mp4")))


def count(sub, ext="npz"):
    return len(glob.glob(os.path.join(CACHE, sub, "*." + ext)))


def cached(sub, ext="npz", slack=0):
    return lambda: bool(n_clips()) and count(sub, ext) >= n_clips() - slack


def has(*rel):
    return lambda: all(os.path.exists(os.path.join(ROOT, p)) for p in rel)


def seeds(prefix, which=(0, 1, 2)):
    return lambda: all(os.path.exists(os.path.join(RESULTS, "%s_s%d.json" % (prefix, s)))
                       for s in which)


def in_predictions(method):
    def check():
        if not os.path.exists(PREDICTIONS):
            return False
        with open(PREDICTIONS, encoding="utf-8") as fh:
            return any(("," + method + ",") in line for line in fh)
    return check


def grid_done():
    return len(glob.glob(os.path.join(RESULTS, "*_v2*.json"))) >= GPU_GRID_RUNS


def marker(name):
    return lambda: os.path.exists(os.path.join(LOGS, name + ".done"))


def cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def stage(name, cmd, done, mins, gpu=False):
    return dict(name=name, cmd=[PY] + cmd, done=done or marker(name), mins=mins, gpu=gpu,
                mark=done is None)


STAGES = [
    stage("detect", ["scripts/cache_gpu.py", "--stage", "detect", "--cache", "cache/cache"],
          cached("detect"), 600),
    stage("pose", ["scripts/cache_gpu.py", "--stage", "pose", "--cache", "cache/cache"],
          cached("pose"), 420),
    stage("setup_ball", ["scripts/setup_wasb.py"],
          has("third_party/WASB-SBDT/src", "monotrack_basketball.pth.tar"), 5),
    stage("ball", ["scripts/run_ball_all.py", "--repo", "third_party/WASB-SBDT",
                   "--weights", "monotrack_basketball.pth.tar"], cached("ball"), 300),
    stage("crops", ["scripts/cache_crops.py", "--all"], cached("crops"), 60),
    stage("team", ["scripts/team_by_template.py"], has("cache/cache/team_template.npz"), 25),
    stage("props", ["scripts/clip_props.py"], has("cache/cache/clip_props.csv"), 10),
    stage("methods", ["scripts/methods_all.py", "--methods", "all"],
          in_predictions("ballcolour_last24"), 180),
    stage("frames_raw", ["finetune/frames.py", "--kind", "raw"],
          cached("ftin_raw", "npy", slack=1), 35),
    stage("frames_ball", ["finetune/frames.py", "--kind", "ball"],
          cached("ftin_ball", "npy", slack=1), 45),
    stage("ribbon", ["finetune/ribbon.py"], cached("ribbon", "npz", slack=1), 20),
    stage("ribbon_baseline", ["finetune/ribbon.py", "--baseline"], None, 5),
    stage("vjepa2_raw", ["finetune/train_frozen.py", "--backbone", "vjepa2", "--input", "raw"],
          seeds("vjepa2_raw_game"), 150),
    stage("videomae_raw", ["finetune/train_frozen.py", "--backbone", "videomae",
                           "--input", "raw"], seeds("videomae_raw_game"), 110),
    stage("vjepa2_ball", ["finetune/train_frozen.py", "--backbone", "vjepa2", "--input", "ball",
                          "--seeds", "0"], seeds("vjepa2_ball_game", (0,)), 150),
    stage("videomae_ball", ["finetune/train_frozen.py", "--backbone", "videomae",
                            "--input", "ball", "--seeds", "0"],
          seeds("videomae_ball_game", (0,)), 110),
    stage("poc", ["poc/run_poc_on_corpus.py"],
          has("finetune/results/poc_mobilenet_lstm_game_s0.json"), 60),
    stage("gpu_check", ["finetune/gpu_check.py"], None, 1, gpu=True),
    stage("gpu_grid", ["finetune/run_all.py"], grid_done, 1260, gpu=True),
    stage("merge", ["scripts/merge_model_predictions.py"], in_predictions("vjepa2_raw"), 2),
]
BY_NAME = {s["name"]: s for s in STAGES}


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def load_status():
    try:
        with open(STATUS, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"stages": {}}


def save_status(st):
    os.makedirs(LOGS, exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATUS)


def run_one(s, st):
    log = os.path.join(LOGS, s["name"] + ".log")
    rec = st["stages"].setdefault(s["name"], {})
    for attempt in range(1, ATTEMPTS + 1):
        print("[%s] %s: running (%d/%d, about %d min) -> logs/%s.log"
              % (now(), s["name"], attempt, ATTEMPTS, s["mins"], s["name"]), flush=True)
        rec.update(state="running", started=now(), attempt=attempt)
        save_status(st)
        t0 = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n===== %s attempt %d: %s\n" % (now(), attempt, " ".join(s["cmd"])))
            fh.flush()
            code = subprocess.run(s["cmd"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                  env=env).returncode
        mins = (time.time() - t0) / 60
        if s["mark"] and code == 0:
            open(os.path.join(LOGS, s["name"] + ".done"), "w").close()
        ok = s["done"]()
        rec.update(state="done" if ok else "failed", exit=code, minutes=round(mins, 1),
                   finished=now())
        save_status(st)
        print("[%s] %s: %s (%.0f min)" % (now(), s["name"], "done" if ok else "FAILED", mins),
              flush=True)
        if ok:
            return True
    return False


def labels():
    sys.path.insert(0, os.path.join(ROOT, "finetune"))
    from common import load_items
    return {i.cid: i.y_home for i in load_items()}


def from_results():
    rows, covered = {}, set()
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "score" not in d or "model" not in d:
            continue
        key = re.sub(r"_s\d+$", "", d.get("name", os.path.basename(f)[:-5]))
        r = rows.setdefault(key, dict(acc=[], n=0, stuck=0, folds=0))
        r["acc"].append(d["score"]["clip"]["acc"])
        r["n"] = d["score"]["clip"]["n"]
        pl = d.get("plateau")
        if isinstance(pl, list):
            r["stuck"] += sum(1 for x in pl if x)
            r["folds"] += len(pl)
        covered.add("%s_%s" % (d["model"], d["input"]))
    out = []
    for key, r in rows.items():
        out.append((key, 100 * sum(r["acc"]) / len(r["acc"]), r["n"], len(r["acc"]),
                    r["stuck"], r["folds"]))
    return out, covered


def from_predictions(truth, covered):
    if not os.path.exists(PREDICTIONS):
        return []
    per = {}
    with open(PREDICTIONS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clip"] in truth:
                per.setdefault(r["method"], {})[r["clip"]] = float(r["p_home"])
    out = []
    for m, d in per.items():
        if m in covered:
            continue
        hits = sum((d[c] > 0.5) == bool(truth[c]) for c in d)
        out.append((m, 100.0 * hits / len(d), len(d), 1, 0, 0))
    return out


def summary():
    truth = labels()
    n = len(truth)
    bar = max(sum(truth.values()), n - sum(truth.values())) / n
    res, covered = from_results()
    rows = res + from_predictions(truth, covered)
    rows.append(("majority", 100 * bar, n, 1, 0, 0))
    rows.sort(key=lambda r: -r[1])
    print("\naccuracy on the clips each method could score:\n")
    for key, acc, clips, ns, stuck, folds in rows:
        line = "%-52s %5.1f%%   %3d clips" % (NAMES.get(key, key), acc, clips)
        if ns > 1:
            line += "   %d seeds" % ns
        if stuck:
            line += "   %d/%d folds never trained" % (stuck, folds)
        print(line)
    print("\nthe bar is the majority class: always answer the more common team")


def show():
    gpu = cuda()
    print("%d clips   CUDA: %s\n" % (n_clips(), "yes" if gpu else "no"))
    st = load_status()
    for s in STAGES:
        if s["gpu"] and not gpu:
            state = "skipped, needs a GPU"
        elif s["done"]():
            state = "done"
        else:
            state = st["stages"].get(s["name"], {}).get("state", "pending")
        print("  %-16s %-22s %s~%5d min" % (s["name"], state, "GPU " if s["gpu"] else "    ",
                                            s["mins"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--from", dest="start", metavar="STAGE")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(ANALYSIS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    if a.list:
        show()
        return
    if a.summary:
        summary()
        return
    if n_clips() == 0:
        sys.exit("no clips found in clips/clips")

    todo = STAGES
    if a.start:
        names = [s["name"] for s in STAGES]
        if a.start not in names:
            sys.exit("unknown stage: " + a.start)
        todo = STAGES[names.index(a.start):]
    if a.only:
        want = [w.strip() for w in a.only.split(",")]
        for w in want:
            if w not in BY_NAME:
                sys.exit("unknown stage: " + w)
        todo = [BY_NAME[w] for w in want]

    gpu = cuda()
    st = load_status()
    failed = []
    for s in todo:
        if s["gpu"] and (a.cpu_only or not gpu):
            print("[%s] %s: skipped, needs a GPU" % (now(), s["name"]), flush=True)
            st["stages"].setdefault(s["name"], {})["state"] = "skipped, needs a GPU"
            save_status(st)
            continue
        if s["done"]() and not a.force:
            print("[%s] %s: already done" % (now(), s["name"]), flush=True)
            continue
        if not run_one(s, st):
            failed.append(s["name"])
            if s["name"] == "gpu_check":
                print("[%s] CUDA gate failed: not starting the GPU grid" % now(), flush=True)
                todo = [x for x in todo if x["name"] != "gpu_grid"]
    print("\n[%s] %s" % (now(), ("failed: " + ", ".join(failed) + " (see logs/)") if failed
                         else "all stages done"))
    summary()


if __name__ == "__main__":
    main()
