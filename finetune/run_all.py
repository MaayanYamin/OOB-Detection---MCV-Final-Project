import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "finetune", "logs")
RESULTS = os.path.join(ROOT, "finetune", "results")
CACHE = os.path.join(ROOT, "cache", "cache")
STATUS = os.path.join(LOGS, "status.json")
LOCK = os.path.join(LOGS, "runner.lock")
PY = sys.executable
WASB_URL = "https://github.com/nttcom/WASB-SBDT"
MONOTRACK_ID = "1uM2FJLG11AtC0fHsurOqBBUuTehRJugs"
WEIGHTS = os.path.join(ROOT, "monotrack_basketball.pth.tar")
MARKER_JOBS = ("cache_crops", "cache_team", "ribbon_baseline", "verify_run",
               "escape_check", "summary")
GATE_JOBS = ("verify_run", "escape_check")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
TAG = "v2"
GRID = os.environ.get("OOB_GRID", "core").strip().lower()
GRIDS = {
    "core": {"vjepa2": ("raw", "ball", "ball_aug"),
             "videomae": ("ball",),
             "videomae_rib": ("ball",),
             "qwen_lora": ("ball",),
             "arena": ("vjepa2", "videomae_rib")},
    "full": {"vjepa2": ("raw", "ball", "ball_aug"),
             "videomae": ("raw", "ball", "ball_aug", "raw_aug"),
             "videomae_rib": ("ball", "ball_aug"),
             "qwen_lora": ("raw", "ball", "ball_aug"),
             "arena": ("videomae", "videomae_rib", "vjepa2", "qwen_lora")},
}
if GRID not in GRIDS:
    sys.exit(f"OOB_GRID={GRID!r} is not one of {sorted(GRIDS)}")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

def now():
    return dt.datetime.now().isoformat(timespec="seconds")

def n_clips():
    return len(glob.glob(os.path.join(ROOT, "clips", "clips", "*.mp4")))

def n_usable():
    from clip_index import load_index
    return sum(1 for c in load_index(ROOT).values()
               if c.label in ("HOME", "AWAY") and c.home in c.colours and c.away in c.colours)

def count(stage, ext="npz"):
    return len(glob.glob(os.path.join(CACHE, stage, f"*.{ext}")))

def results(*names):
    return all(os.path.exists(os.path.join(RESULTS, n + ".json")) for n in names)

def seeds(prefix):
    return results(*[f"{prefix}_s{s}" for s in (0, 1, 2)])

def marker(name):
    return os.path.join(LOGS, name + ".done")

def wasb_ready():
    return (os.path.isdir(os.path.join(ROOT, "third_party", "WASB-SBDT", "src"))
            and os.path.exists(WEIGHTS) and os.path.getsize(WEIGHTS) > 1e6)

def job(name, cmd, done):
    return {"name": name, "cmd": cmd, "done": done}

def is_done(j):
    try:
        return bool(j["done"]())
    except Exception as e:
        print(f"  done-check for {j['name']} raised {type(e).__name__}: {e}", flush=True)
        return False

def gpu_ready():
    try:
        r = subprocess.run([PY, os.path.join(ROOT, "finetune", "gpu_check.py")],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"gpu_check.py could not run: {type(e).__name__}: {e}"
    return r.returncode == 0, (r.stdout or r.stderr).strip()

def needs_gpu(j):
    return any("train_" in str(c) for c in j["cmd"])

def jobs():
    q = [
        job("adopt_edited", [PY, "scripts/adopt_edited.py", "--apply"],
            lambda: os.path.exists(os.path.join(ROOT, "clips", "edited_rename_map.csv"))
            or not os.path.isdir(os.path.join(ROOT, "edited", "edited"))),
        job("cache_detect", [PY, "scripts/cache_gpu.py", "--stage", "detect", "--cache", "cache/cache"],
            lambda: count("detect") >= n_clips()),
        job("cache_pose", [PY, "scripts/cache_gpu.py", "--stage", "pose", "--cache", "cache/cache"],
            lambda: count("pose") >= n_clips()),
        job("setup_wasb", [PY, "finetune/run_all.py", "--setup-wasb"], wasb_ready),
        job("cache_ball", [PY, "scripts/run_ball_all.py", "--repo", "third_party/WASB-SBDT",
                           "--weights", "monotrack_basketball.pth.tar"],
            lambda: count("ball") >= n_clips()),
        job("cache_crops", [PY, "scripts/cache_crops.py", "--all"],
            lambda: os.path.exists(marker("cache_crops"))),
        job("cache_team", [PY, "scripts/team_by_template.py"],
            lambda: os.path.exists(marker("cache_team"))),
        job("frames_raw", [PY, "finetune/frames.py", "--kind", "raw"],
            lambda: count("ftin_raw", "npy") >= n_usable()),
        job("frames_ball", [PY, "finetune/frames.py", "--kind", "ball"],
            lambda: count("ftin_ball", "npy") >= n_usable()),
        job("ribbon", [PY, "finetune/ribbon.py"],
            lambda: count("ribbon", "npz") >= n_usable()),
        job("ribbon_baseline", [PY, "finetune/ribbon.py", "--baseline"],
            lambda: os.path.exists(marker("ribbon_baseline"))),
        job("verify_run", [PY, "finetune/verify_run.py"],
            lambda: os.path.exists(marker("verify_run"))),
        job("escape_check", [PY, "finetune/escape_check.py"],
            lambda: os.path.exists(marker("escape_check"))),
        job("qwen_zs_raw", [PY, "finetune/train_qwen.py", "--mode", "zeroshot",
                            "--input", "raw", "--tag", TAG],
            lambda: results(f"qwen_zs_raw_{TAG}")),
        job("qwen_zs_ball", [PY, "finetune/train_qwen.py", "--mode", "zeroshot",
                             "--input", "ball", "--tag", TAG],
            lambda: results(f"qwen_zs_ball_{TAG}")),
    ]
    g = GRIDS[GRID]
    for inp in g["vjepa2"]:
        extra = ["--epochs", "240"] if inp.endswith("_aug") else []
        q.append(job(f"vjepa2_{inp}", [PY, "finetune/train_vjepa2.py", "--input", inp,
                                       *extra, "--tag", TAG],
                     lambda inp=inp: seeds(f"vjepa2_{inp}_{TAG}_game")))
    for inp in g["videomae"]:
        q.append(job(f"videomae_{inp}", [PY, "finetune/train_videomae.py", "--input", inp,
                                         "--tag", TAG],
                     lambda inp=inp: seeds(f"videomae_{inp}_{TAG}_game")))
    for inp in g["videomae_rib"]:
        q.append(job(f"videomae_{inp}_rib",
                     [PY, "finetune/train_videomae.py", "--input", inp,
                      "--aux-weight", "0.3", "--tag", TAG],
                     lambda inp=inp: seeds(f"videomae_{inp}_rib_{TAG}_game")))
    for inp in g["qwen_lora"]:
        q.append(job(f"qwen_lora_{inp}",
                     [PY, "finetune/train_qwen.py", "--mode", "lora", "--input", inp,
                      "--tag", TAG],
                     lambda inp=inp: seeds(f"qwen_lora_{inp}_{TAG}_game")))
    ARENA = {"videomae": ("train_videomae.py", [], "videomae_ball"),
             "videomae_rib": ("train_videomae.py", ["--aux-weight", "0.3"],
                              "videomae_ball_rib"),
             "vjepa2": ("train_vjepa2.py", [], "vjepa2_ball"),
             "qwen_lora": ("train_qwen.py", ["--mode", "lora"], "qwen_lora_ball")}
    for name in g["arena"]:
        script, extra, run_name = ARENA[name]
        q.append(job(f"arena_{name}", [PY, f"finetune/{script}", *extra, "--input", "ball",
                                       "--protocol", "arena", "--seeds", "0",
                                       "--tag", TAG],
                     lambda r=run_name: results(f"{r}_{TAG}_arena_s0")))
    q.append(job("summary", [PY, "finetune/common.py"], lambda: False))
    return q

def load_status():
    try:
        with open(STATUS, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"jobs": {}}

def save_status(d):
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, STATUS)

class Status:

    def __init__(self):
        self.lock = threading.Lock()
        self.d = load_status()
        self.d.setdefault("jobs", {})

    def update(self, **kw):
        with self.lock:
            self.d.update(kw)
            save_status(self.d)

    def job(self, name, **kw):
        with self.lock:
            r = self.d["jobs"].setdefault(name, {"state": "pending", "attempts": 0})
            r.update(kw)
            save_status(self.d)
            return dict(r)

def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False

def take_lock():
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                with open(LOCK) as fh:
                    other = int(fh.read().strip() or 0)
            except (OSError, ValueError):
                other = 0
            if pid_alive(other):
                return False
            os.remove(LOCK)
    return False

def launch(j, st, env):
    with open(os.path.join(LOGS, j["name"] + ".log"), "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()}  {' '.join(j['cmd'])}\n")
        log.flush()
        p = subprocess.Popen(j["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env=env, creationflags=NO_WINDOW)
        st.update(child_pid=p.pid)
        return p.wait()

def run(retry_failed=False):
    os.makedirs(LOGS, exist_ok=True)
    if not take_lock():
        print("another runner is alive - exiting")
        return
    st = Status()
    st.update(pid=os.getpid(), started=st.d.get("started", now()), restarted=now(),
              finished=False, heartbeat=now(), current=None, child_pid=None)
    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            st.update(heartbeat=now())

    threading.Thread(target=beat, daemon=True).start()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               KMP_DUPLICATE_LIB_OK="TRUE")

    print(f"grid: {GRID}  ({len(jobs()) - 1} jobs)", flush=True)
    cuda, report = gpu_ready()
    st.update(gpu_ok=cuda, gpu_checked=now())
    print(report, flush=True)
    if not cuda:
        msg = ("CUDA is NOT usable from " + PY + ".\n"
               "Every training job is blocked; caching and input jobs still run.\n"
               "Details: finetune/logs/gpu_check.json\n"
               "If torch shows CUDA build None, it is the CPU wheel - reinstall per\n"
               "CLAUDE.md step 1 and never from the root requirements.txt (rule 4),\n"
               "then: .venv\\Scripts\\python finetune\\watchdog.py --retry-failed\n")
        print("\n" + msg, flush=True)
        try:
            with open(os.path.join(LOGS, "NEEDS_A_HUMAN_gpu.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(now() + "\n\n" + msg + "\n" + report + "\n")
        except OSError:
            pass
    gate_failed = None
    try:
        for j in jobs():
            name = j["name"]
            rec = st.job(name)
            if name != "summary" and is_done(j):
                st.job(name, state="done")
                continue
            if gate_failed and needs_gpu(j):
                st.job(name, state="blocked", note=f"{gate_failed} did not pass")
                print(f"  {name}: BLOCKED, {gate_failed} did not pass", flush=True)
                continue
            if not cuda and needs_gpu(j):
                st.job(name, state="blocked", note="CUDA unavailable - see gpu_check.json")
                print(f"  {name}: BLOCKED, CUDA unavailable", flush=True)
                continue
            if rec["state"] in ("failed", "running") and rec["attempts"] >= 2:
                if not retry_failed:
                    st.job(name, state="failed")
                    continue
                rec = st.job(name, attempts=0)
            attempts = rec["attempts"]
            while attempts < 2:
                attempts += 1
                st.job(name, state="running", attempts=attempts, started=now())
                st.update(current=name)
                code = launch(j, st, env)
                if code == 0 and name in MARKER_JOBS:
                    open(marker(name), "w").close()
                ok = code == 0 if name in MARKER_JOBS else is_done(j)
                st.job(name, exit=code, ended=now(), state="done" if ok else "failed")
                if ok:
                    break
            gate_rec = st.job(name)
            if name in GATE_JOBS and gate_rec["state"] == "done":
                try:
                    os.remove(os.path.join(LOGS, f"NEEDS_A_HUMAN_{name}.txt"))
                    print(f"  {name} passed - cleared its earlier NEEDS_A_HUMAN file",
                          flush=True)
                except OSError:
                    pass
            if name in GATE_JOBS and gate_rec["state"] != "done" \
                    and gate_rec.get("exit") != 1:
                print(f"\n  {name} could not run (exit {gate_rec.get('exit')}). That is a"
                      f" fault in the\n  check, not proof that a model cannot learn, so the"
                      f" queue continues.\n  Read finetune/logs/{name}.log before trusting"
                      f" what the grid produces.\n", flush=True)
            elif name in GATE_JOBS and gate_rec["state"] != "done":
                gate_failed = name
                msg = (f"{name} did not pass, so every training job is blocked.\n"
                       f"verify_run asserts what the run assumes - fold splits, colour\n"
                       f"words, step budgets, and that every pretrained tensor reaches\n"
                       f"the model. escape_check asks whether each model can memorise 16\n"
                       f"clips it is allowed to see; one that cannot will not learn 150\n"
                       f"it cannot.\n"
                       f"Read finetune/logs/{name}.log, fix, then:\n"
                       f"  .venv\\Scripts\\python finetune\\watchdog.py --retry-failed\n")
                print("\n" + msg, flush=True)
                try:
                    with open(os.path.join(LOGS, f"NEEDS_A_HUMAN_{name}.txt"), "w",
                              encoding="utf-8") as fh:
                        fh.write(now() + "\n\n" + msg)
                except OSError:
                    pass
            if name == "summary" and os.path.isdir(RESULTS):
                try:
                    shutil.copy(os.path.join(LOGS, "summary.log"),
                                os.path.join(RESULTS, "SUMMARY.txt"))
                except OSError as e:
                    print(f"  could not copy SUMMARY.txt: {e}", flush=True)
        st.update(finished=True, current=None, child_pid=None, ended=now())
        print("queue finished")
    finally:
        stop.set()
        try:
            os.remove(LOCK)
        except OSError:
            pass

def setup_wasb():
    repo = os.path.join(ROOT, "third_party", "WASB-SBDT")
    if not os.path.isdir(os.path.join(repo, "src")):
        os.makedirs(os.path.dirname(repo), exist_ok=True)
        if shutil.which("git"):
            subprocess.run(["git", "clone", "--depth", "1", WASB_URL + ".git", repo], check=True)
        else:
            for branch in ("main", "master"):
                try:
                    z = os.path.join(ROOT, "third_party", "wasb.zip")
                    urllib.request.urlretrieve(f"{WASB_URL}/archive/refs/heads/{branch}.zip", z)
                    with zipfile.ZipFile(z) as zf:
                        top = zf.namelist()[0].split("/")[0]
                        zf.extractall(os.path.dirname(repo))
                    os.replace(os.path.join(os.path.dirname(repo), top), repo)
                    os.remove(z)
                    break
                except Exception as e:
                    print(f"zip download of branch {branch} failed: {e}")
    if not os.path.exists(WEIGHTS) or os.path.getsize(WEIGHTS) < 1e6:
        import gdown
        gdown.download(id=MONOTRACK_ID, output=WEIGHTS, quiet=False)
    print("WASB-SBDT and MonoTrack weights ready" if wasb_ready() else "WASB setup incomplete")
    sys.exit(0 if wasb_ready() else 1)

def show():
    st = load_status()
    for j in jobs():
        rec = st.get("jobs", {}).get(j["name"], {})
        state = "done" if j["name"] != "summary" and is_done(j) else rec.get("state", "pending")
        print(f"  {j['name']:<22}{state:<9}{rec.get('attempts', 0)} attempt(s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--setup-wasb", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.list:
        show()
    elif a.setup_wasb:
        setup_wasb()
    else:
        run(a.retry_failed)
