import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import LOGS, PY, ROOT, is_done, jobs, load_status, now, pid_alive

STALE_BEAT = 10 * 60
SILENT_JOB = 60 * 60
STOPFILE = os.path.join(LOGS, "STOP")
DETACH = (0x00000008 | 0x00000200) if os.name == "nt" else 0

def age(iso):
    try:
        return (dt.datetime.now() - dt.datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")

def log(msg):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "watchdog.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{now()}  {msg}\n")

def kill(pid):
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)

def start_runner(retry=False):
    os.makedirs(LOGS, exist_ok=True)
    out = open(os.path.join(LOGS, "runner.out"), "a", encoding="utf-8")
    cmd = [PY, os.path.join(ROOT, "finetune", "run_all.py")] + (["--retry-failed"] if retry else [])
    subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                     creationflags=DETACH, close_fds=True)

def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no log)"

def gpu():
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found"
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                        "temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()

def stopped():
    return os.path.exists(STOPFILE)

def check(fix=True):
    if stopped():
        msg = (f"[{now()}] STOP requested ({STOPFILE} exists) - not starting anything.\n"
               f"Every finished fold is checkpointed. RUN\\START.bat clears this and resumes.")
        log(msg.replace("\n", "\n    "))
        print(msg)
        return True
    st = load_status()
    q = jobs()
    done = [j["name"] for j in q if j["name"] != "summary" and is_done(j)]
    left = [j["name"] for j in q if j["name"] != "summary" and j["name"] not in done]
    failed = [n for n, r in st.get("jobs", {}).items() if r.get("state") == "failed"]
    lines = [f"[{now()}] {len(done)}/{len(q) - 1} jobs done"
             + (f", FAILED: {', '.join(failed)}" if failed else "")]
    action = "none"

    if st.get("finished") and not left:
        lines.append("ALL DONE - the queue has finished. Summary: finetune/results/SUMMARY.txt")
    elif st.get("finished") and not pid_alive(st.get("pid")):
        lines.append(f"status.json says finished, but {len(left)} jobs are not done "
                     f"(first: {left[0]}) - that flag is from the previous queue")
        if fix:
            start_runner()
            action = "started the runner"
    elif not pid_alive(st.get("pid")):
        lines.append("runner is NOT running")
        if fix:
            start_runner()
            action = "started the runner"
    elif age(st.get("heartbeat")) > STALE_BEAT:
        lines.append(f"runner alive but its heartbeat is {age(st.get('heartbeat')) / 60:.0f} min old")
        if fix:
            kill(st.get("pid"))
            time.sleep(5)
            start_runner()
            action = "restarted a frozen runner"
    else:
        cur = st.get("current")
        rec = st.get("jobs", {}).get(cur, {})
        lp = os.path.join(LOGS, f"{cur}.log")
        quiet = time.time() - os.path.getmtime(lp) if os.path.exists(lp) else 0
        lines.append(f"running: {cur} (attempt {rec.get('attempts')}, "
                     f"{age(rec.get('started')) / 60:.0f} min so far, log quiet {quiet / 60:.0f} min)")
        if quiet > SILENT_JOB and st.get("child_pid"):
            lines.append("  its log has been silent for over an hour - treating it as hung")
            if fix:
                kill(st.get("child_pid"))
                action = f"killed hung job {cur} (the runner retries it once)"
        lines.append("  last lines of its log:\n" + tail(lp, 6).rstrip())

    for n in failed:
        lines.append(f"--- {n} FAILED, last lines of finetune/logs/{n}.log:\n"
                     + tail(os.path.join(LOGS, f"{n}.log")).rstrip())
    free = shutil.disk_usage(ROOT).free / 2**30
    lines.append(f"GPU: {gpu()}   free disk: {free:.0f} GB" + ("  <-- LOW" if free < 5 else ""))
    lines.append(f"action taken: {action}")
    report = "\n".join(lines)
    log(report.replace("\n", "\n    "))
    print(report)
    return bool(st.get("finished")) and not left

def retry_failed():
    st = load_status()
    if pid_alive(st.get("pid")):
        kill(st.get("pid"))
        time.sleep(5)
    start_runner(retry=True)
    log("restarted the runner with --retry-failed")
    print("runner restarted with --retry-failed")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--loop", type=int, metavar="SECONDS")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    if a.retry_failed:
        retry_failed()
    elif a.loop:
        while not check(fix=True):
            time.sleep(a.loop)
    else:
        check(fix=not a.report)
