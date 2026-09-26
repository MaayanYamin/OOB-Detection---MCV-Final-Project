import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "finetune", "logs")
OUT = os.path.join(LOGS, "gpu_check.json")
EXPECT = "RTX 4060"

def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {"found": False, "error": "nvidia-smi not on PATH"}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return {"found": True, "error": f"{type(e).__name__}: {e}"}
    if r.returncode != 0:
        return {"found": True, "error": (r.stderr or r.stdout).strip()[:500]}
    rows = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
    return {"found": True, "gpus": rows}

def torch_info():
    try:
        import torch
    except Exception as e:
        return {"imported": False, "error": f"{type(e).__name__}: {e}"}
    d = {"imported": True, "version": torch.__version__,
         "cuda_build": torch.version.cuda, "available": False}
    try:
        d["available"] = bool(torch.cuda.is_available())
    except Exception as e:
        d["error"] = f"is_available() raised {type(e).__name__}: {e}"
        return d
    if not d["available"]:
        return d
    try:
        d["device_name"] = torch.cuda.get_device_name(0)
        d["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        d["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        d["matmul_ok"] = bool((torch.randn(8, 8, device="cuda") @
                               torch.randn(8, 8, device="cuda")).isfinite().all().item())
    except Exception as e:
        d["error"] = f"{type(e).__name__}: {e}"
        d["matmul_ok"] = False
    return d

def main():
    t = torch_info()
    ok = bool(t.get("available") and t.get("matmul_ok"))
    warn = []
    if ok and EXPECT not in t.get("device_name", ""):
        warn.append(f"expected a {EXPECT}, found {t.get('device_name')!r} — not fatal, "
                    "but the memory budgets in CLAUDE.md assume 8 GB")
    if t.get("imported") and not t.get("available"):
        warn.append("torch imported but reports no CUDA device. The usual cause is the "
                    "CPU wheel: check `cuda_build` below — None means a CPU-only build, "
                    "so reinstall from the CUDA index (CLAUDE.md step 1).")

    rec = {"ok": ok, "interpreter": sys.executable, "python": platform.python_version(),
           "platform": platform.platform(), "nvidia_smi": nvidia_smi(), "torch": t,
           "warnings": warn}
    os.makedirs(LOGS, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, OUT)

    print(f"interpreter : {sys.executable}")
    print(f"nvidia-smi  : {rec['nvidia_smi'].get('gpus') or rec['nvidia_smi'].get('error')}")
    print(f"torch       : {t.get('version')}  (CUDA build {t.get('cuda_build')})"
          if t.get("imported") else f"torch       : NOT IMPORTABLE - {t.get('error')}")
    print(f"is_available: {t.get('available')}")
    if t.get("available"):
        print(f"device 0    : {t.get('device_name')}  sm_{t.get('capability','?')}  "
              f"{t.get('vram_gb','?')} GB   matmul {'ok' if t.get('matmul_ok') else 'FAILED'}")
    for w in warn:
        print(f"WARNING     : {w}")
    print(f"\n{'CUDA is usable - training may start' if ok else 'CUDA NOT USABLE - training is blocked'}")
    print(f"written to {OUT}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
