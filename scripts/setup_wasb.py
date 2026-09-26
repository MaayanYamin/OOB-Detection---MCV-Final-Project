import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WASB_URL = "https://github.com/nttcom/WASB-SBDT"
MONOTRACK_ID = "1uM2FJLG11AtC0fHsurOqBBUuTehRJugs"
REPO = os.path.join(ROOT, "third_party", "WASB-SBDT")
WEIGHTS = os.path.join(ROOT, "monotrack_basketball.pth.tar")


def ready():
    return (os.path.isdir(os.path.join(REPO, "src"))
            and os.path.exists(WEIGHTS) and os.path.getsize(WEIGHTS) > 1e6)


def main():
    if not os.path.isdir(os.path.join(REPO, "src")):
        os.makedirs(os.path.dirname(REPO), exist_ok=True)
        if shutil.which("git"):
            subprocess.run(["git", "clone", "--depth", "1", WASB_URL + ".git", REPO],
                           check=True)
        else:
            for branch in ("main", "master"):
                try:
                    z = os.path.join(ROOT, "third_party", "wasb.zip")
                    urllib.request.urlretrieve(
                        "%s/archive/refs/heads/%s.zip" % (WASB_URL, branch), z)
                    with zipfile.ZipFile(z) as zf:
                        top = zf.namelist()[0].split("/")[0]
                        zf.extractall(os.path.dirname(REPO))
                    os.replace(os.path.join(os.path.dirname(REPO), top), REPO)
                    os.remove(z)
                    break
                except Exception as e:
                    print("zip download of branch %s failed: %s" % (branch, e))
    if not os.path.exists(WEIGHTS) or os.path.getsize(WEIGHTS) < 1e6:
        import gdown
        gdown.download(id=MONOTRACK_ID, output=WEIGHTS, quiet=False)
    print("ball detector ready" if ready() else "ball detector setup incomplete")
    sys.exit(0 if ready() else 1)


if __name__ == "__main__":
    main()
