# Which team touched the ball last?

Out-of-bounds calls from NBA replay clips. Given a 1–5 second clip and the two jersey
colours on the court, predict which team touched the ball last.

`clips/clips/` holds the 191 hand-cut clips and `manifest.csv` the events they come from
with the label for each one.

## Run everything

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On a machine with an NVIDIA GPU, install CUDA PyTorch instead of that second line:

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r finetune/requirements-gpu.txt
```

Then:

```bash
python run_all.py
```

That runs every method in the paper on all 191 clips and ends with one line per method:

```
VideoMAE (frozen) + head, whole frame       58.3%   191 clips   3 seeds
jersey colour around the ball               57.5%   179 clips
majority class (the bar)                    56.0%   191 clips
MobileNetV2 + LSTM                          55.0%   191 clips
CLIP zero-shot, ball window                 53.9%   178 clips
VideoMAE fine-tuned + possession target     53.6%   191 clips   3 seeds
V-JEPA 2 (frozen) + head, whole frame       52.4%   191 clips   3 seeds
MobileNet features + linear classifier      51.8%   191 clips
team of the player nearest the ball         51.1%   176 clips
Qwen2.5-VL zero-shot, whole frame           48.7%   191 clips
Qwen2.5-VL QLoRA, ball window               48.0%   191 clips   3 seeds
ball/player kinematics + linear classifier  47.9%   169 clips
```

Other flags:

```bash
python run_all.py --list        # the stages, their state, and how long each takes
python run_all.py --summary     # the accuracy lines again, without running anything
python run_all.py --only poc    # one stage
python run_all.py --cpu-only    # skip the stages that need a GPU
python run_all.py --force       # re-run a stage whose output is already there
```

Stages skip themselves when their output exists, so the run can be stopped and restarted; a
stage that fails twice is left failed and the rest continue. Logs are in `logs/`, one result
file per run in `finetune/results/`.

Eight of the twelve rows above run on a CPU, in about two days on a laptop — most of it the
person detector and the ball tracker, which run once and are cached in `cache/`. Three need
CUDA: the fine-tuned VideoMAE and the two Qwen2.5-VL runs, roughly 12 hours on an RTX 4060
Laptop, 8 GB. Without a GPU they are reported as skipped and everything else still runs.
`finetune/gpu_check.py` runs first on a GPU machine — `nvidia-smi`, the torch CUDA build, and
one real CUDA matmul, because a mismatched build can import fine and still fail its first
kernel — and the GPU stages are skipped if it fails, rather than training on the CPU for a
week by accident.

## What gets run

Caching from video first: player boxes and poses (YOLO11l), the ball track (WASB-SBDT with
MonoTrack basketball weights, downloaded on demand), jersey crops, a team for each player box
from the two filename colours, and the possession target — per frame, whether the ball sits
at a player's wrist and whose team that player is on, masked wherever the tracker cannot
support a call. Then the methods in the table above.

Every method answers the same question — *the two jerseys are colour A and colour B, which
touched last?* — with the colours in random order and both orders averaged, and every method
is scored the same way: 5 folds grouped by game so no game is in both training and test, the
final epoch, accuracy against the majority-class bar. A setting run with three seeds is
reported as the mean over its seeds. The `leak_*` columns in `manifest.csv` are never given to
a model.

## Filenames

```
20250120_ATLatNYK_P1_0042_3__a1_ATL_red_NYK_blue.mp4
 date     away at home  period clock  angle  away+colour  home+colour
```

`a1`, `a2`, `a3` are camera angles of the same event; angles of one event always stay in the
same fold. A new colour word needs its RGB in `finetune/common.py` and its hue in
`scripts/clip_index.py`.

## Layout

| path | what |
|---|---|
| `run_all.py` | the only entry point |
| `finetune/` | the shared protocol (clips, question, folds, scoring), the model inputs, the possession target, and the trained models |
| `finetune/train_frozen.py` | the frozen-backbone runs |
| `scripts/` | the cache pipeline and the methods that train nothing or train a linear classifier |
| `poc/` | the MobileNetV2 + LSTM model and the script that runs it on the corpus |

## Third-party

The ball detector is [WASB-SBDT](https://github.com/nttcom/WASB-SBDT).
Backbones come from Hugging Face: `facebook/vjepa2-vitl-fpc64-256`,
`MCG-NJU/videomae-base-finetuned-ssv2`, `openai/clip-vit-base-patch32`,
`Qwen/Qwen2.5-VL-3B-Instruct`. Clips are NBA broadcast footage, used for a course project.
