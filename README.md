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

That runs every method on all 191 clips and ends with one line per method:

```
VideoMAE frozen + head, whole frame           59.2%   191 clips   3 seeds
V-JEPA 2 frozen + head, whole frame           58.6%   191 clips   3 seeds
jersey colour around the ball                 57.5%   179 clips
majority class (the bar)                      56.0%   191 clips
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
stage that fails twice is left failed and the rest continue. Logs are in `logs/`, results in
`finetune/results/`.

The CPU part takes about two days on a laptop — most of it the person detector and the ball
tracker, which run once and are cached in `cache/`. The GPU part is about 21 hours on an
RTX 4060 Laptop, 8 GB.

## What gets run

Caching from video first: player boxes and poses (YOLO11l), the ball track (WASB-SBDT with
MonoTrack basketball weights, downloaded on demand), jersey crops, a team for each player
box from the two filename colours, and the possession ribbon — per-frame ball-to-wrist
targets used as extra supervision. Then:

- jersey colour around the ball, and the team of the player nearest the ball
- CLIP zero-shot, on the whole frame and on a window following the ball
- MobileNet features + a linear head, and ball/player kinematics + a linear head
- VideoMAE and V-JEPA 2 frozen, with a small head that also sees the two colours
- on a GPU: VideoMAE fine-tuned, VideoMAE + the ribbon head, V-JEPA 2 on three inputs,
  Qwen2.5-VL-3B zero-shot and with QLoRA, and two unseen-arena runs
- MobileNetV2 + LSTM with the home jersey colour as a second input (`poc/`)

Every method answers the same question — *the two jerseys are colour A and colour B, which
touched last?* — with the colours in random order and both orders averaged, and every
method is scored the same way: 5 folds grouped by game so no game is in both training and
test, the final epoch, accuracy against the majority-class bar. The `leak_*` columns in
`manifest.csv` are never given to a model.

A fold whose final loss sits at ln(2) = 0.693 did not train: averaging both colour orders
makes p = 0.5 an exact fixed point, so that fold's accuracy measures the optimiser, not the
task. Those folds are counted in each result file and in the accuracy lines.

## The GPU stages

`run_all.py` checks for CUDA and reports the GPU stages as skipped without it. With a GPU it
runs `finetune/gpu_check.py` first — `nvidia-smi`, the torch CUDA build, and one real CUDA
matmul, because a mismatched build can import fine and still fail its first kernel — and
blocks training if that fails. It then runs `finetune/run_all.py`, which checks the caches,
runs two gates (`verify_run.py`: the colour words, the fold splits, the colour-order
antisymmetry, the quarantined columns, that every pretrained tensor reaches the model;
`escape_check.py`: can each model memorise 16 clips it is allowed to see), and only then
trains.

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
| `finetune/` | the shared protocol, the model inputs, the ribbon, the trained models, the gates |
| `finetune/train_frozen.py` | the frozen-backbone runs |
| `scripts/` | the cache pipeline and the methods that train nothing or train a linear head |
| `poc/` | the MobileNetV2 + LSTM model and the script that runs it on the corpus |

## Third-party

The ball detector is [WASB-SBDT](https://github.com/nttcom/WASB-SBDT).
Backbones come from Hugging Face: `facebook/vjepa2-vitl-fpc64-256`,
`MCG-NJU/videomae-base-finetuned-ssv2`, `openai/clip-vit-base-patch32`,
`Qwen/Qwen2.5-VL-3B-Instruct`. Clips are NBA broadcast footage, used for a course project.
