"""
Utility functions for OOB Detection
Includes data loading, seeding, and helper functions
"""

import random
import numpy as np
import torch
from pathlib import Path

from config import JERSEY_COLORS, VIDEO_DIR, VIDEO_PATTERN, RANDOM_SEED


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data():
    """Load all videos and prepare for cross-validation"""
    video_dir = Path(VIDEO_DIR)

    video_paths = []
    labels = []
    jersey_colors = []

    for video_file in sorted(video_dir.glob(VIDEO_PATTERN)):
        filename = video_file.stem
        clip_num = filename.split(' - ')[0].lower()

        if "away" in filename.lower():
            label = 0
        elif "home" in filename.lower():
            label = 1
        else:
            continue

        jersey_color = JERSEY_COLORS.get(clip_num, 'blue')

        video_paths.append(str(video_file))
        labels.append(label)
        jersey_colors.append(jersey_color)

    print(f"Found {len(video_paths)} videos")
    print(f"  Home (1): {sum(labels)}, Away (0): {len(labels) - sum(labels)}")
    print(f"  White jerseys: {jersey_colors.count('white')}")
    print(f"  Blue jerseys: {jersey_colors.count('blue')}")
    print(f"  Black jerseys: {jersey_colors.count('black')}")

    return video_paths, labels, jersey_colors
