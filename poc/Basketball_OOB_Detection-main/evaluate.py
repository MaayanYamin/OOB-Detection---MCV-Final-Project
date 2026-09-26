"""
Evaluation functions for OOB Detection
Handles cross-validation and metrics calculation
"""

import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import OOBDatasetWithColor
from train import train_fold
from utils import set_seed
from config import N_FOLDS, BATCH_SIZE, EPOCHS, RANDOM_SEED


def cross_validate(video_paths, labels, jersey_colors):
    """Perform 4-fold cross-validation"""

    if len(video_paths) != 20:
        print(f"Expected 20 videos, found {len(video_paths)}")

    n_folds = N_FOLDS
    fold_size = len(video_paths) // n_folds

    indices = list(range(len(video_paths)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    fold_results = []
    all_predictions = []

    print(f"\nCreating {n_folds} folds with {fold_size} videos each")

    for fold in range(n_folds):
        set_seed(RANDOM_SEED + fold)
        
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_indices = indices[val_start:val_end]
        train_indices = [i for i in indices if i not in val_indices]

        train_paths = [video_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        train_colors = [jersey_colors[i] for i in train_indices]

        val_paths = [video_paths[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        val_colors = [jersey_colors[i] for i in val_indices]

        print(f"\nFold {fold+1}: Train on {len(train_paths)} videos, Validate on {len(val_paths)} videos")
        print(f"Val videos: {[Path(p).name[:15] for p in val_paths]}")

        train_dataset = OOBDatasetWithColor(train_paths, train_labels, train_colors,
                                           is_train=True, augment=True)
        val_dataset = OOBDatasetWithColor(val_paths, val_labels, val_colors,
                                         is_train=False, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        val_acc, predictions = train_fold(train_loader, val_loader, fold+1, epochs=EPOCHS)

        fold_results.append(val_acc)
        all_predictions.extend(predictions)

    print(f"\n{'-'*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'-'*60}")

    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.1f}%")

    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    overall_correct = sum(p['correct'] for p in all_predictions)
    overall_acc = 100 * overall_correct / len(all_predictions)
    print(f"Overall Accuracy: {overall_acc:.1f}% ({overall_correct}/{len(all_predictions)})")

    home_preds = [p for p in all_predictions if p['true'] == 1]
    away_preds = [p for p in all_predictions if p['true'] == 0]

    home_acc = 100 * sum(p['correct'] for p in home_preds) / len(home_preds) if home_preds else 0
    away_acc = 100 * sum(p['correct'] for p in away_preds) / len(away_preds) if away_preds else 0

    print(f"\nPerformance by Label:")
    print(f"  Home team ball: {home_acc:.1f}% ({sum(p['correct'] for p in home_preds)}/{len(home_preds)})")
    print(f"  Away team ball: {away_acc:.1f}% ({sum(p['correct'] for p in away_preds)}/{len(away_preds)})")

    return fold_results, all_predictions
