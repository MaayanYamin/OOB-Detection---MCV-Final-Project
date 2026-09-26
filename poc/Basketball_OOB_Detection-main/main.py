"""
Main entry point for OOB Detection
Runs cross-validation and prints results
"""

import warnings
from pathlib import Path

from utils import set_seed, prepare_data
from evaluate import cross_validate
from config import RANDOM_SEED, VIDEO_DIR

warnings.filterwarnings('ignore')


def main():
    """Main function to run cross-validation"""
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    print("OOB Detection - 4-Fold Cross Validation")
    print(f"Seed: {RANDOM_SEED}")
    
    # Check if videos are available
    video_count = len(list(Path(VIDEO_DIR).glob('clip*.mp4')))
    if video_count < 20:
        print("Please upload all 20 video clips first!")
        print("Use the Files sidebar to upload your clips to /content/")
        return
    
    # Prepare data
    video_paths, labels, jersey_colors = prepare_data()
    
    # Run cross-validation
    fold_results, all_predictions = cross_validate(video_paths, labels, jersey_colors)


if __name__ == "__main__":
    main()
