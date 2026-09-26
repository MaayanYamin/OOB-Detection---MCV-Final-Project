"""
Dataset class for OOB Detection
Handles video loading, frame extraction, and data augmentation
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms

from config import COLOR_ENCODING, NUM_FRAMES, AUGMENTATION_MULTIPLIER


class OOBDatasetWithColor(Dataset):
    """Dataset with jersey color metadata"""

    def __init__(self, video_paths, labels, jersey_colors, is_train=True, num_frames=NUM_FRAMES, augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.jersey_colors = jersey_colors
        self.num_frames = num_frames
        self.is_train = is_train
        self.augment = augment and is_train

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5 if self.augment else 0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if self.augment else transforms.Lambda(lambda x: x),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths) * (AUGMENTATION_MULTIPLIER if self.augment else 1)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.video_paths)
        aug_version = idx // len(self.video_paths)

        video_path = self.video_paths[actual_idx]
        label = self.labels[actual_idx]
        jersey_color = self.jersey_colors[actual_idx]

        frames = self.extract_frames(video_path, aug_version, seed=idx)
        color_vec = torch.tensor(COLOR_ENCODING[jersey_color], dtype=torch.float32)

        return {
            'frames': frames,
            'label': torch.tensor(label, dtype=torch.long),
            'color': color_vec,
            'video_name': Path(video_path).name
        }

    def extract_frames(self, video_path, aug_version=0, seed=0):
        """Extract frames with temporal augmentation"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.num_frames:
            indices = np.array([i % total_frames for i in range(self.num_frames)])
        else:
            if self.augment and aug_version > 0:
                if aug_version == 1:
                    # Focus on first 2/3 of video
                    max_frame = int(total_frames * 0.66)
                    indices = np.linspace(0, max_frame, self.num_frames, dtype=int)
                else:  # aug_version == 2
                    # Focus on last 2/3 of video
                    min_frame = int(total_frames * 0.33)
                    indices = np.linspace(min_frame, total_frames-1, self.num_frames, dtype=int)
            else:
                # Standard: evenly spaced across entire video
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)

        cap.release()
        frames = torch.stack(frames) if frames else torch.zeros(self.num_frames, 3, 224, 224)
        return frames
