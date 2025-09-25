# surgical_ai_system/core/datasets.py

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List

from surgical_ai_system.config.system_config import app_config
from surgical_ai_system.core.data_processing import load_case_data, extract_frames

class PhaseDetectionDataset(Dataset):
    """
    PyTorch Dataset for the Phase Detection model.
    
    This dataset prepares data by:
    1. Loading case data (video path, phases).
    2. Generating a frame-by-frame label array for the entire video.
    3. Sampling frames at a specified FPS for training.
    """
    def __init__(self, video_data_dir: str, transforms=None):
        self.video_data_dir = Path(video_data_dir)
        self.transforms = transforms
        self.cases = [d for d in self.video_data_dir.iterdir() if d.is_dir()]
        
        self.phase_to_id = {label: i for i, label in enumerate(app_config.labels['phases'])}
        self.id_to_phase = {i: label for label, i in self.phase_to_id.items()}
        
        self.samples = self._create_samples()

    def _create_samples(self):
        """
        Prepares all frame samples and their corresponding labels.
        """
        samples = []
        for case_dir in self.cases:
            case_data = load_case_data(case_dir)
            
            # Create a mapping of frame number to phase ID
            frame_labels = np.full(case_data.total_frames, self.phase_to_id['Background'])
            for phase in case_data.phases:
                if phase.phase_name in self.phase_to_id:
                    start = phase.start_frame
                    end = phase.end_frame
                    frame_labels[start:end+1] = self.phase_to_id[phase.phase_name]

            # Sample frames based on the desired FPS
            sampling_rate = int(case_data.fps / app_config.data_processing['phase_model_fps'])
            for frame_idx in range(0, case_data.total_frames, sampling_rate):
                samples.append({
                    "video_path": case_data.video_path,
                    "frame_idx": frame_idx,
                    "label": frame_labels[frame_idx]
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        frame_idx = sample['frame_idx']
        label = sample['label']

        # Extract the single frame
        frame_generator = extract_frames(video_path, [frame_idx])
        frame = next(frame_generator)

        if self.transforms:
            frame = self.transforms(frame)
            
        return frame, torch.tensor(label, dtype=torch.long)

# Placeholder for the Event Detection Dataset
class EventDetectionDataset(Dataset):
    """
    PyTorch Dataset for the Event Detection model.
    This will be implemented later. It will serve clips of frames.
    """
    def __init__(self, video_data_dir: str, transforms=None):
        # TODO: Implement logic to create clips around events
        pass

    def __len__(self):
        # TODO
        return 0

    def __getitem__(self, idx):
        # TODO
        pass
