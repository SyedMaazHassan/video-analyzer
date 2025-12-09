import os
from typing import Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torchvision
import torchvision.models as models

from models._base.base_extractor import BaseExtractor
from models.transforms import (CenterCrop, Normalize, Resize,
                               ToFloatTensorInZeroOne)
from utils.io import reencode_video_with_diff_fps
from utils.utils import form_slices, show_predictions_on_dataset


class ExtractR21D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        r21d_model_cfgs = {
            'r2plus1d_18_16_kinetics': {
                'repo': None,
                'stack_size': 16, 'step_size': 16, 'num_classes': 400, 'dataset': 'kinetics'
            },
            'r2plus1d_34_32_ig65m_ft_kinetics': {
                'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_32_kinetics',
                'stack_size': 32, 'step_size': 32, 'num_classes': 400, 'dataset': 'kinetics'
            },
            'r2plus1d_34_8_ig65m_ft_kinetics': {
                'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_8_kinetics',
                'stack_size': 8, 'step_size': 8, 'num_classes': 400, 'dataset': 'kinetics'
            },
        }
        self.model_name = args.model_name
        self.model_def = r21d_model_cfgs[self.model_name]
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = self.model_def['step_size']
        if self.stack_size is None:
            self.stack_size = self.model_def['stack_size']
        self.show_pred = args.show_pred
        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.
        Processes video in chunks to avoid memory overflow.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # Use cv2 to read frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_name = os.path.basename(video_path)
        print(f"  Processing video: {video_name} ({total_frames} frames)...", flush=True)

        if total_frames == 0:
            cap.release()
            raise ValueError(f"No frames in video: {video_path}")

        # Process in chunks to avoid memory overflow
        # Each chunk processes enough frames for one stack + overlap
        chunk_size = self.stack_size * 10  # Process 10 stacks worth at a time
        target_size = (171, 128)  # (width, height) for cv2.resize

        vid_feats = []
        frame_idx = 0

        # Calculate total stacks for progress bar
        total_stacks = max(1, (total_frames - self.stack_size) // self.step_size + 1)
        pbar = tqdm(total=total_stacks, desc='  Extracting features', unit='stack', leave=False)

        while frame_idx < total_frames:
            # Read a chunk of frames
            chunk_frames = []
            frames_to_read = min(chunk_size, total_frames - frame_idx)

            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and resize immediately
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                chunk_frames.append(frame)

            if len(chunk_frames) < self.stack_size:
                # Not enough frames for even one stack
                if len(vid_feats) == 0 and len(chunk_frames) > 0:
                    # Pad to stack_size if this is the only chunk
                    while len(chunk_frames) < self.stack_size:
                        chunk_frames.append(chunk_frames[-1])
                else:
                    break

            # Convert chunk to tensor
            frames_array = np.stack(chunk_frames)  # (T, H, W, C)
            frames_tensor = torch.from_numpy(frames_array)

            # Apply transforms
            rgb = self.transforms(frames_tensor)  # (C, T, H, W)
            rgb = rgb.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T, C, H, W)

            # Get slices for this chunk
            slices = form_slices(rgb.size(1), self.stack_size, self.step_size)

            for start_idx, end_idx in slices:
                stack = rgb[:, start_idx:end_idx, :, :, :]
                stack = stack.permute(0, 2, 1, 3, 4)  # (1, C, stack_size, H, W)

                # inference
                output = self.name2module['model'](stack.to(self.device))
                vid_feats.extend(output.tolist())
                pbar.update(1)

            # Move frame index, keeping overlap for next chunk
            frame_idx += len(chunk_frames) - self.stack_size + self.step_size

            # Clear memory
            del frames_array, frames_tensor, rgb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
        cap.release()

        if len(vid_feats) == 0:
            raise ValueError(f"No features extracted from video: {video_path}")

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize((128, 171)),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            CenterCrop((112, 112)),
        ])

        if self.model_name == 'r2plus1d_18_16_kinetics':
            weights_key = 'DEFAULT'
            model = models.get_model('r2plus1d_18', weights=weights_key)
        else:
            model = torch.hub.load(
                self.model_def['repo'],
                model=self.model_def['model_name_in_repo'],
                num_classes=self.model_def['num_classes'],
                pretrained=True,
            )

        model = model.to(self.device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()

        return {
            'model': model,
            'class_head': class_head,
        }

    def maybe_show_pred(self, visual_feats: torch.Tensor, start_idx: int, end_idx: int):
        if self.show_pred:
            logits = self.name2module['class_head'](visual_feats)
            print(f'At frames ({start_idx}, {end_idx})')
            show_predictions_on_dataset(logits, self.model_def['dataset'])
