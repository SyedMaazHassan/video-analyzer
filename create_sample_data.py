#!/usr/bin/env python3
"""
Create Sample Training Data
==========================
Generates minimal sample video data for training when real videos are not available.
"""

import cv2
import numpy as np
from pathlib import Path
import json

def create_sample_video(video_path: str, duration_seconds: int = 30, fps: int = 30):
    """Create a sample video with random frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create random colored frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to make it look more surgical
        # Add circular "arthroscope view"
        center = (320, 240)
        radius = 200
        cv2.circle(frame, center, radius, (0, 0, 0), 5)
        
        # Add some random "instrument" shapes
        if frame_num % 60 < 30:  # Simulate instrument presence
            cv2.rectangle(frame, (100, 100), (200, 150), (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {video_path}")

def main():
    """Create sample training data."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    # Create 5 sample videos
    for i in range(1, 6):
        video_path = videos_dir / f"video_{i:05d}.mp4"
        create_sample_video(str(video_path), duration_seconds=10)  # Short videos for quick training
    
    print(f"Created 5 sample videos in {videos_dir}")
    print("Ready for training!")

if __name__ == "__main__":
    main()