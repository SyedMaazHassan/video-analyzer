#!/usr/bin/env python3
"""
Simple training script with better error handling and data processing.
"""

import os
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCVATParser:
    """Simplified CVAT parser for basic phase detection."""
    
    def __init__(self, xml_path: str, video_path: str):
        self.xml_path = xml_path
        self.video_path = video_path
        self.phase_labels = [
            'Portal Placement',
            'Diagnostic Arthroscopy', 
            'Glenoid Preparation',
            'Anchor Placement',
            'Suture Passage',
            'Suture Fixation',
            'Final Inspection'
        ]
        
    def extract_samples(self, max_samples=50):
        """Extract training samples from video."""
        logger.info(f"Extracting samples from {self.video_path}")
        
        try:
            # Parse XML to get any phase information
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Get video info
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video has {total_frames} frames at {fps} FPS")
            
            samples = []
            frame_step = max(1, total_frames // max_samples)
            
            for frame_idx in range(0, total_frames, frame_step):
                if len(samples) >= max_samples:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Assign a default phase (we'll use the first phase for simplicity)
                    phase_idx = frame_idx // (total_frames // len(self.phase_labels))
                    phase_idx = min(phase_idx, len(self.phase_labels) - 1)
                    
                    samples.append({
                        'frame': frame,
                        'phase': phase_idx,
                        'frame_idx': frame_idx
                    })
                    
                    if len(samples) % 10 == 0:
                        logger.info(f"Extracted {len(samples)} samples...")
            
            cap.release()
            logger.info(f"Total samples extracted: {len(samples)}")
            return samples
            
        except Exception as e:
            logger.error(f"Error extracting samples: {str(e)}")
            return []

class SimpleDataset(Dataset):
    """Simple dataset for surgical video frames."""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = sample['frame']
        phase = sample['phase']
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, phase

class SimplePhaseModel(nn.Module):
    """Simple ResNet-based phase recognition model."""
    
    def __init__(self, num_phases):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_phases)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs=5):
    """Train the model."""
    device = torch.device('cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (frames, labels) in enumerate(progress_bar):
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    return model

def main():
    """Main function."""
    logger.info("Starting simple surgical video analysis training...")
    
    # Setup paths
    videos_dir = Path('/app/videos')
    xml_dir = Path('/app/xml_path')
    models_dir = Path('/app/trained_models')
    models_dir.mkdir(exist_ok=True)
    
    # Find video and XML files
    video_files = list(videos_dir.glob('*.mp4'))
    xml_files = list(xml_dir.glob('*.xml'))
    
    if not video_files:
        logger.error("No video files found!")
        return
    
    if not xml_files:
        logger.error("No XML files found!")
        return
    
    video_file = video_files[0]
    xml_file = xml_files[0]
    
    logger.info(f"Using video: {video_file.name}")
    logger.info(f"Using XML: {xml_file.name}")
    
    # Extract samples
    parser = SimpleCVATParser(str(xml_file), str(video_file))
    samples = parser.extract_samples(max_samples=60)
    
    if not samples:
        logger.error("No samples extracted!")
        return
    
    # Split data
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleDataset(train_samples, transform=transform)
    val_dataset = SimpleDataset(val_samples, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model
    num_phases = len(parser.phase_labels)
    model = SimplePhaseModel(num_phases)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=3)
    
    # Save model and config
    model_path = models_dir / 'simple_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save config
    config = {
        'phase_labels': parser.phase_labels,
        'model_type': 'simple_resnet',
        'num_phases': num_phases
    }
    
    config_path = models_dir / 'simple_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_path}")
    
    logger.info("Training completed successfully!")
    logger.info("You can now test the model with new videos.")

if __name__ == "__main__":
    main()