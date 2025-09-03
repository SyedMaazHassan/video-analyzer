#!/usr/bin/env python3
"""
Fixed setup and training script using ResNet instead of EfficientNet
to avoid download issues.
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
import torch
import torch.nn as nn
from torchvision.models import resnet50
from script import *

def setup_configuration():
    """Set up the configuration matching your CVAT labels."""
    
    config = {
        'phase_labels': [
            'Portal Placement',
            'Diagnostic Arthroscopy', 
            'Glenoid Preparation',
            'Anchor Placement',
            'Suture Passage',
            'Suture Fixation',
            'Final Inspection'
        ],
        'instrument_labels': [
            'Spinal Needle',
            'Reusable Cannula', 
            'Shaver',
            'RF Wand',
            'Burr',
            'Anchor Punch',
            'Bird Beak',
            'Suture Passer',
            'Grasper',
            'Suture Cutter'
        ],
        'event_labels': [
            'Bleeding',
            'Suture Attempt', 
            'Anchor Reposition',
            'Anchor Pullout',
            'Custom Event'
        ]
    }
    
    return config

# Override the PhaseRecognitionModel to use ResNet50
class SimplePhaseRecognitionModel(nn.Module):
    """Simplified CNN model for surgical phase recognition using ResNet50."""
    
    def __init__(self, num_phases: int):
        super().__init__()
        
        # Use ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final layer
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_phases)
        )
        
    def forward(self, x):
        # Handle sequence input by taking last frame
        if len(x.shape) == 5:  # batch, seq, channels, h, w
            x = x[:, -1]  # Take last frame
        
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def setup_paths():
    """Set up project paths."""
    
    paths = {
        'project_root': Path('/app'),
        'videos_dir': Path('/app/videos'),
        'annotations_dir': Path('/app/xml_path'),
        'models_dir': Path('/app/trained_models'),
        'results_dir': Path('/app/results')
    }
    
    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name != 'project_root':
            path.mkdir(parents=True, exist_ok=True)
    
    return paths

def get_annotation_configs(paths):
    """Automatically detect video and annotation pairs."""
    
    videos_dir = paths['videos_dir']
    annotations_dir = paths['annotations_dir']
    
    configs = []
    
    # Find all video files
    video_files = list(videos_dir.glob('*.mp4'))
    
    for video_file in video_files:
        # Try to find corresponding XML file
        video_stem = video_file.stem
        
        # Try different naming patterns
        possible_xml_names = [
            f'{video_stem}.xml',
            f'annotation_{video_stem.split("_")[-1]}.xml' if '_' in video_stem else f'annotation_{video_stem}.xml',
            f'{video_stem}_annotations.xml'
        ]
        
        xml_file = None
        for xml_name in possible_xml_names:
            xml_path = annotations_dir / xml_name
            if xml_path.exists():
                xml_file = xml_path
                break
        
        if xml_file:
            configs.append({
                'xml_path': str(xml_file),
                'video_path': str(video_file)
            })
            logger.info(f"Found pair: {video_file.name} -> {xml_file.name}")
        else:
            logger.warning(f"No annotation found for video: {video_file.name}")
    
    return configs

def train_simple_models(train_samples, val_samples, config):
    """Train simplified models with fewer epochs for testing."""
    
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    
    # Create datasets with simpler transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SurgicalVideoDataset(
        samples=train_samples,
        phase_labels=config['phase_labels'],
        instrument_labels=config['instrument_labels'],
        transform=transform
    )
    
    val_dataset = SurgicalVideoDataset(
        samples=val_samples,
        phase_labels=config['phase_labels'],
        instrument_labels=config['instrument_labels'],
        transform=transform
    )
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    
    # Train simplified phase model with fewer epochs
    logger.info("Training Simplified Phase Recognition Model...")
    phase_model = SimplePhaseRecognitionModel(num_phases=len(config['phase_labels']))
    
    device = torch.device('cpu')  # Force CPU since we're in WSL
    phase_model = phase_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(phase_model.parameters(), lr=1e-4)
    
    # Train for just 5 epochs for testing
    num_epochs = 5
    logger.info(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        phase_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Handle the sequence data
            if len(sequences.shape) == 5:
                sequences = sequences[:, -1]  # Take last frame
            
            sequences = sequences.to(device)
            phase_labels = []
            
            for target in targets:
                if isinstance(target['phase'], int) and target['phase'] >= 0:
                    phase_labels.append(target['phase'])
                else:
                    phase_labels.append(0)  # Default to first phase
            
            phase_labels = torch.tensor(phase_labels, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = phase_model(sequences)
            loss = criterion(outputs, phase_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += phase_labels.size(0)
            correct += (predicted == phase_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    # Save model
    model_path = Path('/app/trained_models/simple_phase_model.pth')
    torch.save(phase_model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return phase_model

def main():
    """Main training pipeline."""
    
    logger.info("Starting simplified surgical video analysis pipeline...")
    
    # Setup configuration and paths
    config = setup_configuration()
    paths = setup_paths()
    
    # Save configuration
    config_file = paths['models_dir'] / 'model_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_file}")
    
    # Find video-annotation pairs
    annotation_configs = get_annotation_configs(paths)
    
    if not annotation_configs:
        logger.error("No video-annotation pairs found!")
        return
    
    logger.info(f"Found {len(annotation_configs)} video-annotation pairs")
    
    # Prepare training data with smaller sample rate for faster processing
    logger.info("Preparing training data...")
    try:
        # Override sample rate to process fewer frames
        for config_item in annotation_configs:
            parser = CVATAnnotationParser(
                xml_path=config_item['xml_path'],
                video_path=config_item['video_path']
            )
            # Extract samples every 60 frames instead of 30 for faster processing
            samples = parser.extract_training_samples(sample_rate=60)
            logger.info(f"Extracted {len(samples)} samples from {Path(config_item['video_path']).name}")
            
            if len(samples) > 0:
                # Just use a subset for quick testing
                samples = samples[:min(100, len(samples))]  # Max 100 samples
                train_samples = samples[:int(0.8 * len(samples))]
                val_samples = samples[int(0.8 * len(samples)):]
                
                logger.info(f"Using {len(train_samples)} training, {len(val_samples)} validation samples")
                
                # Train simplified model
                model = train_simple_models(train_samples, val_samples, config)
                
                logger.info("Quick training completed successfully!")
                logger.info("You can now test video analysis with the trained model.")
                break
            else:
                logger.warning(f"No samples extracted from {config_item['video_path']}")
                
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()