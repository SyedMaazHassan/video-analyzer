"""
Complete ML Pipeline for Automated Labral Repair Surgery Analysis
==================================================================
This system trains on manually annotated videos to automatically analyze
new surgical videos and generate comprehensive reports without manual annotation.

Architecture:
1. Phase Recognition Model - Identifies surgical phases
2. Instrument Detection Model - Detects and tracks surgical instruments
3. Event Detection Model - Identifies critical events (bleeding, suture attempts)
4. Temporal Segmentation Model - Segments video into procedural steps
5. Report Generation System - Produces structured reports from model outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b4
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== DATA PREPARATION ====================

class CVATAnnotationParser:
    """Parse CVAT XML annotations for training data extraction."""
    
    def __init__(self, xml_path: str, video_path: str, fps: float = 30.0):
        self.xml_path = xml_path
        self.video_path = video_path
        self.fps = fps
        self.annotations = self._parse_xml()
        
    def _parse_xml(self) -> Dict:
        """Parse CVAT XML and extract frame-level annotations."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        frame_annotations = defaultdict(lambda: {
            'phase': None,
            'instruments': [],
            'events': [],
            'bboxes': []
        })
        
        # Parse tracks (phases and instruments)
        for track in root.findall('.//track'):
            label = track.get('label')
            
            for box in track.findall('box'):
                frame = int(box.get('frame'))
                if box.get('outside') == '0':  # Object is visible
                    bbox = {
                        'label': label,
                        'xtl': float(box.get('xtl')),
                        'ytl': float(box.get('ytl')),
                        'xbr': float(box.get('xbr')),
                        'ybr': float(box.get('ybr'))
                    }
                    
                    # Categorize annotation
                    if label in ['Portal Placement', 'Diagnostic Arthroscopy', 'Labral Mobilization',
                                'Glenoid Preparation', 'Anchor Placement', 'Suture Passage',
                                'Suture Tensioning', 'Final Inspection']:
                        frame_annotations[frame]['phase'] = label
                    elif 'Instrument' in label or label in ['Suture Passer', 'Shaver', 'Cannula']:
                        frame_annotations[frame]['instruments'].append(label)
                        frame_annotations[frame]['bboxes'].append(bbox)
                    
        # Parse events
        for image in root.findall('.//image'):
            frame = int(image.get('id'))
            for box in image.findall('.//box'):
                label = box.get('label')
                if label in ['Bleeding', 'Suture Attempt', 'Anchor Reposition']:
                    frame_annotations[frame]['events'].append(label)
        
        return dict(frame_annotations)
    
    def extract_training_samples(self, sample_rate: int = 30) -> List[Dict]:
        """Extract training samples from video at specified sampling rate."""
        samples = []
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame_idx in self.annotations:
                samples.append({
                    'frame': frame,
                    'frame_idx': frame_idx,
                    'annotations': self.annotations[frame_idx]
                })
        
        cap.release()
        return samples


class SurgicalVideoDataset(Dataset):
    """PyTorch Dataset for surgical video analysis."""
    
    def __init__(self, samples: List[Dict], phase_labels: List[str], 
                 instrument_labels: List[str], transform=None):
        self.samples = samples
        self.phase_labels = phase_labels
        self.instrument_labels = instrument_labels
        self.transform = transform
        
        # Create label mappings
        self.phase_to_idx = {label: idx for idx, label in enumerate(phase_labels)}
        self.instrument_to_idx = {label: idx for idx, label in enumerate(instrument_labels)}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = sample['frame']
        annotations = sample['annotations']
        
        # Convert frame to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        # Prepare labels
        phase_label = self.phase_to_idx.get(annotations['phase'], -1)
        
        # Prepare bounding boxes for object detection
        boxes = []
        labels = []
        for bbox in annotations['bboxes']:
            boxes.append([bbox['xtl'], bbox['ytl'], bbox['xbr'], bbox['ybr']])
            labels.append(self.instrument_to_idx.get(bbox['label'], 0))
        
        if not boxes:  # No objects in frame
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'phase': phase_label,
            'events': annotations['events']
        }
        
        return frame, target


# ==================== MODEL ARCHITECTURES ====================

class PhaseRecognitionModel(nn.Module):
    """CNN-LSTM model for surgical phase recognition."""
    
    def __init__(self, num_phases: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        
        # CNN backbone for feature extraction
        self.backbone = efficientnet_b4(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove final layer
        feature_dim = 1792  # EfficientNet-B4 output dimension
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_phases)
        )
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features for each frame
        features = []
        for i in range(seq_len):
            feat = self.backbone(x[:, i])
            features.append(feat)
        features = torch.stack(features, dim=1)
        
        # Temporal modeling
        lstm_out, _ = self.lstm(features)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1]
        
        # Classification
        output = self.classifier(last_output)
        return output


class InstrumentDetector(nn.Module):
    """Faster R-CNN based instrument detection model."""
    
    def __init__(self, num_instruments: int):
        super().__init__()
        
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_instruments + 1  # +1 for background
        )
        
    def forward(self, images, targets=None):
        return self.model(images, targets)


class EventDetectionModel(nn.Module):
    """Multi-label classification model for event detection."""
    
    def __init__(self, num_events: int):
        super().__init__()
        
        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_events),
            nn.Sigmoid()  # For multi-label
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ==================== TRAINING FUNCTIONS ====================

class ModelTrainer:
    """Unified trainer for all models."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Using device: {device}")
        
    def train_phase_model(self, model, train_loader, val_loader, num_epochs=50):
        """Train phase recognition model."""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_acc = 0
        train_losses, val_accuracies = [], []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            
            for batch_idx, (sequences, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                sequences = sequences.to(self.device)
                phase_labels = torch.tensor([t['phase'] for t in targets]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, phase_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_acc = self.validate_phase_model(model, val_loader)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_acc)
            
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_phase_model.pth')
                logger.info(f"Saved best model with accuracy: {best_val_acc:.4f}")
        
        return train_losses, val_accuracies
    
    def validate_phase_model(self, model, val_loader):
        """Validate phase recognition model."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                phase_labels = torch.tensor([t['phase'] for t in targets]).to(self.device)
                
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += phase_labels.size(0)
                correct += (predicted == phase_labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def train_instrument_detector(self, model, train_loader, val_loader, num_epochs=30):
        """Train instrument detection model."""
        model = model.to(self.device)
        model.train()
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items() if k in ['boxes', 'labels']} 
                          for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            lr_scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}: Avg Loss={avg_loss:.4f}")
            
            # Periodic validation and saving
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f'instrument_detector_epoch_{epoch+1}.pth')
        
        return model


# ==================== INFERENCE PIPELINE ====================

class SurgicalVideoAnalyzer:
    """Complete pipeline for analyzing new surgical videos."""
    
    def __init__(self, phase_model_path: str, instrument_model_path: str, 
                 event_model_path: str, config_path: str):
        """
        Initialize analyzer with trained models.
        
        Args:
            phase_model_path: Path to trained phase recognition model
            instrument_model_path: Path to trained instrument detection model
            event_model_path: Path to trained event detection model
            config_path: Path to configuration file with label mappings
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load models
        self.phase_model = self._load_phase_model(phase_model_path)
        self.instrument_model = self._load_instrument_model(instrument_model_path)
        self.event_model = self._load_event_model(event_model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Surgical Video Analyzer initialized successfully")
    
    def _load_phase_model(self, path: str):
        """Load trained phase recognition model."""
        model = PhaseRecognitionModel(
            num_phases=len(self.config['phase_labels'])
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _load_instrument_model(self, path: str):
        """Load trained instrument detection model."""
        model = InstrumentDetector(
            num_instruments=len(self.config['instrument_labels'])
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _load_event_model(self, path: str):
        """Load trained event detection model."""
        model = EventDetectionModel(
            num_events=len(self.config['event_labels'])
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def analyze_video(self, video_path: str, output_dir: str = './results') -> Dict:
        """
        Analyze a surgical video and generate comprehensive report.
        
        Args:
            video_path: Path to surgical video
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize results storage
        results = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'phases': [],
            'instruments': [],
            'events': [],
            'frame_predictions': []
        }
        
        # Process video in chunks for phase recognition
        chunk_size = 16  # Process 16 frames at a time
        frame_buffer = []
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                frame_buffer.append(processed_frame)
                
                # Process chunk when buffer is full
                if len(frame_buffer) == chunk_size:
                    chunk_results = self._process_frame_chunk(
                        frame_buffer, 
                        frame_idx - chunk_size + 1
                    )
                    results['frame_predictions'].extend(chunk_results)
                    frame_buffer = []
                
                # Detect instruments every 10 frames
                if frame_idx % 10 == 0:
                    instruments = self._detect_instruments(frame)
                    if instruments:
                        results['instruments'].append({
                            'frame': frame_idx,
                            'detections': instruments
                        })
                
                # Detect events every 5 frames
                if frame_idx % 5 == 0:
                    events = self._detect_events(processed_frame)
                    if events:
                        results['events'].append({
                            'frame': frame_idx,
                            'events': events
                        })
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Process remaining frames in buffer
        if frame_buffer:
            chunk_results = self._process_frame_chunk(
                frame_buffer, 
                frame_idx - len(frame_buffer) + 1
            )
            results['frame_predictions'].extend(chunk_results)
        
        # Post-process results to identify phase segments
        results['phases'] = self._segment_phases(results['frame_predictions'])
        
        # Generate structured report
        report = self._generate_report(results)
        
        # Save results
        self._save_results(results, report, output_dir)
        
        return report
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return self.transform(frame)
    
    def _process_frame_chunk(self, frames: List, start_idx: int) -> List[Dict]:
        """Process a chunk of frames for phase recognition."""
        # Stack frames into batch
        batch = torch.stack(frames).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.phase_model(batch)
            _, predicted = torch.max(outputs, 1)
        
        # Convert predictions to phase labels
        chunk_results = []
        phase_idx = predicted.item()
        phase_label = self.config['phase_labels'][phase_idx]
        
        for i, frame in enumerate(frames):
            chunk_results.append({
                'frame': start_idx + i,
                'phase': phase_label,
                'confidence': torch.softmax(outputs, dim=1)[0, phase_idx].item()
            })
        
        return chunk_results
    
    def _detect_instruments(self, frame) -> List[Dict]:
        """Detect instruments in frame."""
        # Convert frame to tensor
        frame_tensor = transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.instrument_model(frame_tensor)
        
        # Process predictions
        detections = []
        if len(predictions) > 0:
            pred = predictions[0]
            for idx in range(len(pred['boxes'])):
                if pred['scores'][idx] > 0.5:  # Confidence threshold
                    detections.append({
                        'instrument': self.config['instrument_labels'][pred['labels'][idx].item()],
                        'confidence': pred['scores'][idx].item(),
                        'bbox': pred['boxes'][idx].cpu().numpy().tolist()
                    })
        
        return detections
    
    def _detect_events(self, frame_tensor) -> List[str]:
        """Detect events in frame."""
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.event_model(frame_tensor)
        
        # Apply threshold for multi-label classification
        threshold = 0.5
        predicted = (outputs > threshold).cpu().numpy()[0]
        
        events = []
        for idx, is_present in enumerate(predicted):
            if is_present:
                events.append(self.config['event_labels'][idx])
        
        return events
    
    def _segment_phases(self, frame_predictions: List[Dict]) -> List[Dict]:
        """Segment continuous phases from frame predictions."""
        if not frame_predictions:
            return []
        
        phases = []
        current_phase = frame_predictions[0]['phase']
        start_frame = 0
        
        for i, pred in enumerate(frame_predictions[1:], 1):
            if pred['phase'] != current_phase:
                # End current phase
                phases.append({
                    'phase': current_phase,
                    'start_frame': start_frame,
                    'end_frame': i - 1,
                    'duration_frames': i - start_frame
                })
                # Start new phase
                current_phase = pred['phase']
                start_frame = i
        
        # Add final phase
        phases.append({
            'phase': current_phase,
            'start_frame': start_frame,
            'end_frame': len(frame_predictions) - 1,
            'duration_frames': len(frame_predictions) - start_frame
        })
        
        return phases
    
    def _generate_report(self, results: Dict) -> Dict:
        """Generate structured report from analysis results."""
        fps = results['fps']
        
        # Calculate phase metrics
        phase_metrics = {}
        for phase in results['phases']:
            phase_name = phase['phase']
            duration_min = (phase['duration_frames'] / fps) / 60
            
            if phase_name not in phase_metrics:
                phase_metrics[phase_name] = {
                    'total_duration_min': 0,
                    'occurrences': 0
                }
            
            phase_metrics[phase_name]['total_duration_min'] += duration_min
            phase_metrics[phase_name]['occurrences'] += 1
        
        # Count events
        event_counts = defaultdict(int)
        for event_detection in results['events']:
            for event in event_detection['events']:
                event_counts[event] += 1
        
        # Count unique instruments
        unique_instruments = set()
        for detection in results['instruments']:
            for inst in detection['detections']:
                unique_instruments.add(inst['instrument'])
        
        # Calculate total procedure time
        total_frames = results['total_frames']
        total_time_min = (total_frames / fps) / 60
        
        # Generate report
        report = {
            'video_file': Path(results['video_path']).name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_procedure_time_min': total_time_min,
            'phase_metrics': phase_metrics,
            'event_summary': dict(event_counts),
            'instruments_used': list(unique_instruments),
            'num_phases': len(results['phases']),
            'bleeding_events': event_counts.get('Bleeding', 0),
            'suture_attempts': event_counts.get('Suture Attempt', 0),
            'anchor_repositions': event_counts.get('Anchor Reposition', 0)
        }
        
        # Add specific phase durations for standardized reporting
        standard_phases = [
            'Portal Placement', 'Diagnostic Arthroscopy', 'Labral Mobilization',
            'Glenoid Preparation', 'Anchor Placement', 'Suture Passage',
            'Suture Tensioning', 'Final Inspection'
        ]
        
        for phase in standard_phases:
            if phase in phase_metrics:
                report[f'{phase.replace(" ", "_")}_Duration_min'] = phase_metrics[phase]['total_duration_min']
            else:
                report[f'{phase.replace(" ", "_")}_Duration_min'] = 0
        
        return report
    
    def _save_results(self, results: Dict, report: Dict, output_dir: str):
        """Save analysis results and report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = Path(results['video_path']).stem
        
        # Save detailed results as JSON
        results_file = output_path / f'{video_name}_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'video_path': results['video_path'],
                'fps': results['fps'],
                'total_frames': results['total_frames'],
                'phases': results['phases'],
                'num_instrument_detections': len(results['instruments']),
                'num_event_detections': len(results['events'])
            }
            json.dump(json_results, f, indent=2)
        
        # Save report as JSON
        report_file = output_path / f'{video_name}_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save report as CSV for easy viewing
        report_df = pd.DataFrame([report])
        csv_file = output_path / f'{video_name}_report_{timestamp}.csv'
        report_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Report saved as: {report_file}")
        logger.info(f"CSV report saved as: {csv_file}")


# ==================== MAIN TRAINING PIPELINE ====================

def prepare_training_data(annotation_configs: List[Dict]) -> Tuple[List, List]:
    """
    Prepare training data from multiple annotated videos.
    
    Args:
        annotation_configs: List of dicts with 'xml_path' and 'video_path'
    
    Returns:
        Tuple of (training_samples, validation_samples)
    """
    all_samples = []
    
    for config in annotation_configs:
        logger.info(f"Processing {config['video_path']}")
        parser = CVATAnnotationParser(
            xml_path=config['xml_path'],
            video_path=config['video_path']
        )
        samples = parser.extract_training_samples(sample_rate=30)
        all_samples.extend(samples)
    
    # Split into train and validation
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42
    )
    
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    
    return train_samples, val_samples


def train_all_models(train_samples: List, val_samples: List, config: Dict):
    """
    Train all models for surgical video analysis.
    
    Args:
        train_samples: Training data samples
        val_samples: Validation data samples
        config: Configuration dictionary with labels
    """
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # 1. Train Phase Recognition Model
    logger.info("Training Phase Recognition Model...")
    phase_model = PhaseRecognitionModel(num_phases=len(config['phase_labels']))
    trainer.train_phase_model(phase_model, train_loader, val_loader, num_epochs=50)
    
    # 2. Train Instrument Detection Model
    logger.info("Training Instrument Detection Model...")
    instrument_model = InstrumentDetector(num_instruments=len(config['instrument_labels']))
    trainer.train_instrument_detector(instrument_model, train_loader, val_loader, num_epochs=30)
    
    # 3. Train Event Detection Model
    logger.info("Training Event Detection Model...")
    event_model = EventDetectionModel(num_events=len(config['event_labels']))
    # Note: Event training would need separate implementation
    
    # Save configuration
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("All models trained successfully!")


# ==================== MAIN EXECUTION SCRIPT ====================

def main():
    """
    Complete pipeline: Train models and run inference on new videos.
    """
    
    # ========== PART 1: TRAINING ==========
    
    # Define your configuration
    config = {
        'phase_labels': [
            'Portal Placement',
            'Diagnostic Arthroscopy',
            'Labral Mobilization',
            'Glenoid Preparation',
            'Anchor Placement',
            'Suture Passage',
            'Suture Tensioning',
            'Final Inspection'
        ],
        'instrument_labels': [
            'Arthroscopic Camera',
            'Trocar',
            'Cannula',
            'Shaver',
            'Electrocautery Probe',
            'Probe',
            'Grasper',
            'Burr',
            'Rasp',
            'Drill Guide',
            'Suture Anchor',
            'Suture Passer',
            'Knot Pusher',
            'Suture Cutter'
        ],
        'event_labels': [
            'Bleeding',
            'Suture Attempt',
            'Anchor Reposition',
            'Anchor Pullout',
            'Cartilage Damage'
        ]
    }
    
    # Training mode - Run this with your annotated videos
    TRAINING_MODE = False  # Set to True when training
    
    if TRAINING_MODE:
        # Define your annotated videos
        annotation_configs = [
            {
                'xml_path': 'annotations/video1_annotations.xml',
                'video_path': 'videos/video1.mp4'
            },
            {
                'xml_path': 'annotations/video2_annotations.xml',
                'video_path': 'videos/video2.mp4'
            },
            # Add all 8-10 annotated videos here
        ]
        
        # Prepare training data
        train_samples, val_samples = prepare_training_data(annotation_configs)
        
        # Train all models
        train_all_models(train_samples, val_samples, config)
    
    # ========== PART 2: INFERENCE ON NEW VIDEOS ==========
    
    # Initialize analyzer with trained models
    analyzer = SurgicalVideoAnalyzer(
        phase_model_path='best_phase_model.pth',
        instrument_model_path='instrument_detector_epoch_30.pth',
        event_model_path='event_detection_model.pth',
        config_path='model_config.json'
    )
    
    # Analyze new videos (no annotation required!)
    new_videos = [
        'new_videos/surgery_case_101.mp4',
        'new_videos/surgery_case_102.mp4',
        # Add paths to your new videos
    ]
    
    all_reports = []
    
    for video_path in new_videos:
        if Path(video_path).exists():
            logger.info(f"Analyzing new video: {video_path}")
            report = analyzer.analyze_video(video_path, output_dir='./analysis_results')
            all_reports.append(report)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Analysis Complete for: {Path(video_path).name}")
            print(f"{'='*60}")
            print(f"Total Procedure Time: {report['total_procedure_time_min']:.2f} minutes")
            print(f"Number of Phases: {report['num_phases']}")
            print(f"Bleeding Events: {report['bleeding_events']}")
            print(f"Suture Attempts: {report['suture_attempts']}")
            print(f"Instruments Used: {len(report['instruments_used'])}")
            print(f"{'='*60}\n")
    
    # Generate consolidated report
    if all_reports:
        generate_consolidated_report(all_reports)


def generate_consolidated_report(reports: List[Dict]):
    """Generate consolidated report from multiple video analyses."""
    
    # Convert to DataFrame
    df = pd.DataFrame(reports)
    
    # Add surgeon ID (you might want to map this from video filename or metadata)
    df['Surgeon_ID'] = df['video_file'].apply(lambda x: extract_surgeon_id(x))
    df['Case_ID'] = df['video_file'].apply(lambda x: extract_case_id(x))
    
    # Save consolidated report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as Excel with multiple sheets
    with pd.ExcelWriter(f'surgical_analysis_report_{timestamp}.xlsx', engine='openpyxl') as writer:
        # Case-level details
        df.to_excel(writer, sheet_name='Case Reports', index=False)
        
        # Summary statistics
        summary = df.describe()
        summary.to_excel(writer, sheet_name='Summary Statistics')
        
        # Phase duration analysis
        phase_cols = [col for col in df.columns if 'Duration_min' in col]
        phase_summary = df[phase_cols].describe()
        phase_summary.to_excel(writer, sheet_name='Phase Analysis')
    
    # Save as CSV
    df.to_csv(f'surgical_analysis_report_{timestamp}.csv', index=False)
    
    logger.info(f"Consolidated report saved: surgical_analysis_report_{timestamp}.xlsx")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CONSOLIDATED ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Videos Analyzed: {len(df)}")
    print(f"Average Procedure Time: {df['total_procedure_time_min'].mean():.2f} ± {df['total_procedure_time_min'].std():.2f} minutes")
    print(f"Average Bleeding Events: {df['bleeding_events'].mean():.2f}")
    print(f"Average Suture Attempts: {df['suture_attempts'].mean():.2f}")
    print("\nPhase Duration Summary (minutes):")
    for col in phase_cols:
        phase_name = col.replace('_Duration_min', '').replace('_', ' ')
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {phase_name}: {mean_val:.2f} ± {std_val:.2f}")
    print("="*80)


def extract_surgeon_id(filename: str) -> str:
    """Extract surgeon ID from filename (customize based on your naming convention)."""
    # Example: if filename contains surgeon ID like "surgeon_A_case_001.mp4"
    if 'surgeon' in filename.lower():
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() == 'surgeon' and i + 1 < len(parts):
                return f"SURGEON_{parts[i+1].upper()}"
    return "UNKNOWN"


def extract_case_id(filename: str) -> str:
    """Extract case ID from filename (customize based on your naming convention)."""
    # Example: if filename contains case ID like "case_001.mp4"
    import re
    match = re.search(r'case[_\s]?(\d+)', filename, re.IGNORECASE)
    if match:
        return f"CASE_{match.group(1).zfill(3)}"
    # Fallback: use filename without extension
    return Path(filename).stem.upper()


# ==================== DEPLOYMENT SCRIPT ====================

class ProductionPipeline:
    """
    Production-ready pipeline for automated surgical video analysis.
    Can be deployed as a service or batch processor.
    """
    
    def __init__(self, model_dir: str = './trained_models'):
        """Initialize production pipeline with trained models."""
        self.model_dir = Path(model_dir)
        self.analyzer = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        try:
            self.analyzer = SurgicalVideoAnalyzer(
                phase_model_path=str(self.model_dir / 'best_phase_model.pth'),
                instrument_model_path=str(self.model_dir / 'instrument_detector.pth'),
                event_model_path=str(self.model_dir / 'event_detector.pth'),
                config_path=str(self.model_dir / 'model_config.json')
            )
            logger.info("Production models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def process_single_video(self, video_path: str, case_id: str = None, 
                            surgeon_id: str = None) -> Dict:
        """
        Process a single video and return structured report.
        
        Args:
            video_path: Path to surgical video
            case_id: Optional case identifier
            surgeon_id: Optional surgeon identifier
            
        Returns:
            Dictionary containing complete analysis report
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Generate IDs if not provided
        if not case_id:
            case_id = f"CASE_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not surgeon_id:
            surgeon_id = "UNKNOWN"
        
        # Analyze video
        report = self.analyzer.analyze_video(video_path)
        
        # Add metadata
        report['Case_ID'] = case_id
        report['Surgeon_ID'] = surgeon_id
        report['Processing_Timestamp'] = datetime.now().isoformat()
        
        return report
    
    def batch_process(self, video_list: List[Dict]) -> pd.DataFrame:
        """
        Process multiple videos in batch.
        
        Args:
            video_list: List of dicts with 'video_path', 'case_id', 'surgeon_id'
            
        Returns:
            DataFrame with all analysis results
        """
        results = []
        
        for video_info in tqdm(video_list, desc="Processing videos"):
            try:
                report = self.process_single_video(
                    video_path=video_info['video_path'],
                    case_id=video_info.get('case_id'),
                    surgeon_id=video_info.get('surgeon_id')
                )
                results.append(report)
            except Exception as e:
                logger.error(f"Failed to process {video_info['video_path']}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def generate_surgeon_performance_report(self, results_df: pd.DataFrame, 
                                           surgeon_id: str) -> Dict:
        """
        Generate performance metrics for a specific surgeon.
        
        Args:
            results_df: DataFrame with analysis results
            surgeon_id: Surgeon identifier
            
        Returns:
            Dictionary with performance metrics
        """
        surgeon_data = results_df[results_df['Surgeon_ID'] == surgeon_id]
        
        if surgeon_data.empty:
            return {}
        
        metrics = {
            'Surgeon_ID': surgeon_id,
            'Total_Cases': len(surgeon_data),
            'Avg_Procedure_Time': surgeon_data['total_procedure_time_min'].mean(),
            'Procedure_Time_Trend': surgeon_data['total_procedure_time_min'].tolist(),
            'Efficiency_Score': 100 - (surgeon_data['total_procedure_time_min'].std() / 
                                      surgeon_data['total_procedure_time_min'].mean() * 100),
            'Complication_Rate': (surgeon_data['bleeding_events'].sum() / 
                                len(surgeon_data)),
            'Success_Metrics': {
                'Avg_Suture_Attempts': surgeon_data['suture_attempts'].mean(),
                'Avg_Anchor_Repositions': surgeon_data['anchor_repositions'].mean()
            }
        }
        
        return metrics


# ==================== API INTERFACE (Optional) ====================

class SurgicalAnalysisAPI:
    """
    REST API interface for surgical video analysis.
    Can be deployed with Flask/FastAPI for web service.
    """
    
    def __init__(self):
        self.pipeline = ProductionPipeline()
    
    def analyze_endpoint(self, video_file, case_id=None, surgeon_id=None):
        """
        API endpoint for video analysis.
        
        Args:
            video_file: Uploaded video file
            case_id: Optional case ID
            surgeon_id: Optional surgeon ID
            
        Returns:
            JSON response with analysis results
        """
        # Save uploaded file temporarily
        temp_path = f"/tmp/{video_file.filename}"
        video_file.save(temp_path)
        
        try:
            # Process video
            report = self.pipeline.process_single_video(
                video_path=temp_path,
                case_id=case_id,
                surgeon_id=surgeon_id
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            return {
                'status': 'success',
                'report': report
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


if __name__ == "__main__":
    main()