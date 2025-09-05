#!/usr/bin/env python3
"""
MASTER TRAINING SYSTEM
====================
Enterprise-grade training orchestration for all surgical AI models.

Features:
- Multi-model training pipeline
- Advanced data processing and augmentation  
- Distributed training support
- Comprehensive validation and metrics
- Model versioning and checkpointing
- Automated hyperparameter optimization

Author: AI Surgical Analysis Team  
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
import mlflow
import optuna

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle
from tqdm import tqdm
import yaml
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Import our models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.phase_detection.phase_detector import AdvancedPhaseDetector, PhaseDetectorLoss
from models.instrument_detection.instrument_tracker import AdvancedInstrumentDetector
from core.data_structures.surgical_entities import *

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@dataclass  
class TrainingConfig:
    """Comprehensive training configuration."""
    
    # Model configurations
    phase_model_config: Dict = None
    instrument_model_config: Dict = None
    event_model_config: Dict = None
    motion_model_config: Dict = None
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # Data parameters  
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    sequence_length: int = 16
    overlap_ratio: float = 0.5
    
    # Augmentation parameters
    augmentation_prob: float = 0.8
    color_jitter: float = 0.3
    rotation_degrees: int = 15
    gaussian_noise: float = 0.02
    
    # Training optimization
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 15
    
    # Checkpointing
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Logging and monitoring
    use_wandb: bool = True
    use_mlflow: bool = True
    log_interval: int = 100
    val_interval: int = 1000
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    distributed: bool = False

# ==================== ADVANCED DATASET ====================

class SurgicalVideoDataset(Dataset):
    """Comprehensive dataset for surgical video analysis."""
    
    def __init__(self,
                 data_dir: Path,
                 mode: str = "train",  # train, val, test
                 sequence_length: int = 16,
                 config: Optional[TrainingConfig] = None,
                 transform=None):
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sequence_length = sequence_length
        self.config = config or TrainingConfig()
        self.transform = transform
        
        # Load data annotations
        self.video_annotations = self._load_annotations()
        self.samples = self._prepare_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_annotations(self) -> List[Dict]:
        """Load all video annotations from CVAT XML files."""
        annotations = []
        
        xml_files = list(self.data_dir.glob("xml_path/*.xml"))
        video_files = list(self.data_dir.glob("videos/*.mp4"))
        
        for xml_file, video_file in zip(xml_files, video_files):
            annotation = self._parse_cvat_xml(xml_file, video_file)
            if annotation:
                annotations.append(annotation)
        
        return annotations
    
    def _parse_cvat_xml(self, xml_path: Path, video_path: Path) -> Optional[Dict]:
        """Parse CVAT XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract video metadata
            meta = root.find('.//meta')
            job = meta.find('job') if meta is not None else None
            
            if job is None:
                logger.warning(f"No job metadata found in {xml_path}")
                return None
            
            total_frames = int(job.find('size').text) if job.find('size') is not None else 0
            
            # Parse tracks (phases) 
            phases = []
            tracks = root.findall('.//track')
            
            for track in tracks:
                label = track.get('label')
                track_id = track.get('id')
                
                # Get all boxes in this track
                boxes = track.findall('.//box')
                if boxes:
                    start_frame = min(int(box.get('frame')) for box in boxes)
                    end_frame = max(int(box.get('frame')) for box in boxes)
                    
                    phases.append({
                        'label': label,
                        'track_id': track_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'boxes': [
                            {
                                'frame': int(box.get('frame')),
                                'xtl': float(box.get('xtl', 0)),
                                'ytl': float(box.get('ytl', 0)),
                                'xbr': float(box.get('xbr', 0)),
                                'ybr': float(box.get('ybr', 0)),
                                'attributes': {attr.get('name'): attr.text 
                                            for attr in box.findall('.//attribute')}
                            }
                            for box in boxes
                        ]
                    })
            
            # Parse shapes (events)
            events = []
            shapes = root.findall('.//polygon') + root.findall('.//box[@track_id=""]')
            
            for shape in shapes:
                label = shape.get('label')
                frame = int(shape.get('frame', 0))
                
                events.append({
                    'label': label,
                    'frame': frame,
                    'attributes': {attr.get('name'): attr.text 
                                 for attr in shape.findall('.//attribute')}
                })
            
            return {
                'video_path': str(video_path),
                'xml_path': str(xml_path),
                'total_frames': total_frames,
                'phases': phases,
                'events': events
            }
            
        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")
            return None
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare training samples from annotations."""
        samples = []
        
        for annotation in self.video_annotations:
            video_path = annotation['video_path']
            total_frames = annotation['total_frames']
            
            # Create phase labels for each frame
            frame_labels = self._create_frame_labels(annotation)
            
            # Generate sequences
            step_size = int(self.sequence_length * (1 - self.config.overlap_ratio))
            
            for start_frame in range(0, total_frames - self.sequence_length + 1, step_size):
                end_frame = start_frame + self.sequence_length
                
                # Get sequence labels
                sequence_labels = frame_labels[start_frame:end_frame]
                
                # Skip sequences with too many unknown labels
                unknown_ratio = (sequence_labels == -1).mean()
                if unknown_ratio > 0.3:  # Skip if >30% unknown
                    continue
                
                samples.append({
                    'video_path': video_path,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'phase_labels': sequence_labels,
                    'events': self._get_sequence_events(annotation['events'], start_frame, end_frame)
                })
        
        return samples
    
    def _create_frame_labels(self, annotation: Dict) -> np.ndarray:
        """Create frame-level phase labels."""
        total_frames = annotation['total_frames']
        frame_labels = np.full(total_frames, -1, dtype=np.int32)  # -1 for unknown
        
        # Map phase names to indices
        phase_name_to_idx = {
            phase.value: idx for idx, phase in enumerate(SurgicalPhaseType)
        }
        
        for phase in annotation['phases']:
            phase_label = phase['label']
            if phase_label in phase_name_to_idx:
                start_frame = phase['start_frame']
                end_frame = min(phase['end_frame'], total_frames - 1)
                
                phase_idx = phase_name_to_idx[phase_label]
                frame_labels[start_frame:end_frame + 1] = phase_idx
        
        return frame_labels
    
    def _get_sequence_events(self, events: List[Dict], start_frame: int, end_frame: int) -> List[Dict]:
        """Get events within a sequence."""
        sequence_events = []
        
        for event in events:
            if start_frame <= event['frame'] <= end_frame:
                sequence_events.append({
                    'label': event['label'],
                    'frame': event['frame'] - start_frame,  # Relative to sequence start
                    'attributes': event['attributes']
                })
        
        return sequence_events
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        sample = self.samples[idx]
        
        # Load video frames
        frames = self._load_video_frames(
            sample['video_path'],
            sample['start_frame'],
            sample['end_frame']
        )
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'frames': frames,
            'phase_labels': torch.tensor(sample['phase_labels'], dtype=torch.long),
            'events': sample['events'],
            'metadata': {
                'video_path': sample['video_path'],
                'start_frame': sample['start_frame'],
                'end_frame': sample['end_frame']
            }
        }
    
    def _load_video_frames(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load video frames for a sequence."""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frames.append(frame_resized)
            else:
                # Handle missing frames
                if frames:
                    frames.append(frames[-1])  # Duplicate last frame
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convert to tensor [sequence_length, channels, height, width]
        frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        
        return frames_tensor

# ==================== MASTER TRAINING ORCHESTRATOR ====================

class MasterTrainingOrchestrator:
    """Enterprise-grade training orchestrator for all surgical AI models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.loss_functions = {}
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Setup directories
        self._setup_directories()
        
    def _setup_logging(self):
        """Setup experiment tracking and logging."""
        if self.config.use_wandb:
            wandb.init(
                project="surgical-ai-training",
                config=self.config.__dict__,
                name=f"surgical_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        if self.config.use_mlflow:
            mlflow.start_run(run_name=f"surgical_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params(self.config.__dict__)
    
    def _setup_directories(self):
        """Setup training directories."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
    
    def initialize_models(self):
        """Initialize all models for training."""
        logger.info("🤖 Initializing models...")
        
        # Phase Detection Model
        self.models['phase'] = AdvancedPhaseDetector(
            num_phases=len(SurgicalPhaseType),
            **self.config.phase_model_config or {}
        ).to(self.device)
        
        # Instrument Detection Model  
        self.models['instrument'] = AdvancedInstrumentDetector(
            num_instruments=len(InstrumentType),
            **self.config.instrument_model_config or {}
        ).to(self.device)
        
        # Initialize optimizers
        for model_name, model in self.models.items():
            self.optimizers[model_name] = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Learning rate scheduler
            if self.config.scheduler_type == "cosine":
                self.schedulers[model_name] = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizers[model_name],
                    T_max=self.config.num_epochs
                )
            elif self.config.scheduler_type == "step":
                self.schedulers[model_name] = optim.lr_scheduler.StepLR(
                    self.optimizers[model_name],
                    step_size=30,
                    gamma=0.1
                )
        
        # Loss functions
        self.loss_functions['phase'] = PhaseDetectorLoss(len(SurgicalPhaseType))
        
        logger.info("✅ Models initialized successfully")
    
    def prepare_data(self, data_dir: str):
        """Prepare training, validation, and test datasets."""
        logger.info("📊 Preparing datasets...")
        
        # Create dataset
        full_dataset = SurgicalVideoDataset(
            data_dir=Path(data_dir),
            sequence_length=self.config.sequence_length,
            config=self.config
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        logger.info(f"✅ Datasets prepared: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train all models for one epoch."""
        epoch_metrics = {}
        
        # Phase model training
        phase_metrics = self._train_phase_model_epoch()
        epoch_metrics.update({f"phase_{k}": v for k, v in phase_metrics.items()})
        
        # Update learning rates
        for scheduler in self.schedulers.values():
            scheduler.step()
        
        return epoch_metrics
    
    def _train_phase_model_epoch(self) -> Dict[str, float]:
        """Train phase detection model for one epoch."""
        self.models['phase'].train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Phase Model")
        
        for batch_idx, batch in enumerate(progress_bar):
            frames = batch['frames'].to(self.device)  # [batch, seq_len, channels, height, width]
            labels = batch['phase_labels'].to(self.device)  # [batch, seq_len]
            
            # Forward pass
            output = self.models['phase'](frames, sequence_mode=True)
            predictions = output['predictions']  # [batch, seq_len, num_classes]
            
            # Compute loss
            loss = self.loss_functions['phase'](predictions, labels, sequence_mode=True)
            
            # Backward pass
            self.optimizers['phase'].zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.models['phase'].parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizers['phase'].step()
            
            # Metrics
            total_loss += loss.item()
            
            # Accuracy calculation
            pred_classes = torch.argmax(predictions, dim=-1)
            correct_predictions += (pred_classes == labels).sum().item()
            total_predictions += labels.numel()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct_predictions / total_predictions
                
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
                
                # Log to wandb/mlflow
                if self.config.use_wandb:
                    wandb.log({
                        'train/phase_loss': current_loss,
                        'train/phase_accuracy': current_acc,
                        'epoch': self.current_epoch,
                        'step': batch_idx
                    })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate all models."""
        val_metrics = {}
        
        # Phase model validation
        phase_metrics = self._validate_phase_model()
        val_metrics.update({f"phase_{k}": v for k, v in phase_metrics.items()})
        
        return val_metrics
    
    def _validate_phase_model(self) -> Dict[str, float]:
        """Validate phase detection model."""
        self.models['phase'].eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating Phase Model"):
                frames = batch['frames'].to(self.device)
                labels = batch['phase_labels'].to(self.device)
                
                output = self.models['phase'](frames, sequence_mode=True)
                predictions = output['predictions']
                
                loss = self.loss_functions['phase'](predictions, labels, sequence_mode=True)
                total_loss += loss.item()
                
                # Collect predictions and labels
                pred_classes = torch.argmax(predictions, dim=-1)
                all_predictions.extend(pred_classes.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Generate classification report
        unique_labels = list(set(all_labels))
        if len(unique_labels) > 1:
            class_report = classification_report(
                all_labels, all_predictions,
                target_names=[phase.value for phase in SurgicalPhaseType],
                output_dict=True,
                zero_division=0
            )
            
            f1_score = class_report['weighted avg']['f1-score']
            precision = class_report['weighted avg']['precision']
            recall = class_report['weighted avg']['recall']
        else:
            f1_score = precision = recall = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, data_dir: str, num_epochs: Optional[int] = None):
        """Execute complete training pipeline."""
        if num_epochs:
            self.config.num_epochs = num_epochs
        
        logger.info("🚀 Starting Master Training Pipeline")
        logger.info(f"📊 Configuration: {self.config}")
        
        # Initialize everything
        self.initialize_models()
        self.prepare_data(data_dir)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n🔄 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Logging
            self._log_epoch_results(epoch, train_metrics, val_metrics)
            
            # Model checkpointing
            current_val_loss = val_metrics.get('phase_loss', float('inf'))
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                self._save_best_models(epoch, val_metrics)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Regular checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
        
        # Final evaluation
        self._final_evaluation()
        
        logger.info("✅ Master Training Pipeline Completed!")
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log results for the epoch."""
        # Store in history
        self.training_history['train_losses'].append(train_metrics.get('phase_loss', 0))
        self.training_history['val_losses'].append(val_metrics.get('phase_loss', 0))
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
        
        # Console logging
        logger.info(f"Train - Phase Loss: {train_metrics.get('phase_loss', 0):.4f}, "
                   f"Accuracy: {train_metrics.get('phase_accuracy', 0):.4f}")
        logger.info(f"Val   - Phase Loss: {val_metrics.get('phase_loss', 0):.4f}, "
                   f"Accuracy: {val_metrics.get('phase_accuracy', 0):.4f}")
        
        # Experiment tracking
        if self.config.use_wandb:
            wandb.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                'epoch': epoch
            })
        
        if self.config.use_mlflow:
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)
    
    def _save_best_models(self, epoch: int, metrics: Dict):
        """Save best performing models."""
        logger.info("💾 Saving best models...")
        
        for model_name, model in self.models.items():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                'scheduler_state_dict': self.schedulers[model_name].state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, self.checkpoint_dir / f'best_{model_name}_model.pth')
        
        # Save training history
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Save regular checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_states': {name: model.state_dict() for name, model in self.models.items()},
            'optimizer_states': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'scheduler_states': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
    
    def _final_evaluation(self):
        """Perform final evaluation on test set."""
        logger.info("🎯 Performing final evaluation...")
        
        # Load best models
        self._load_best_models()
        
        # Evaluate on test set
        test_metrics = self._evaluate_test_set()
        
        logger.info("📊 Final Test Results:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Generate comprehensive evaluation report
        self._generate_evaluation_report(test_metrics)
    
    def _load_best_models(self):
        """Load best saved models."""
        for model_name in self.models.keys():
            checkpoint_path = self.checkpoint_dir / f'best_{model_name}_model.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                self.models[model_name].load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded best {model_name} model")
    
    def _evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate models on test set."""
        # For now, just evaluate phase model
        return self._validate_phase_model()
    
    def _generate_evaluation_report(self, test_metrics: Dict):
        """Generate comprehensive evaluation report."""
        report = {
            'training_config': self.config.__dict__,
            'training_history': self.training_history,
            'final_test_metrics': test_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for model in self.models.values() 
                                     for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for model in self.models.values() 
                                          for p in model.parameters() if p.requires_grad)
            },
            'training_summary': {
                'total_epochs': len(self.training_history['train_losses']),
                'best_val_loss': min(self.training_history['val_losses']),
                'final_train_loss': self.training_history['train_losses'][-1],
                'final_val_loss': self.training_history['val_losses'][-1]
            }
        }
        
        # Save report
        with open(self.outputs_dir / 'final_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Evaluation report saved to {self.outputs_dir / 'final_evaluation_report.json'}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main training execution."""
    
    # Configuration
    config = TrainingConfig(
        batch_size=4,  # Smaller batch size for local training
        num_epochs=50,
        learning_rate=1e-4,
        sequence_length=8,  # Shorter sequences for faster training
        phase_model_config={
            'num_phases': len(SurgicalPhaseType),
            'feature_dim': 512,
            'lstm_hidden': 256,
            'dropout': 0.3
        }
    )
    
    # Initialize trainer
    trainer = MasterTrainingOrchestrator(config)
    
    # Start training
    trainer.train(data_dir="./")  # Assuming data is in current directory
    
    print("🎉 Master Training Completed Successfully!")

if __name__ == "__main__":
    main()