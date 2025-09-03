#!/usr/bin/env python3
"""
Simplified setup and training script for surgical video analysis.
This script configures paths and runs the training pipeline.
"""

import os
import json
import logging
from pathlib import Path
from script import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        ],
        'anatomy_labels': [
            'Labrum',
            'Glenoid',
            'Humeral Head', 
            'Biceps Tendon',
            'Capsule',
            'Rotator Cuff',
            'Cartilage',
            'Subscapularis'
        ]
    }
    
    return config

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

def main():
    """Main training pipeline."""
    
    logger.info("Starting surgical video analysis pipeline setup...")
    
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
        logger.info("Please ensure you have:")
        logger.info("- Video files in: /app/videos/")
        logger.info("- Corresponding XML files in: /app/xml_path/")
        return
    
    logger.info(f"Found {len(annotation_configs)} video-annotation pairs")
    
    # Prepare training data
    logger.info("Preparing training data...")
    try:
        train_samples, val_samples = prepare_training_data(annotation_configs)
        
        if len(train_samples) == 0:
            logger.error("No training samples extracted! Check your annotation files.")
            return
            
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        return
    
    # Train models
    logger.info("Starting model training...")
    try:
        train_all_models(train_samples, val_samples, config)
        logger.info("Training completed successfully!")
        
        # List trained models
        model_files = list(paths['models_dir'].glob('*.pth'))
        logger.info(f"Trained models saved:")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return
    
    logger.info("Setup and training completed!")
    logger.info("You can now use the trained models to analyze new videos.")

if __name__ == "__main__":
    main()