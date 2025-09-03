#!/usr/bin/env python3
"""
Script to analyze new surgical videos using trained models.
Usage: python analyze_video.py <video_path>
"""

import sys
import json
import logging
from pathlib import Path
from script import SurgicalVideoAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_video.py <video_path>")
        print("Example: python analyze_video.py videos/new_surgery.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Check if video exists
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Check if models exist
    models_dir = Path('trained_models')
    required_files = [
        'best_phase_model.pth',
        'instrument_detector_epoch_30.pth', 
        'model_config.json'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (models_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error("Missing trained model files:")
        for file_name in missing_files:
            logger.error(f"  - {models_dir / file_name}")
        logger.info("Please run training first: python setup_and_train.py")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        logger.info("Loading trained models...")
        analyzer = SurgicalVideoAnalyzer(
            phase_model_path=str(models_dir / 'best_phase_model.pth'),
            instrument_model_path=str(models_dir / 'instrument_detector_epoch_30.pth'),
            event_model_path=str(models_dir / 'best_phase_model.pth'),  # Using phase model as fallback
            config_path=str(models_dir / 'model_config.json')
        )
        
        # Analyze video
        logger.info(f"Analyzing video: {video_path}")
        report = analyzer.analyze_video(video_path, output_dir='./analysis_results')
        
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
        
        logger.info("Analysis completed successfully!")
        logger.info("Detailed results saved in: analysis_results/")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()