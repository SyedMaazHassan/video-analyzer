import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime

class ModelType(Enum):
    PHASE_DETECTION = "phase_detection"
    INSTRUMENT_TRACKING = "instrument_tracking"  
    EVENT_DETECTION = "event_detection"
    MOTION_ANALYSIS = "motion_analysis"

class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"

@dataclass
class ModelConfig:
    """Configuration for individual AI models"""
    model_type: ModelType
    model_path: Optional[str] = None
    enabled: bool = True
    confidence_threshold: float = 0.5
    batch_size: int = 32
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingConfig:
    """Video processing configuration"""
    target_fps: float = 30.0
    frame_skip: int = 1
    resize_width: int = 640
    resize_height: int = 480
    parallel_processing: bool = True
    max_workers: int = 4
    gpu_enabled: bool = True
    gpu_memory_limit: Optional[float] = None  # GB

@dataclass
class OutputConfig:
    """Output configuration"""
    formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.JSON, OutputFormat.CSV, OutputFormat.EXCEL])
    output_directory: str = "results"
    create_timestamp_folder: bool = True
    detailed_timeline: bool = True
    include_confidence_scores: bool = True
    export_visualizations: bool = False

@dataclass
class PhaseDetectionConfig:
    """Specific configuration for phase detection model"""
    sequence_length: int = 30
    overlap_threshold: float = 0.7
    temporal_smoothing: bool = True
    smoothing_window: int = 5
    min_phase_duration: float = 2.0  # seconds
    transition_buffer: float = 0.5   # seconds

@dataclass
class InstrumentTrackingConfig:
    """Specific configuration for instrument tracking model"""
    max_tracked_objects: int = 10
    tracking_confidence_threshold: float = 0.3
    nms_threshold: float = 0.5
    track_buffer: int = 30
    match_threshold: float = 0.8
    usage_timeline_enabled: bool = True

@dataclass
class EventDetectionConfig:
    """Specific configuration for event detection model"""
    bleeding_detection_threshold: float = 0.4
    suture_attempt_threshold: float = 0.5
    anchor_event_threshold: float = 0.6
    event_clustering_enabled: bool = True
    clustering_time_window: float = 2.0  # seconds
    min_event_duration: float = 0.5      # seconds

@dataclass
class MotionAnalysisConfig:
    """Specific configuration for motion analysis model"""
    motion_threshold_low: float = 0.1
    motion_threshold_high: float = 0.3
    idle_threshold_short: float = 3.0    # seconds
    idle_threshold_long: float = 10.0    # seconds
    instrument_confidence_threshold: float = 0.3
    optical_flow_enabled: bool = True
    smoothing_window: int = 5

@dataclass
class SystemConfiguration:
    """Master system configuration"""
    # Core settings
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Component configurations
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Model-specific configurations
    phase_detection: PhaseDetectionConfig = field(default_factory=PhaseDetectionConfig)
    instrument_tracking: InstrumentTrackingConfig = field(default_factory=InstrumentTrackingConfig)
    event_detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    motion_analysis: MotionAnalysisConfig = field(default_factory=MotionAnalysisConfig)
    
    # Model registry
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "phase_detector": ModelConfig(ModelType.PHASE_DETECTION),
        "instrument_tracker": ModelConfig(ModelType.INSTRUMENT_TRACKING),
        "event_detector": ModelConfig(ModelType.EVENT_DETECTION),
        "motion_analyzer": ModelConfig(ModelType.MOTION_ANALYSIS)
    })
    
    # Advanced settings
    logging_level: str = "INFO"
    enable_profiling: bool = False
    cache_enabled: bool = True
    cache_size_mb: int = 512

class ConfigurationManager:
    """Professional configuration management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config: SystemConfiguration = SystemConfiguration()
        self.logger = self._setup_logger()
        
        # Load configuration if exists
        if self.config_path.exists():
            self.load_configuration()
        else:
            self.save_configuration()  # Create default config
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        config_dir = Path(__file__).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "surgical_ai_config.yaml"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup configuration manager logger"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_configuration(self, config_path: Optional[str] = None) -> SystemConfiguration:
        """Load configuration from file"""
        if config_path:
            self.config_path = Path(config_path)
        
        try:
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
            elif self.config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            # Update configuration object
            self._update_config_from_dict(config_dict)
            
            # Update modification timestamp
            self.config.modified_at = datetime.now().isoformat()
            
            self.logger.info(f"Configuration loaded from: {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.logger.info("Using default configuration")
            return self.config
    
    def save_configuration(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if config_path:
            self.config_path = Path(config_path)
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update modification timestamp
        self.config.modified_at = datetime.now().isoformat()
        
        try:
            config_dict = asdict(self.config)
            
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif self.config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration object from dictionary"""
        try:
            # Update processing config
            if 'processing' in config_dict:
                proc_dict = config_dict['processing']
                self.config.processing = ProcessingConfig(**proc_dict)
            
            # Update output config
            if 'output' in config_dict:
                out_dict = config_dict['output']
                if 'formats' in out_dict:
                    out_dict['formats'] = [OutputFormat(fmt) if isinstance(fmt, str) else fmt 
                                         for fmt in out_dict['formats']]
                self.config.output = OutputConfig(**out_dict)
            
            # Update model-specific configs
            if 'phase_detection' in config_dict:
                self.config.phase_detection = PhaseDetectionConfig(**config_dict['phase_detection'])
            
            if 'instrument_tracking' in config_dict:
                self.config.instrument_tracking = InstrumentTrackingConfig(**config_dict['instrument_tracking'])
            
            if 'event_detection' in config_dict:
                self.config.event_detection = EventDetectionConfig(**config_dict['event_detection'])
            
            if 'motion_analysis' in config_dict:
                self.config.motion_analysis = MotionAnalysisConfig(**config_dict['motion_analysis'])
            
            # Update models registry
            if 'models' in config_dict:
                for model_name, model_dict in config_dict['models'].items():
                    if isinstance(model_dict, dict):
                        model_type = ModelType(model_dict.get('model_type', model_name))
                        self.config.models[model_name] = ModelConfig(
                            model_type=model_type,
                            **{k: v for k, v in model_dict.items() if k != 'model_type'}
                        )
            
            # Update scalar values
            for key in ['version', 'created_at', 'logging_level', 'enable_profiling', 'cache_enabled', 'cache_size_mb']:
                if key in config_dict:
                    setattr(self.config, key, config_dict[key])
                    
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.config.models.get(model_name)
    
    def update_model_config(self, model_name: str, **kwargs) -> None:
        """Update configuration for specific model"""
        if model_name in self.config.models:
            model_config = self.config.models[model_name]
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                else:
                    model_config.custom_parameters[key] = value
        else:
            self.logger.warning(f"Model '{model_name}' not found in configuration")
    
    def set_model_path(self, model_name: str, model_path: str) -> None:
        """Set model file path"""
        if model_name in self.config.models:
            self.config.models[model_name].model_path = model_path
            self.logger.info(f"Updated {model_name} model path: {model_path}")
        else:
            self.logger.warning(f"Model '{model_name}' not found in configuration")
    
    def enable_model(self, model_name: str, enabled: bool = True) -> None:
        """Enable or disable specific model"""
        if model_name in self.config.models:
            self.config.models[model_name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Model '{model_name}' {status}")
        else:
            self.logger.warning(f"Model '{model_name}' not found in configuration")
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration"""
        return self.config.processing
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration"""
        return self.config.output
    
    def create_runtime_config(self) -> Dict[str, Any]:
        """Create runtime configuration dictionary for inference engine"""
        return {
            'models': {name: asdict(config) for name, config in self.config.models.items()},
            'processing': asdict(self.config.processing),
            'output': asdict(self.config.output),
            'phase_detection': asdict(self.config.phase_detection),
            'instrument_tracking': asdict(self.config.instrument_tracking),
            'event_detection': asdict(self.config.event_detection),
            'motion_analysis': asdict(self.config.motion_analysis)
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check model paths
        for model_name, model_config in self.config.models.items():
            if model_config.enabled and model_config.model_path:
                if not Path(model_config.model_path).exists():
                    issues.append(f"Model file not found: {model_config.model_path}")
        
        # Check output directory
        try:
            Path(self.config.output.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        # Check processing parameters
        if self.config.processing.target_fps <= 0:
            issues.append("Target FPS must be positive")
        
        if self.config.processing.max_workers <= 0:
            issues.append("Max workers must be positive")
        
        # Check thresholds
        for model_name in ['phase_detection', 'instrument_tracking', 'event_detection']:
            model_config = self.config.models.get(model_name)
            if model_config and not 0 <= model_config.confidence_threshold <= 1:
                issues.append(f"{model_name} confidence threshold must be between 0 and 1")
        
        return issues
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self.config = SystemConfiguration()
        self.logger.info("Configuration reset to defaults")
    
    def export_template(self, template_path: str) -> None:
        """Export configuration template with comments"""
        template_config = SystemConfiguration()
        
        # Add comments to template
        template_dict = asdict(template_config)
        
        # Save with comments
        template_path = Path(template_path)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_path, 'w') as f:
            f.write("# Surgical AI System Configuration Template\n")
            f.write("# This file contains all configurable parameters for the surgical video analysis system\n\n")
            yaml.dump(template_dict, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Configuration template exported to: {template_path}")

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None or config_path:
        _config_manager = ConfigurationManager(config_path)
    return _config_manager

def get_system_config() -> SystemConfiguration:
    """Get current system configuration"""
    return get_config_manager().config

if __name__ == "__main__":
    # Demo configuration management
    config_manager = ConfigurationManager()
    
    # Validate configuration
    issues = config_manager.validate_configuration()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Export template
    config_manager.export_template("config_template.yaml")
    
    print(f"Configuration file: {config_manager.config_path}")
    print(f"Version: {config_manager.config.version}")
    print(f"Models enabled: {[name for name, config in config_manager.config.models.items() if config.enabled]}")