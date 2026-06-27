# src/utils/logging_utils.py
"""
Centralized logging utilities for consistent experiment tracking
Provides structured logging with file output, console formatting, and experiment metadata
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import colorlog

def setup_logger(
    name: str = "agentic_self_rag",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup structured logger with color formatting
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler with colors
    if console:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        console_format = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(blue)s[%(name)s]%(reset)s %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Enhanced logger for experiment tracking with metadata
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file)
        )
        
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'log_file': str(log_file)
        }
        
        self.logger.info(f"Experiment started: {experiment_name}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        self.metadata['config'] = config
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if 'metrics' not in self.metadata:
            self.metadata['metrics'] = []
        
        metric_entry = {'name': name, 'value': value}
        if step is not None:
            metric_entry['step'] = step
        
        self.metadata['metrics'].append(metric_entry)
        self.logger.info(f"Metric - {name}: {value}" + (f" (step {step})" if step else ""))
    
    def log_result(self, result: Dict[str, Any]):
        """Log experiment results"""
        self.metadata['results'] = result
        self.logger.info(f"Results: {json.dumps(result, indent=2)}")
    
    def save_metadata(self):
        """Save experiment metadata to JSON"""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        metadata_file = self.output_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to: {metadata_file}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Exception occurred: {exc_val}", exc_info=True)
        self.save_metadata()
        self.logger.info(f"Experiment completed: {self.experiment_name}")


def log_system_info(logger: logging.Logger):
    """Log system and environment information"""
    import platform
    import psutil
    
    logger.info("="*70)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*70)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.info("PyTorch: Not installed")
    
    logger.info("="*70)