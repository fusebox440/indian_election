"""
Logging Utilities
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Provides centralized logging configuration for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from .config import config

class TwitterSentimentLogger:
    """Centralized logger for the Twitter Sentiment Analysis project."""
    
    def __init__(self, name: str = "twitter_sentiment"):
        """
        Initialize logger with configuration.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with configuration from config file."""
        log_config = config.get('logging', {})
        
        # Set log level
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        log_file = log_config.get('file', 'logs/twitter_sentiment.log')
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger

class ExperimentLogger:
    """Logger for tracking machine learning experiments."""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs/experiments"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.experiment_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': config.config,
            'stages': []
        }
    
    def log_stage(self, stage_name: str, data: dict):
        """
        Log a stage of the experiment.
        
        Args:
            stage_name: Name of the stage
            data: Data to log for this stage
        """
        stage_data = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        self.experiment_data['stages'].append(stage_data)
        self._save_experiment()
    
    def log_model_performance(self, model_name: str, metrics: dict, additional_data: Optional[dict] = None):
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            additional_data: Additional data to log
        """
        performance_data = {
            'model_name': model_name,
            'metrics': metrics
        }
        
        if additional_data:
            performance_data.update(additional_data)
        
        self.log_stage('model_evaluation', performance_data)
    
    def log_data_processing(self, stage: str, input_size: int, output_size: int, parameters: Optional[dict] = None):
        """
        Log data processing stage.
        
        Args:
            stage: Processing stage name
            input_size: Size of input data
            output_size: Size of output data
            parameters: Processing parameters
        """
        processing_data = {
            'processing_stage': stage,
            'input_size': input_size,
            'output_size': output_size
        }
        
        if parameters:
            processing_data['parameters'] = parameters
        
        self.log_stage('data_processing', processing_data)
    
    def finalize_experiment(self, status: str = 'completed', final_results: Optional[dict] = None):
        """
        Finalize the experiment log.
        
        Args:
            status: Final status of the experiment
            final_results: Final results to log
        """
        self.experiment_data['end_time'] = datetime.now().isoformat()
        self.experiment_data['status'] = status
        
        if final_results:
            self.experiment_data['final_results'] = final_results
        
        self._save_experiment()
    
    def _save_experiment(self):
        """Save experiment data to file."""
        with open(self.experiment_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)

# Global logger instance
logger = TwitterSentimentLogger().get_logger()

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup global logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path
    """
    global logger
    
    # Update config if provided
    if log_file:
        config.set('logging.file', log_file)
    
    config.set('logging.level', log_level)
    
    # Recreate logger with new config
    logger = TwitterSentimentLogger().get_logger()

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (uses global logger if None)
        
    Returns:
        Logger instance
    """
    if name is None:
        return logger
    else:
        return TwitterSentimentLogger(name).get_logger()