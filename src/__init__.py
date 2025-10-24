"""
Twitter Sentiment Analysis for Indian Elections
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

A comprehensive machine learning pipeline for analyzing Twitter sentiment 
to predict election outcomes for Indian elections.
"""

__version__ = "2.0.0"
__author__ = "Lakshya Khetan"
__email__ = "lakshyaketan00@gmail.com"
__license__ = "MIT"

# Import main classes for easy access
from .data.collector import TwitterDataCollector
from .data.preprocessor import TextPreprocessor
from .models.sentiment_models import GloVeModel, LSTMModel, ModelTrainer
from .models.predictor import SentimentPredictor, ElectionPredictor
from .utils.config import config
from .utils.logger import get_logger, setup_logging
from .utils.visualization import SentimentVisualizer, ModelVisualizer

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # Main classes
    "TwitterDataCollector",
    "TextPreprocessor",
    "GloVeModel",
    "LSTMModel", 
    "ModelTrainer",
    "SentimentPredictor",
    "ElectionPredictor",
    
    # Utilities
    "config",
    "get_logger",
    "setup_logging",
    "SentimentVisualizer",
    "ModelVisualizer"
]