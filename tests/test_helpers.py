"""
Utility functions for testing.
Author: Lakshya Khetan
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from unittest.mock import Mock


def create_sample_tweets(count: int = 10) -> List[str]:
    """Create sample tweet texts for testing."""
    positive_tweets = [
        "Great work by Modi government! #BJP2024",
        "Excellent policies implemented by current govt",
        "Modi ji is doing fantastic job for the country",
        "BJP has transformed India's economy",
        "Proud of our Prime Minister's leadership"
    ]
    
    negative_tweets = [
        "Not happy with current government policies",
        "Congress had better vision for the country",
        "Unemployment rate is increasing under BJP",
        "Government failed to deliver on promises",  
        "Opposition parties are raising valid concerns"
    ]
    
    neutral_tweets = [
        "Election results will be announced soon",
        "Political parties are campaigning actively",
        "Voters are preparing for upcoming elections",
        "Democracy allows freedom of choice",
        "Politics affects everyone's daily life"
    ]
    
    all_tweets = positive_tweets + negative_tweets + neutral_tweets
    
    # Return requested number of tweets, cycling through if needed
    return [all_tweets[i % len(all_tweets)] for i in range(count)]


def create_sample_dataframe(count: int = 10) -> pd.DataFrame:
    """Create sample tweet DataFrame for testing."""
    tweets = create_sample_tweets(count)
    
    return pd.DataFrame({
        'id': range(1, count + 1),
        'text': tweets,
        'created_at': pd.date_range('2023-01-01', periods=count),
        'user': [f'user_{i}' for i in range(1, count + 1)],
        'lang': ['en'] * count,
        'retweet_count': np.random.randint(0, 100, count),
        'like_count': np.random.randint(0, 500, count)
    })


def create_mock_twitter_api() -> Mock:
    """Create a mock Twitter API for testing."""
    mock_api = Mock()
    
    # Create mock tweet objects
    mock_tweets = []
    sample_texts = create_sample_tweets(5)
    
    for i, text in enumerate(sample_texts):
        mock_tweet = Mock()
        mock_tweet.id = 1000 + i
        mock_tweet.text = text
        mock_tweet.full_text = text
        mock_tweet.author_id = 2000 + i
        mock_tweet.created_at = "2023-01-01T00:00:00Z"
        mock_tweet.lang = 'en'
        mock_tweet.public_metrics = {
            'like_count': i * 10,
            'retweet_count': i * 5,
            'reply_count': i * 2,
            'quote_count': i
        }
        mock_tweet.possibly_sensitive = False
        mock_tweets.append(mock_tweet)
    
    # Mock API responses
    mock_response = Mock()
    mock_response.data = mock_tweets
    mock_api.search_recent_tweets.return_value = mock_response
    mock_api.verify_credentials.return_value = True
    
    return mock_api


def create_test_config() -> Dict[str, Any]:
    """Create test configuration dictionary."""
    return {
        'twitter': {
            'consumer_key': 'test_consumer_key',
            'consumer_secret': 'test_consumer_secret',
            'access_token': 'test_access_token',
            'access_token_secret': 'test_access_token_secret'
        },
        'data_collection': {
            'keywords': ['modi', 'bjp', 'congress', 'election'],
            'count': 100,
            'lang': 'en',
            'result_type': 'recent'
        },
        'preprocessing': {
            'text_cleaning': {
                'remove_urls': True,
                'remove_mentions': True,
                'remove_rt': True,
                'remove_hashtags': False,
                'convert_lowercase': True,
                'remove_punctuation': True,
                'remove_stopwords': True,
                'remove_numbers': True
            },
            'tokenization': {
                'max_features': 10000,
                'max_sequence_length': 100,
                'oov_token': '<OOV>'
            }
        },
        'models': {
            'lstm': {
                'embedding_dim': 128,
                'lstm_units': 64,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'training': {
                    'batch_size': 32,
                    'epochs': 10,
                    'learning_rate': 0.001,
                    'validation_split': 0.2
                }
            }
        },
        'prediction': {
            'confidence_threshold': 0.6,
            'batch_size': 32
        },
        'output': {
            'model_path': 'models/sentiment_model.h5',
            'tokenizer_path': 'models/tokenizer.pickle',
            'predictions_path': 'results/predictions.csv',
            'logs_path': 'logs/sentiment_analysis.log'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


def save_test_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save test configuration to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_mock_model_predictions(count: int = 10) -> np.ndarray:
    """Create mock model predictions for testing."""
    # Generate random probabilities between 0 and 1
    np.random.seed(42)  # For reproducibility
    return np.random.random((count, 1))


def create_sample_training_data(count: int = 100) -> tuple:
    """Create sample training data (X, y) for model testing."""
    np.random.seed(42)
    
    # Create random sequence data
    X = np.random.randint(1, 1000, size=(count, 50))  # 50 is sequence length
    
    # Create binary labels
    y = np.random.randint(0, 2, size=count)
    
    return X, y


def create_prediction_results(count: int = 10) -> List[Dict[str, Any]]:
    """Create sample prediction results for testing."""
    tweets = create_sample_tweets(count)
    np.random.seed(42)
    
    results = []
    for i, tweet in enumerate(tweets):
        # Determine sentiment based on content for realistic results
        if any(word in tweet.lower() for word in ['great', 'excellent', 'fantastic', 'good']):
            sentiment = 'positive'
            prob = 0.7 + np.random.random() * 0.3  # 0.7 to 1.0
            prediction = 1
        elif any(word in tweet.lower() for word in ['not happy', 'failed', 'concerns', 'bad']):
            sentiment = 'negative'
            prob = np.random.random() * 0.4  # 0.0 to 0.4
            prediction = 0
        else:
            sentiment = 'neutral'
            prob = 0.4 + np.random.random() * 0.2  # 0.4 to 0.6
            prediction = 1 if prob > 0.5 else 0
        
        results.append({
            'original_text': tweet,
            'clean_text': tweet.lower().replace('#', '').replace('@', ''),
            'prediction': prediction,
            'probability': round(prob, 3),
            'sentiment': sentiment
        })
    
    return results


def create_temp_files(temp_dir: Path, files: Dict[str, str]) -> Dict[str, Path]:
    """Create temporary files with specified content."""
    created_files = {}
    
    for filename, content in files.items():
        file_path = temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content based on file extension
        if filename.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(json.loads(content), f, indent=2)
        elif filename.endswith(('.yaml', '.yml')):
            with open(file_path, 'w') as f:
                yaml.dump(yaml.safe_load(content), f)
        else:
            with open(file_path, 'w') as f:
                f.write(content)
        
        created_files[filename] = file_path
    
    return created_files


def assert_file_exists_and_not_empty(file_path: Union[str, Path]) -> None:
    """Assert that file exists and is not empty."""
    path = Path(file_path)
    assert path.exists(), f"File {file_path} does not exist"
    assert path.stat().st_size > 0, f"File {file_path} is empty"


def assert_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Assert that DataFrame has required structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert not df.empty, "DataFrame is empty"
    
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' missing from DataFrame"
    
    # Check for no null values in required columns
    for col in required_columns:
        assert not df[col].isna().any(), f"Column '{col}' has null values"


def assert_prediction_results_structure(results: List[Dict[str, Any]]) -> None:
    """Assert that prediction results have correct structure."""
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Results list is empty"
    
    required_keys = ['original_text', 'clean_text', 'prediction', 'probability', 'sentiment']
    
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} is not a dictionary"
        
        for key in required_keys:
            assert key in result, f"Required key '{key}' missing from result {i}"
        
        # Check data types
        assert isinstance(result['original_text'], str), f"Result {i}: original_text should be string"
        assert isinstance(result['clean_text'], str), f"Result {i}: clean_text should be string"
        assert result['prediction'] in [0, 1], f"Result {i}: prediction should be 0 or 1"
        assert 0 <= result['probability'] <= 1, f"Result {i}: probability should be between 0 and 1"
        assert result['sentiment'] in ['positive', 'negative', 'neutral'], f"Result {i}: invalid sentiment"


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any], ignore_keys: List[str] = None) -> bool:
    """Compare two configuration dictionaries, optionally ignoring certain keys."""
    if ignore_keys is None:
        ignore_keys = []
    
    def remove_ignored_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove ignored keys from dictionary."""
        if not isinstance(d, dict):
            return d
        
        return {k: remove_ignored_keys(v) for k, v in d.items() if k not in ignore_keys}
    
    clean_config1 = remove_ignored_keys(config1)
    clean_config2 = remove_ignored_keys(config2)
    
    return clean_config1 == clean_config2


def create_performance_test_data(size: str = 'small') -> Dict[str, Any]:
    """Create test data of different sizes for performance testing."""
    sizes = {
        'small': 100,
        'medium': 1000,
        'large': 10000
    }
    
    count = sizes.get(size, 100)
    
    return {
        'tweets': create_sample_tweets(count),
        'dataframe': create_sample_dataframe(count),
        'training_data': create_sample_training_data(count),
        'predictions': create_prediction_results(min(count, 1000))  # Limit predictions for memory
    }


def measure_execution_time(func, *args, **kwargs) -> tuple:
    """Measure execution time of a function."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return result, execution_time


def setup_test_environment(temp_dir: Path) -> Dict[str, Path]:
    """Set up a complete test environment with directories and files."""
    # Create directory structure
    directories = [
        'data',
        'models', 
        'results',
        'logs',
        'config'
    ]
    
    created_paths = {}
    
    for dir_name in directories:
        dir_path = temp_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        created_paths[dir_name] = dir_path
    
    # Create test configuration file
    config = create_test_config()
    config_file = created_paths['config'] / 'test_config.yaml'
    save_test_config(config, config_file)
    created_paths['config_file'] = config_file
    
    # Create sample data file
    sample_df = create_sample_dataframe(50)
    data_file = created_paths['data'] / 'sample_tweets.csv'
    sample_df.to_csv(data_file, index=False)
    created_paths['data_file'] = data_file
    
    return created_paths


def cleanup_test_environment(paths: Dict[str, Path]) -> None:
    """Clean up test environment by removing created files and directories."""
    import shutil
    
    for path in paths.values():
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def validate_model_architecture(model, expected_layers: List[str]) -> bool:
    """Validate that a model has expected layer types."""
    if not hasattr(model, 'layers'):
        return False
    
    layer_types = [type(layer).__name__ for layer in model.layers]
    
    for expected_layer in expected_layers:
        if expected_layer not in layer_types:
            return False
    
    return True


def generate_test_report(test_results: Dict[str, Any], output_file: Path) -> None:
    """Generate a test report with results and statistics."""
    report = {
        'test_summary': {
            'total_tests': len(test_results),
            'passed_tests': sum(1 for result in test_results.values() if result.get('passed', False)),
            'failed_tests': sum(1 for result in test_results.values() if not result.get('passed', False))
        },
        'test_details': test_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


# Test data generators for specific components
class TestDataGenerator:
    """Class to generate test data for specific components."""
    
    @staticmethod
    def twitter_api_response(count: int = 5) -> Dict[str, Any]:
        """Generate mock Twitter API response."""
        tweets = create_sample_tweets(count)
        
        return {
            'data': [
                {
                    'id': str(1000 + i),
                    'text': tweet,
                    'author_id': str(2000 + i),
                    'created_at': '2023-01-01T00:00:00.000Z',
                    'lang': 'en',
                    'public_metrics': {
                        'retweet_count': i * 2,
                        'like_count': i * 10,
                        'reply_count': i,
                        'quote_count': 0
                    }
                }
                for i, tweet in enumerate(tweets)
            ],
            'meta': {
                'newest_id': str(1000 + count - 1),
                'oldest_id': '1000',
                'result_count': count
            }
        }
    
    @staticmethod
    def preprocessed_sequences(count: int = 10, seq_length: int = 50) -> np.ndarray:
        """Generate preprocessed sequences for model input."""
        np.random.seed(42)
        return np.random.randint(1, 1000, size=(count, seq_length))
    
    @staticmethod
    def model_training_history() -> Dict[str, List[float]]:
        """Generate mock model training history."""
        epochs = 5
        return {
            'loss': [0.8 - i * 0.1 for i in range(epochs)],
            'accuracy': [0.6 + i * 0.05 for i in range(epochs)],
            'val_loss': [0.9 - i * 0.08 for i in range(epochs)],
            'val_accuracy': [0.55 + i * 0.06 for i in range(epochs)]
        }