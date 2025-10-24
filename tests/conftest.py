"""
Test configuration and fixtures for the Twitter Sentiment Analysis project.
Author: Lakshya Khetan
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
from unittest.mock import Mock

# Test data fixtures
@pytest.fixture
def sample_tweets():
    """Sample tweet data for testing."""
    return [
        "Modi is doing great work for the country! #BJP",
        "Not happy with current govt policies",
        "Congress has better vision for future https://example.com",
        "Election 2024 will be interesting #Democracy",
        "RT @user: This is a retweet with emoji 😊"
    ]

@pytest.fixture
def sample_tweet_dataframe():
    """Sample tweet DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            "Modi is doing great work for the country! #BJP",
            "Not happy with current govt policies",
            "Congress has better vision for future",
            "Election 2024 will be interesting #Democracy",
            "This is a neutral statement about politics"
        ],
        'created_at': pd.date_range('2023-01-01', periods=5),
        'user': ['user1', 'user2', 'user3', 'user4', 'user5']
    })

@pytest.fixture
def mock_twitter_api():
    """Mock Twitter API responses."""
    mock_api = Mock()
    
    # Mock tweet object
    mock_status = Mock()
    mock_status.id = 123456789
    mock_status.text = "Sample tweet text for testing"
    mock_status.full_text = "Sample tweet text for testing"
    mock_status.created_at = "2023-01-01T00:00:00Z"
    mock_status.lang = 'en'
    mock_status.favorite_count = 10
    mock_status.retweet_count = 5
    mock_status.user.screen_name = "test_user"
    mock_status.entities = {
        'hashtags': [{'text': 'test'}],
        'user_mentions': [{'screen_name': 'mentioned_user'}]
    }
    mock_status.possibly_sensitive = False
    mock_status.user.location = "Test Location"
    mock_status.place = None
    
    # Mock API search response
    mock_api.search_tweets.return_value = [mock_status]
    mock_api.verify_credentials.return_value = True
    
    return mock_api

@pytest.fixture
def test_config():
    """Test configuration data."""
    return {
        'twitter': {
            'consumer_key': 'test_key',
            'consumer_secret': 'test_secret',
            'access_token': 'test_token',
            'access_token_secret': 'test_token_secret'
        },
        'preprocessing': {
            'text_cleaning': {
                'remove_urls': True,
                'remove_mentions': True,
                'remove_rt': True,
                'convert_lowercase': True,
                'remove_stopwords': True
            },
            'tokenization': {
                'max_features': 1000,
                'max_sequence_length': 100
            }
        },
        'models': {
            'lstm': {
                'embedding_dim': 64,
                'lstm_units': 32,
                'training': {
                    'batch_size': 16,
                    'epochs': 2,
                    'learning_rate': 0.001
                }
            }
        }
    }

@pytest.fixture
def temp_config_file(test_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        return f.name

@pytest.fixture
def sample_model_data():
    """Sample data for model training/testing."""
    np.random.seed(42)
    X = np.random.random((100, 50))  # 100 samples, 50 features
    y = np.random.randint(0, 2, 100)  # Binary labels
    
    return X, y

@pytest.fixture
def mock_trained_model():
    """Mock trained model for testing predictions."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8]])
    return mock_model

@pytest.fixture
def sample_predictions():
    """Sample prediction results."""
    return [
        {
            'original_text': 'Great work by the government!',
            'clean_text': 'great work government',
            'prediction': 1,
            'probability': 0.8,
            'sentiment': 'positive'
        },
        {
            'original_text': 'Not happy with current policies',
            'clean_text': 'happy current policies',
            'prediction': 0,
            'probability': 0.3,
            'sentiment': 'negative'
        }
    ]

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

# Test data paths
@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("TWITTER_CONSUMER_KEY", "test_key")
    monkeypatch.setenv("TWITTER_CONSUMER_SECRET", "test_secret")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "test_token")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN_SECRET", "test_token_secret")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

# Pytest configuration
def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )