"""
Unit tests for utility modules.
Author: Lakshya Khetan
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.utils.visualization import create_sentiment_charts


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        config_manager = ConfigManager(temp_config_file)
        config = config_manager.get_config()
        
        assert config is not None
        assert 'twitter' in config
        assert 'preprocessing' in config
        assert 'models' in config
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent_config.yaml")
    
    def test_get_nested_config_value(self, temp_config_file):
        """Test getting nested configuration values."""
        config_manager = ConfigManager(temp_config_file)
        
        # Test getting nested values
        embedding_dim = config_manager.get('models.lstm.embedding_dim')
        assert embedding_dim == 64
        
        batch_size = config_manager.get('models.lstm.training.batch_size')
        assert batch_size == 16
    
    def test_get_config_value_with_default(self, temp_config_file):
        """Test getting config value with default fallback."""
        config_manager = ConfigManager(temp_config_file)
        
        # Test existing value
        existing_value = config_manager.get('models.lstm.embedding_dim', default=100)
        assert existing_value == 64
        
        # Test non-existing value with default
        non_existing_value = config_manager.get('nonexistent.key', default='default_value')
        assert non_existing_value == 'default_value'
    
    def test_get_config_value_not_found(self, temp_config_file):
        """Test getting non-existent config value without default."""
        config_manager = ConfigManager(temp_config_file)
        
        with pytest.raises(KeyError):
            config_manager.get('nonexistent.key')
    
    def test_environment_variable_substitution(self, monkeypatch):
        """Test environment variable substitution in config."""
        # Set environment variable
        monkeypatch.setenv("TEST_API_KEY", "secret_key_123")
        
        # Create config with environment variable
        config_data = {
            'api': {
                'key': '${TEST_API_KEY}',
                'url': 'https://api.example.com'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            api_key = config_manager.get('api.key')
            assert api_key == 'secret_key_123'
        finally:
            os.unlink(config_file)
    
    def test_update_config_value(self, temp_config_file):
        """Test updating configuration values."""
        config_manager = ConfigManager(temp_config_file)
        
        # Update existing value
        config_manager.update('models.lstm.embedding_dim', 128)
        updated_value = config_manager.get('models.lstm.embedding_dim')
        assert updated_value == 128
        
        # Update non-existing nested value
        config_manager.update('new.nested.key', 'new_value')
        new_value = config_manager.get('new.nested.key')
        assert new_value == 'new_value'
    
    def test_save_config(self, temp_config_file, temp_directory):
        """Test saving configuration to file."""
        config_manager = ConfigManager(temp_config_file)
        
        # Update a value
        config_manager.update('models.lstm.embedding_dim', 256)
        
        # Save to new file
        new_config_file = temp_directory / "new_config.yaml"
        success = config_manager.save_config(str(new_config_file))
        
        assert success is True
        assert new_config_file.exists()
        
        # Verify saved config
        new_config_manager = ConfigManager(str(new_config_file))
        embedding_dim = new_config_manager.get('models.lstm.embedding_dim')
        assert embedding_dim == 256
    
    def test_validate_config_structure(self, temp_config_file):
        """Test configuration structure validation."""
        config_manager = ConfigManager(temp_config_file)
        
        # Test with valid structure
        is_valid = config_manager.validate_config(['twitter', 'preprocessing', 'models'])
        assert is_valid is True
        
        # Test with invalid structure
        is_valid = config_manager.validate_config(['twitter', 'nonexistent_section'])
        assert is_valid is False


class TestLogger:
    """Test cases for logger utility."""
    
    def test_setup_logger_default(self):
        """Test setting up logger with default configuration."""
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_level(self):
        """Test setting up logger with specific level."""
        logger = setup_logger("test_logger", level="DEBUG")
        
        assert logger.level <= 10  # DEBUG level
    
    def test_setup_logger_with_file(self, temp_directory):
        """Test setting up logger with file handler."""
        log_file = temp_directory / "test.log"
        logger = setup_logger("test_logger", log_file=str(log_file))
        
        # Test logging
        logger.info("Test message")
        
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
            assert "Test message" in content
    
    def test_logger_formatting(self, temp_directory):
        """Test logger message formatting."""
        log_file = temp_directory / "format_test.log"
        logger = setup_logger("format_test", log_file=str(log_file))
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        with open(log_file) as f:
            content = f.read()
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content
            assert "format_test" in content
    
    @pytest.mark.parametrize("level,message", [
        ("DEBUG", "Debug message"),
        ("INFO", "Info message"),
        ("WARNING", "Warning message"),
        ("ERROR", "Error message"),
        ("CRITICAL", "Critical message")
    ])
    def test_logger_levels(self, temp_directory, level, message):
        """Parametrized test for different log levels."""
        log_file = temp_directory / f"{level.lower()}_test.log"
        logger = setup_logger(f"{level.lower()}_test", level=level, log_file=str(log_file))
        
        # Log message at the specified level
        getattr(logger, level.lower())(message)
        
        with open(log_file) as f:
            content = f.read()
            assert message in content


class TestVisualization:
    """Test cases for visualization utilities."""
    
    def test_create_sentiment_charts_basic(self, sample_predictions, temp_directory):
        """Test creating basic sentiment charts."""
        output_path = temp_directory / "sentiment_chart.png"
        
        success = create_sentiment_charts(
            sample_predictions,
            str(output_path)
        )
        
        assert success is True
        assert output_path.exists()
    
    def test_create_sentiment_charts_with_dataframe(self, temp_directory):
        """Test creating charts from DataFrame."""
        import pandas as pd
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'sentiment': ['positive', 'negative', 'positive', 'neutral', 'negative'],
            'probability': [0.8, 0.3, 0.9, 0.5, 0.2],
            'text': ['Great!', 'Bad', 'Excellent', 'Okay', 'Terrible']
        })
        
        output_path = temp_directory / "df_chart.png"
        
        success = create_sentiment_charts(
            df.to_dict('records'),
            str(output_path)
        )
        
        assert success is True
        assert output_path.exists()
    
    def test_create_sentiment_charts_empty_data(self, temp_directory):
        """Test creating charts with empty data."""
        output_path = temp_directory / "empty_chart.png"
        
        with pytest.raises(ValueError):
            create_sentiment_charts([], str(output_path))
    
    def test_create_sentiment_charts_invalid_data(self, temp_directory):
        """Test creating charts with invalid data structure."""
        invalid_data = [
            {'text': 'No sentiment field'},
            {'sentiment': 'positive'}  # No other required fields
        ]
        
        output_path = temp_directory / "invalid_chart.png"
        
        with pytest.raises(KeyError):
            create_sentiment_charts(invalid_data, str(output_path))
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_sentiment_distribution_chart(self, mock_show, mock_savefig, sample_predictions):
        """Test creating sentiment distribution chart."""
        from src.utils.visualization import create_sentiment_distribution_chart
        
        create_sentiment_distribution_chart(sample_predictions)
        
        # Verify matplotlib functions were called
        mock_savefig.assert_not_called()  # Since no save path provided
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_confidence_histogram(self, mock_savefig, sample_predictions, temp_directory):
        """Test creating confidence histogram."""
        from src.utils.visualization import create_confidence_histogram
        
        output_path = temp_directory / "confidence_hist.png"
        
        create_confidence_histogram(sample_predictions, str(output_path))
        
        mock_savefig.assert_called_once()
    
    def test_visualization_with_different_data_sizes(self, temp_directory):
        """Test visualization with different data sizes."""
        # Test with small dataset
        small_data = [
            {'sentiment': 'positive', 'probability': 0.8, 'text': 'Good'},
            {'sentiment': 'negative', 'probability': 0.3, 'text': 'Bad'}
        ]
        
        small_output = temp_directory / "small_chart.png"
        success = create_sentiment_charts(small_data, str(small_output))
        assert success is True
        
        # Test with larger dataset
        large_data = [
            {'sentiment': 'positive' if i % 2 == 0 else 'negative', 
             'probability': 0.5 + (i % 10) * 0.05, 
             'text': f'Text {i}'}
            for i in range(100)
        ]
        
        large_output = temp_directory / "large_chart.png"
        success = create_sentiment_charts(large_data, str(large_output))
        assert success is True
    
    @pytest.mark.parametrize("chart_type", [
        "distribution",
        "confidence",
        "timeline"
    ])
    def test_different_chart_types(self, sample_predictions, temp_directory, chart_type):
        """Parametrized test for different chart types."""
        output_path = temp_directory / f"{chart_type}_chart.png"
        
        success = create_sentiment_charts(
            sample_predictions,
            str(output_path),
            chart_type=chart_type
        )
        
        assert success is True
        assert output_path.exists()


class TestUtilityHelpers:
    """Test cases for utility helper functions."""
    
    def test_ensure_directory_exists(self, temp_directory):
        """Test directory creation utility."""
        from src.utils.helpers import ensure_directory_exists
        
        new_dir = temp_directory / "new_directory" / "nested"
        ensure_directory_exists(str(new_dir))
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_safe_file_write(self, temp_directory):
        """Test safe file writing utility."""
        from src.utils.helpers import safe_file_write
        
        file_path = temp_directory / "safe_write_test.txt"
        content = "Test content for safe writing"
        
        success = safe_file_write(str(file_path), content)
        
        assert success is True
        assert file_path.exists()
        
        with open(file_path) as f:
            assert f.read() == content
    
    def test_validate_file_extension(self):
        """Test file extension validation utility."""
        from src.utils.helpers import validate_file_extension
        
        assert validate_file_extension("model.h5", [".h5", ".keras"]) is True
        assert validate_file_extension("data.csv", [".csv", ".json"]) is True
        assert validate_file_extension("config.yaml", [".yaml", ".yml"]) is True
        
        assert validate_file_extension("model.txt", [".h5", ".keras"]) is False
        assert validate_file_extension("data.xlsx", [".csv", ".json"]) is False
    
    def test_format_file_size(self):
        """Test file size formatting utility."""
        from src.utils.helpers import format_file_size
        
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(512) == "512 B"
    
    def test_get_timestamp(self):
        """Test timestamp generation utility."""
        from src.utils.helpers import get_timestamp
        
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        
        # Test different formats
        timestamp_date = get_timestamp(format='%Y-%m-%d')
        assert '-' in timestamp_date
        
        timestamp_time = get_timestamp(format='%H:%M:%S')
        assert ':' in timestamp_time
    
    def test_calculate_metrics(self, sample_predictions):
        """Test metrics calculation utility."""
        from src.utils.helpers import calculate_metrics
        
        # Convert predictions to required format
        y_true = [1 if pred['sentiment'] == 'positive' else 0 
                 for pred in sample_predictions]
        y_pred = [pred['prediction'] for pred in sample_predictions]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Verify metric ranges
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1
    
    @pytest.mark.integration
    def test_utility_integration(self, temp_directory, sample_predictions):
        """Integration test for utility functions working together."""
        from src.utils.helpers import ensure_directory_exists, safe_file_write, get_timestamp
        from src.utils.config import ConfigManager
        from src.utils.logger import setup_logger
        
        # Create nested directory structure
        output_dir = temp_directory / "integration_test" / get_timestamp(format='%Y%m%d')
        ensure_directory_exists(str(output_dir))
        
        # Setup logger
        log_file = output_dir / "integration.log"
        logger = setup_logger("integration_test", log_file=str(log_file))
        
        # Create and save config
        config_data = {
            'test': {
                'timestamp': get_timestamp(),
                'output_dir': str(output_dir)
            }
        }
        
        config_file = output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config and log information
        config_manager = ConfigManager(str(config_file))
        timestamp = config_manager.get('test.timestamp')
        
        logger.info(f"Integration test started at {timestamp}")
        logger.info(f"Output directory: {output_dir}")
        
        # Save predictions data
        predictions_content = str(sample_predictions)
        predictions_file = output_dir / "predictions.txt"
        success = safe_file_write(str(predictions_file), predictions_content)
        
        logger.info(f"Predictions saved: {success}")
        
        # Verify everything worked
        assert output_dir.exists()
        assert log_file.exists()
        assert config_file.exists()
        assert predictions_file.exists()
        assert success is True
        
        # Verify log content
        with open(log_file) as f:
            log_content = f.read()
            assert "Integration test started" in log_content
            assert "Predictions saved: True" in log_content