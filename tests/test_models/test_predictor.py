"""
Unit tests for the predictor module.
Author: Lakshya Khetan
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.models.predictor import SentimentPredictor


class TestSentimentPredictor:
    """Test cases for SentimentPredictor class."""
    
    def test_init_with_valid_config(self, test_config):
        """Test initialization with valid configuration."""
        predictor = SentimentPredictor(test_config)
        
        assert predictor.config == test_config
        assert predictor.logger is not None
        assert predictor.model is None
        assert predictor.preprocessor is None
    
    def test_load_model_and_preprocessor(self, test_config, temp_directory):
        """Test loading model and preprocessor."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor files
        model_path = temp_directory / "model.h5"
        preprocessor_path = temp_directory / "tokenizer.pickle"
        
        # Create mock files
        model_path.touch()
        preprocessor_path.touch()
        
        with patch('src.models.sentiment_models.LSTMSentimentModel') as mock_model_class:
            with patch('src.data.preprocessor.TextPreprocessor') as mock_preprocessor_class:
                mock_model = Mock()
                mock_model.load_model.return_value = True
                mock_model_class.return_value = mock_model
                
                mock_preprocessor = Mock()
                mock_preprocessor.load_tokenizer.return_value = True
                mock_preprocessor_class.return_value = mock_preprocessor
                
                success = predictor.load_model_and_preprocessor(
                    str(model_path),
                    str(preprocessor_path)
                )
                
                assert success is True
                assert predictor.model is not None
                assert predictor.preprocessor is not None
    
    def test_load_model_file_not_found(self, test_config):
        """Test loading non-existent model file."""
        predictor = SentimentPredictor(test_config)
        
        with pytest.raises(FileNotFoundError):
            predictor.load_model_and_preprocessor(
                "nonexistent_model.h5",
                "nonexistent_tokenizer.pickle"
            )
    
    def test_predict_single_text(self, test_config):
        """Test predicting sentiment for single text."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8]])
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.return_value = "clean text"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 3, 0, 0]])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        result = predictor.predict_text("Great work by the government!")
        
        assert isinstance(result, dict)
        assert 'original_text' in result
        assert 'clean_text' in result
        assert 'prediction' in result
        assert 'probability' in result
        assert 'sentiment' in result
        assert result['sentiment'] == 'positive'
    
    def test_predict_single_text_negative(self, test_config):
        """Test predicting negative sentiment."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2]])  # Low probability = negative
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.return_value = "clean text"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 3, 0, 0]])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        result = predictor.predict_text("Not happy with current policies")
        
        assert result['sentiment'] == 'negative'
        assert result['prediction'] == 0
    
    def test_predict_text_no_model(self, test_config):
        """Test predicting without loaded model."""
        predictor = SentimentPredictor(test_config)
        
        with pytest.raises(ValueError, match="Model and preprocessor not loaded"):
            predictor.predict_text("Test text")
    
    def test_predict_batch_texts(self, test_config, sample_tweets):
        """Test predicting sentiment for batch of texts."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8], [0.3], [0.7], [0.1], [0.9]])
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x[:10]}"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        mock_preprocessor.pad_sequences.return_value = np.array([
            [1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0], [9, 10, 0]
        ])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        results = predictor.predict_batch(sample_tweets)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_tweets)
        
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert 'original_text' in result
            assert 'sentiment' in result
            assert result['original_text'] == sample_tweets[i]
    
    def test_predict_dataframe(self, test_config, sample_tweet_dataframe):
        """Test predicting sentiment for DataFrame."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8], [0.3], [0.7], [0.1], [0.9]])
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x[:10]}"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        mock_preprocessor.pad_sequences.return_value = np.array([
            [1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0], [9, 10, 0]
        ])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        result_df = predictor.predict_dataframe(
            sample_tweet_dataframe,
            text_column='text'
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_tweet_dataframe)
        assert 'clean_text' in result_df.columns
        assert 'prediction' in result_df.columns
        assert 'probability' in result_df.columns
        assert 'sentiment' in result_df.columns
    
    def test_predict_dataframe_invalid_column(self, test_config, sample_tweet_dataframe):
        """Test predicting DataFrame with invalid text column."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        predictor.model = Mock()
        predictor.preprocessor = Mock()
        
        with pytest.raises(KeyError):
            predictor.predict_dataframe(
                sample_tweet_dataframe,
                text_column='nonexistent_column'
            )
    
    def test_predict_with_confidence_threshold(self, test_config):
        """Test predictions with confidence threshold."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.55]])  # Borderline confidence
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.return_value = "clean text"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 3, 0, 0]])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        # Test with different confidence thresholds
        result_low_threshold = predictor.predict_text(
            "Neutral statement", 
            confidence_threshold=0.5
        )
        assert result_low_threshold['sentiment'] == 'positive'
        
        result_high_threshold = predictor.predict_text(
            "Neutral statement", 
            confidence_threshold=0.7
        )
        assert result_high_threshold['sentiment'] == 'neutral'
    
    def test_get_prediction_stats(self, test_config, sample_predictions):
        """Test getting prediction statistics."""
        predictor = SentimentPredictor(test_config)
        
        stats = predictor.get_prediction_stats(sample_predictions)
        
        assert isinstance(stats, dict)
        assert 'total_predictions' in stats
        assert 'positive_count' in stats
        assert 'negative_count' in stats
        assert 'neutral_count' in stats
        assert 'average_confidence' in stats
        assert 'sentiment_distribution' in stats
    
    def test_save_predictions_csv(self, test_config, sample_predictions, temp_directory):
        """Test saving predictions to CSV file."""
        predictor = SentimentPredictor(test_config)
        
        file_path = temp_directory / "predictions.csv"
        success = predictor.save_predictions(sample_predictions, str(file_path), format='csv')
        
        assert success is True
        assert file_path.exists()
        
        # Verify saved data
        saved_df = pd.read_csv(file_path)
        assert len(saved_df) == len(sample_predictions)
        assert 'original_text' in saved_df.columns
        assert 'sentiment' in saved_df.columns
    
    def test_save_predictions_json(self, test_config, sample_predictions, temp_directory):
        """Test saving predictions to JSON file."""
        predictor = SentimentPredictor(test_config)
        
        file_path = temp_directory / "predictions.json"
        success = predictor.save_predictions(sample_predictions, str(file_path), format='json')
        
        assert success is True
        assert file_path.exists()
        
        # Verify saved data
        import json
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == len(sample_predictions)
    
    def test_save_predictions_invalid_format(self, test_config, sample_predictions):
        """Test saving predictions with invalid format."""
        predictor = SentimentPredictor(test_config)
        
        with pytest.raises(ValueError):
            predictor.save_predictions(sample_predictions, "test.txt", format='invalid')
    
    def test_filter_predictions_by_sentiment(self, test_config, sample_predictions):
        """Test filtering predictions by sentiment."""
        predictor = SentimentPredictor(test_config)
        
        positive_predictions = predictor.filter_predictions_by_sentiment(
            sample_predictions, 
            'positive'
        )
        
        assert isinstance(positive_predictions, list)
        assert all(pred['sentiment'] == 'positive' for pred in positive_predictions)
    
    def test_filter_predictions_by_confidence(self, test_config, sample_predictions):
        """Test filtering predictions by confidence threshold."""
        predictor = SentimentPredictor(test_config)
        
        high_confidence_predictions = predictor.filter_predictions_by_confidence(
            sample_predictions,
            min_confidence=0.7
        )
        
        assert isinstance(high_confidence_predictions, list)
        assert all(pred['probability'] >= 0.7 for pred in high_confidence_predictions)
    
    def test_predict_streaming(self, test_config, sample_tweets):
        """Test streaming prediction functionality."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([[0.8]]), np.array([[0.3]]), np.array([[0.7]])
        ]
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x[:10]}"
        mock_preprocessor.texts_to_sequences.side_effect = [[[1, 2]], [[3, 4]], [[5, 6]]]
        mock_preprocessor.pad_sequences.side_effect = [
            np.array([[1, 2, 0]]), np.array([[3, 4, 0]]), np.array([[5, 6, 0]])
        ]
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        # Test streaming predictions
        results = []
        for text in sample_tweets[:3]:
            result = predictor.predict_text(text)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
    
    @pytest.mark.parametrize("probability,expected_sentiment", [
        (0.9, 'positive'),
        (0.7, 'positive'),
        (0.6, 'positive'),
        (0.4, 'negative'),
        (0.3, 'negative'),
        (0.1, 'negative')
    ])
    def test_sentiment_classification_thresholds(self, test_config, probability, expected_sentiment):
        """Parametrized test for sentiment classification thresholds."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[probability]])
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.return_value = "clean text"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 3, 0, 0]])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        result = predictor.predict_text("Test text")
        assert result['sentiment'] == expected_sentiment
    
    def test_batch_prediction_performance(self, test_config):
        """Test batch prediction performance optimization."""
        predictor = SentimentPredictor(test_config)
        
        # Create large batch of texts
        large_batch = ["Test text " + str(i) for i in range(100)]
        
        # Mock model and preprocessor for batch processing
        mock_model = Mock()
        mock_model.predict.return_value = np.random.random((100, 1))
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x}"
        mock_preprocessor.texts_to_sequences.return_value = [[i] for i in range(100)]
        mock_preprocessor.pad_sequences.return_value = np.random.randint(0, 1000, (100, 50))
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        results = predictor.predict_batch(large_batch)
        
        # Verify batch processing works efficiently
        assert len(results) == 100
        assert mock_model.predict.call_count == 1  # Should be called once for entire batch
    
    def test_prediction_error_handling(self, test_config):
        """Test error handling in predictions."""
        predictor = SentimentPredictor(test_config)
        
        # Mock model that raises an exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction error")
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text.return_value = "clean text"
        mock_preprocessor.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 3, 0, 0]])
        
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        
        with pytest.raises(Exception):
            predictor.predict_text("Test text")
    
    @pytest.mark.integration
    def test_integration_prediction_pipeline(self, test_config, temp_directory):
        """Integration test for complete prediction pipeline."""
        predictor = SentimentPredictor(test_config)
        
        # Mock the complete pipeline
        with patch('src.models.sentiment_models.LSTMSentimentModel') as mock_model_class:
            with patch('src.data.preprocessor.TextPreprocessor') as mock_preprocessor_class:
                # Setup mocks
                mock_model = Mock()
                mock_model.load_model.return_value = True
                mock_model.predict.return_value = np.array([[0.8], [0.3]])
                mock_model_class.return_value = mock_model
                
                mock_preprocessor = Mock()
                mock_preprocessor.load_tokenizer.return_value = True
                mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x[:10]}"
                mock_preprocessor.texts_to_sequences.return_value = [[1, 2], [3, 4]]
                mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 0], [3, 4, 0]])
                mock_preprocessor_class.return_value = mock_preprocessor
                
                # Create mock files
                model_path = temp_directory / "model.h5"
                preprocessor_path = temp_directory / "tokenizer.pickle"
                model_path.touch()
                preprocessor_path.touch()
                
                # Test complete pipeline
                success = predictor.load_model_and_preprocessor(
                    str(model_path),
                    str(preprocessor_path)
                )
                assert success is True
                
                # Test predictions
                texts = ["Great work!", "Not good"]
                results = predictor.predict_batch(texts)
                
                assert len(results) == 2
                assert results[0]['sentiment'] == 'positive'
                assert results[1]['sentiment'] == 'negative'
                
                # Test saving results
                output_path = temp_directory / "results.csv"
                save_success = predictor.save_predictions(results, str(output_path))
                assert save_success is True
                assert output_path.exists()