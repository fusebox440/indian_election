"""
Unit tests for the sentiment models module.
Author: Lakshya Khetan
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.sentiment_models import SentimentModelFactory, LSTMSentimentModel


class TestSentimentModelFactory:
    """Test cases for SentimentModelFactory class."""
    
    def test_create_lstm_model(self, test_config):
        """Test creating LSTM model through factory."""
        factory = SentimentModelFactory(test_config)
        
        model = factory.create_model(
            model_type='lstm',
            vocab_size=1000,
            max_sequence_length=100
        )
        
        assert isinstance(model, LSTMSentimentModel)
        assert model.vocab_size == 1000
        assert model.max_sequence_length == 100
    
    def test_create_invalid_model_type(self, test_config):
        """Test creating model with invalid type."""
        factory = SentimentModelFactory(test_config)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            factory.create_model(
                model_type='invalid_type',
                vocab_size=1000,
                max_sequence_length=100
            )
    
    def test_get_available_models(self, test_config):
        """Test getting list of available models."""
        factory = SentimentModelFactory(test_config)
        available_models = factory.get_available_models()
        
        assert isinstance(available_models, list)
        assert 'lstm' in available_models


class TestLSTMSentimentModel:
    """Test cases for LSTMSentimentModel class."""
    
    def test_init_with_valid_params(self, test_config):
        """Test initialization with valid parameters."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        assert model.vocab_size == 1000
        assert model.max_sequence_length == 100
        assert model.config == test_config
        assert model.model is None
        assert model.history is None
    
    def test_init_with_invalid_params(self, test_config):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            LSTMSentimentModel(
                config=test_config,
                vocab_size=0,  # Invalid vocab size
                max_sequence_length=100
            )
        
        with pytest.raises(ValueError):
            LSTMSentimentModel(
                config=test_config,
                vocab_size=1000,
                max_sequence_length=0  # Invalid sequence length
            )
    
    def test_build_model(self, test_config):
        """Test building the LSTM model architecture."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        built_model = model.build_model()
        
        assert built_model is not None
        assert isinstance(built_model, tf.keras.Model)
        assert model.model is not None
        
        # Check model architecture
        assert len(built_model.layers) > 0
        assert built_model.input_shape == (None, 100)
        assert built_model.output_shape == (None, 1)
    
    def test_compile_model(self, test_config):
        """Test compiling the model."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model.build_model()
        model.compile_model()
        
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        assert len(model.model.metrics) > 0
    
    def test_train_model(self, test_config, sample_model_data):
        """Test training the model."""
        X, y = sample_model_data
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=50
        )
        
        model.build_model()
        model.compile_model()
        
        # Train model
        history = model.train(X, y, validation_split=0.2)
        
        assert history is not None
        assert model.history is not None
        assert 'loss' in history.history
        assert 'accuracy' in history.history
    
    def test_train_model_with_validation_data(self, test_config, sample_model_data):
        """Test training model with separate validation data."""
        X, y = sample_model_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=50
        )
        
        model.build_model()
        model.compile_model()
        
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        assert history is not None
        assert 'val_loss' in history.history
        assert 'val_accuracy' in history.history
    
    def test_predict_single(self, test_config, mock_trained_model):
        """Test making predictions on single sample."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model.model = mock_trained_model
        
        # Single sample prediction
        sample_sequence = np.random.randint(0, 1000, (1, 100))
        prediction = model.predict(sample_sequence)
        
        assert prediction is not None
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape[0] == 1
    
    def test_predict_batch(self, test_config, mock_trained_model):
        """Test making predictions on batch of samples."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model.model = mock_trained_model
        
        # Batch prediction
        batch_sequences = np.random.randint(0, 1000, (5, 100))
        predictions = model.predict(batch_sequences)
        
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 5
    
    def test_predict_no_model(self, test_config):
        """Test making predictions without trained model."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        sample_sequence = np.random.randint(0, 1000, (1, 100))
        
        with pytest.raises(ValueError, match="Model not built or trained"):
            model.predict(sample_sequence)
    
    def test_evaluate_model(self, test_config, sample_model_data, mock_trained_model):
        """Test evaluating model performance."""
        X, y = sample_model_data
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=50
        )
        
        # Mock model evaluation
        mock_trained_model.evaluate.return_value = [0.5, 0.8]  # [loss, accuracy]
        model.model = mock_trained_model
        
        loss, accuracy = model.evaluate(X, y)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_save_model(self, test_config, temp_directory):
        """Test saving trained model."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        # Build model
        model.build_model()
        
        # Save model
        model_path = temp_directory / "test_model.h5"
        success = model.save_model(str(model_path))
        
        assert success is True
        assert model_path.exists()
    
    def test_save_model_no_model(self, test_config, temp_directory):
        """Test saving model when no model exists."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model_path = temp_directory / "test_model.h5"
        
        with pytest.raises(ValueError, match="No model to save"):
            model.save_model(str(model_path))
    
    def test_load_model(self, test_config, temp_directory):
        """Test loading saved model."""
        # First create and save a model
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model.build_model()
        model_path = temp_directory / "test_model.h5"
        model.save_model(str(model_path))
        
        # Create new model instance and load
        new_model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        success = new_model.load_model(str(model_path))
        
        assert success is True
        assert new_model.model is not None
    
    def test_load_model_file_not_found(self, test_config):
        """Test loading model from non-existent file."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        with pytest.raises(FileNotFoundError):
            model.load_model("nonexistent_model.h5")
    
    def test_get_model_summary(self, test_config):
        """Test getting model summary."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        model.build_model()
        summary = model.get_model_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "lstm" in summary.lower()
    
    def test_get_model_summary_no_model(self, test_config):
        """Test getting summary when no model exists."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        with pytest.raises(ValueError, match="Model not built"):
            model.get_model_summary()
    
    def test_get_training_history(self, test_config, sample_model_data):
        """Test getting training history."""
        X, y = sample_model_data
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=50
        )
        
        model.build_model()
        model.compile_model()
        model.train(X, y, validation_split=0.2)
        
        history = model.get_training_history()
        
        assert history is not None
        assert 'loss' in history
        assert 'accuracy' in history
    
    def test_get_training_history_no_training(self, test_config):
        """Test getting history when model wasn't trained."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        history = model.get_training_history()
        assert history is None
    
    def test_reset_model(self, test_config):
        """Test resetting model state."""
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        # Build and train model
        model.build_model()
        assert model.model is not None
        
        # Reset model
        model.reset()
        
        assert model.model is None
        assert model.history is None
    
    @pytest.mark.parametrize("embedding_dim,lstm_units", [
        (32, 16),
        (64, 32),
        (128, 64)
    ])
    def test_different_architectures(self, test_config, embedding_dim, lstm_units):
        """Test building models with different architectures."""
        # Update config for this test
        test_config['models']['lstm']['embedding_dim'] = embedding_dim
        test_config['models']['lstm']['lstm_units'] = lstm_units
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        built_model = model.build_model()
        
        assert built_model is not None
        # Check that the model was built successfully with different params
        assert len(built_model.layers) > 0
    
    def test_model_with_dropout(self, test_config):
        """Test model with dropout layers."""
        # Add dropout to config
        test_config['models']['lstm']['dropout'] = 0.3
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=100
        )
        
        built_model = model.build_model()
        
        assert built_model is not None
        # Model should still build successfully with dropout
        assert len(built_model.layers) > 0
    
    @pytest.mark.slow
    def test_integration_train_predict_cycle(self, test_config, sample_model_data):
        """Integration test for complete train-predict cycle."""
        X, y = sample_model_data
        
        model = LSTMSentimentModel(
            config=test_config,
            vocab_size=1000,
            max_sequence_length=50
        )
        
        # Build and train model
        model.build_model()
        model.compile_model()
        history = model.train(X, y, validation_split=0.2)
        
        # Make predictions
        predictions = model.predict(X[:5])
        
        # Evaluate model
        loss, accuracy = model.evaluate(X, y)
        
        # Verify everything worked
        assert history is not None
        assert predictions is not None
        assert len(predictions) == 5
        assert 0 <= accuracy <= 1
        assert loss >= 0