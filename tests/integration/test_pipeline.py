"""
Integration tests for the complete Twitter sentiment analysis pipeline.
Author: Lakshya Khetan
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.collector import TwitterDataCollector
from src.data.preprocessor import TextPreprocessor
from src.models.sentiment_models import SentimentModelFactory
from src.models.predictor import SentimentPredictor
from src.utils.config import ConfigManager


class TestPipelineIntegration:
    """Integration tests for the complete sentiment analysis pipeline."""
    
    @pytest.mark.integration
    def test_data_collection_to_preprocessing_pipeline(self, test_config, temp_directory):
        """Test integration from data collection to preprocessing."""
        # Setup data collector
        collector = TwitterDataCollector(test_config)
        
        # Mock Twitter API response
        with patch('src.data.collector.tweepy.Client') as mock_client:
            mock_instance = Mock()
            mock_tweet = Mock()
            mock_tweet.id = 123
            mock_tweet.text = "Great work by Modi! #BJP2024"
            mock_tweet.author_id = 456
            mock_tweet.created_at = "2023-01-01"
            mock_tweet.lang = 'en'
            mock_tweet.public_metrics = {'like_count': 10, 'retweet_count': 5}
            
            mock_response = Mock()
            mock_response.data = [mock_tweet]
            mock_instance.search_recent_tweets.return_value = mock_response
            mock_client.return_value = mock_instance
            
            collector.api = mock_instance
            
            # Collect tweets
            tweets_df = collector.search_tweets(keywords=["modi", "bjp"], count=1)
            assert len(tweets_df) > 0
            
            # Save collected data
            data_file = temp_directory / "collected_tweets.csv"
            collector.save_data(str(data_file))
            
            # Load data for preprocessing
            preprocessor = TextPreprocessor(test_config)
            
            # Load and preprocess the collected data
            collector.load_data(str(data_file))
            raw_df = pd.DataFrame(collector.data)
            
            processed_df = preprocessor.preprocess_dataframe(raw_df, text_column='text')
            
            # Verify the pipeline
            assert len(processed_df) == len(tweets_df)
            assert 'clean_text' in processed_df.columns
            assert processed_df['clean_text'].notna().all()
    
    @pytest.mark.integration  
    def test_preprocessing_to_model_training_pipeline(self, test_config, sample_tweet_dataframe, temp_directory):
        """Test integration from preprocessing to model training."""
        # Preprocess data
        preprocessor = TextPreprocessor(test_config)
        processed_df = preprocessor.preprocess_dataframe(sample_tweet_dataframe, text_column='text')
        
        # Fit tokenizer and create sequences
        preprocessor.fit_tokenizer(processed_df['clean_text'].tolist())
        sequences = preprocessor.create_sequences_from_dataframe(processed_df, text_column='clean_text')
        
        # Create dummy labels for training
        labels = np.random.randint(0, 2, len(sequences))
        
        # Create and train model
        factory = SentimentModelFactory(test_config)
        model = factory.create_model(
            model_type='lstm',
            vocab_size=preprocessor.get_vocabulary_size(),
            max_sequence_length=sequences.shape[1]
        )
        
        model.build_model()
        model.compile_model()
        
        # Train model (with reduced epochs for testing)
        test_config['models']['lstm']['training']['epochs'] = 1
        history = model.train(sequences, labels, validation_split=0.2)
        
        # Verify training completed
        assert history is not None
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1  # 1 epoch
        
        # Save model and preprocessor
        model_path = temp_directory / "trained_model.h5"
        tokenizer_path = temp_directory / "fitted_tokenizer.pickle"
        
        model.save_model(str(model_path))
        preprocessor.save_tokenizer(str(tokenizer_path))
        
        assert model_path.exists()
        assert tokenizer_path.exists()
    
    @pytest.mark.integration
    def test_model_training_to_prediction_pipeline(self, test_config, temp_directory):
        """Test integration from model training to predictions."""
        # Create mock trained model and tokenizer files
        model_path = temp_directory / "model.h5"
        tokenizer_path = temp_directory / "tokenizer.pickle"
        
        # Create predictor
        predictor = SentimentPredictor(test_config)
        
        # Mock the loading process
        with patch('src.models.sentiment_models.LSTMSentimentModel') as mock_model_class:
            with patch('src.data.preprocessor.TextPreprocessor') as mock_preprocessor_class:
                # Setup mocks
                mock_model = Mock()
                mock_model.load_model.return_value = True
                mock_model.predict.return_value = np.array([[0.8], [0.3], [0.7]])
                mock_model_class.return_value = mock_model
                
                mock_preprocessor = Mock()
                mock_preprocessor.load_tokenizer.return_value = True
                mock_preprocessor.clean_text.side_effect = lambda x: f"clean {x[:10]}"
                mock_preprocessor.texts_to_sequences.return_value = [[1, 2], [3, 4], [5, 6]]
                mock_preprocessor.pad_sequences.return_value = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0]])
                mock_preprocessor_class.return_value = mock_preprocessor
                
                # Create mock files
                model_path.touch()
                tokenizer_path.touch()
                
                # Load model and preprocessor
                success = predictor.load_model_and_preprocessor(str(model_path), str(tokenizer_path))
                assert success is True
                
                # Make predictions
                test_texts = [
                    "Great work by the government!",
                    "Not happy with current policies",
                    "Excellent leadership shown"
                ]
                
                results = predictor.predict_batch(test_texts)
                
                # Verify predictions
                assert len(results) == 3
                assert all('sentiment' in result for result in results)
                assert results[0]['sentiment'] == 'positive'  # 0.8 probability
                assert results[1]['sentiment'] == 'negative'  # 0.3 probability
                assert results[2]['sentiment'] == 'positive'  # 0.7 probability
                
                # Save predictions
                predictions_file = temp_directory / "predictions.csv"
                save_success = predictor.save_predictions(results, str(predictions_file))
                assert save_success is True
                assert predictions_file.exists()
    
    @pytest.mark.integration
    def test_complete_end_to_end_pipeline(self, test_config, temp_directory):
        """Test complete end-to-end pipeline integration."""
        # Step 1: Data Collection (mocked)
        collector = TwitterDataCollector(test_config)
        
        with patch('src.data.collector.tweepy.Client') as mock_client:
            # Mock multiple tweets
            mock_tweets = []
            for i in range(5):
                mock_tweet = Mock()
                mock_tweet.id = 100 + i
                mock_tweet.text = f"Test tweet {i} about elections #politics"
                mock_tweet.author_id = 500 + i
                mock_tweet.created_at = "2023-01-01"
                mock_tweet.lang = 'en'
                mock_tweet.public_metrics = {'like_count': i * 2, 'retweet_count': i}
                mock_tweets.append(mock_tweet)
            
            mock_instance = Mock()
            mock_response = Mock()
            mock_response.data = mock_tweets
            mock_instance.search_recent_tweets.return_value = mock_response
            mock_client.return_value = mock_instance
            
            collector.api = mock_instance
            
            # Collect data
            tweets_df = collector.search_tweets(keywords=["elections"], count=5)
            assert len(tweets_df) == 5
            
            # Save collected data
            raw_data_file = temp_directory / "raw_tweets.csv"
            collector.save_data(str(raw_data_file))
        
        # Step 2: Data Preprocessing
        preprocessor = TextPreprocessor(test_config)
        processed_df = preprocessor.preprocess_dataframe(tweets_df, text_column='text')
        
        # Fit tokenizer
        preprocessor.fit_tokenizer(processed_df['clean_text'].tolist())
        
        # Create sequences for training
        sequences = preprocessor.create_sequences_from_dataframe(processed_df, text_column='clean_text')
        
        # Save preprocessor
        tokenizer_file = temp_directory / "tokenizer.pickle"
        preprocessor.save_tokenizer(str(tokenizer_file))
        
        # Step 3: Model Training (simplified for testing)
        factory = SentimentModelFactory(test_config)
        model = factory.create_model(
            model_type='lstm',
            vocab_size=preprocessor.get_vocabulary_size(),
            max_sequence_length=sequences.shape[1]
        )
        
        model.build_model()
        model.compile_model()
        
        # Create dummy labels and train
        labels = np.random.randint(0, 2, len(sequences))
        test_config['models']['lstm']['training']['epochs'] = 1  # Reduce for testing
        
        history = model.train(sequences, labels)
        
        # Save trained model
        model_file = temp_directory / "sentiment_model.h5"
        model.save_model(str(model_file))
        
        # Step 4: Making Predictions
        predictor = SentimentPredictor(test_config)
        
        # Load saved model and preprocessor
        predictor.load_model_and_preprocessor(str(model_file), str(tokenizer_file))
        
        # Make predictions on new data
        new_texts = [
            "Excellent leadership by the current government",
            "Very disappointed with the recent policies",
            "Looking forward to the upcoming elections"
        ]
        
        predictions = predictor.predict_batch(new_texts)
        
        # Step 5: Save Results
        results_file = temp_directory / "final_predictions.json"
        predictor.save_predictions(predictions, str(results_file), format='json')
        
        # Verify complete pipeline
        assert raw_data_file.exists()
        assert tokenizer_file.exists()
        assert model_file.exists()
        assert results_file.exists()
        assert len(predictions) == 3
        assert all('sentiment' in pred for pred in predictions)
        
        # Verify data flow consistency
        assert len(tweets_df) == 5
        assert len(processed_df) == 5
        assert sequences.shape[0] == 5
        assert len(predictions) == 3
    
    @pytest.mark.integration
    def test_config_driven_pipeline(self, temp_directory):
        """Test pipeline driven by configuration file."""
        # Create comprehensive config
        config_data = {
            'twitter': {
                'consumer_key': 'test_key',
                'consumer_secret': 'test_secret',
                'access_token': 'test_token',
                'access_token_secret': 'test_token_secret'
            },
            'data_collection': {
                'keywords': ['modi', 'bjp', 'congress'],
                'count': 10,
                'lang': 'en'
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
                        'epochs': 1,
                        'learning_rate': 0.001
                    }
                }
            },
            'output': {
                'model_path': 'models/sentiment_model.h5',
                'tokenizer_path': 'models/tokenizer.pickle',
                'predictions_path': 'results/predictions.csv'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/pipeline.log'
            }
        }
        
        # Save config
        config_file = temp_directory / "pipeline_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config and run pipeline
        config_manager = ConfigManager(str(config_file))
        config = config_manager.get_config()
        
        # Test that each component can be initialized with this config
        collector = TwitterDataCollector(config)
        preprocessor = TextPreprocessor(config)
        factory = SentimentModelFactory(config)
        predictor = SentimentPredictor(config)
        
        # Verify config-driven initialization
        assert collector.config == config
        assert preprocessor.config == config
        assert factory.config == config
        assert predictor.config == config
        
        # Test config value retrieval
        assert config_manager.get('data_collection.count') == 10
        assert config_manager.get('models.lstm.embedding_dim') == 64
        assert config_manager.get('preprocessing.tokenization.max_features') == 1000
    
    @pytest.mark.integration
    def test_error_handling_in_pipeline(self, test_config, temp_directory):
        """Test error handling throughout the pipeline."""
        # Test data collection error handling
        collector = TwitterDataCollector(test_config)
        
        # Test with invalid API configuration
        invalid_config = test_config.copy()
        invalid_config['twitter']['consumer_key'] = ''
        
        with pytest.raises(Exception):  # Should raise authentication error
            invalid_collector = TwitterDataCollector(invalid_config)
            invalid_collector._setup_api_connection()
        
        # Test preprocessing error handling
        preprocessor = TextPreprocessor(test_config)
        
        # Test with invalid DataFrame
        invalid_df = pd.DataFrame({'wrong_column': ['text1', 'text2']})
        
        with pytest.raises(KeyError):
            preprocessor.preprocess_dataframe(invalid_df, text_column='text')
        
        # Test model error handling
        factory = SentimentModelFactory(test_config)
        
        with pytest.raises(ValueError):
            factory.create_model('invalid_model_type', 1000, 100)
        
        # Test predictor error handling
        predictor = SentimentPredictor(test_config)
        
        with pytest.raises(ValueError):
            predictor.predict_text("Test text")  # No model loaded
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_benchmarks(self, test_config, temp_directory):
        """Test performance benchmarks for the pipeline."""
        import time
        
        # Benchmark data preprocessing
        preprocessor = TextPreprocessor(test_config)
        
        # Create large dataset for benchmarking
        large_texts = [f"Test tweet number {i} with political content" for i in range(1000)]
        
        start_time = time.time()
        clean_texts = [preprocessor.clean_text(text) for text in large_texts]
        preprocessing_time = time.time() - start_time
        
        assert preprocessing_time < 30  # Should complete within 30 seconds
        assert len(clean_texts) == 1000
        
        # Benchmark tokenization
        start_time = time.time()
        preprocessor.fit_tokenizer(clean_texts)
        tokenization_time = time.time() - start_time
        
        assert tokenization_time < 10  # Should complete within 10 seconds
        assert preprocessor.is_fitted is True
        
        # Benchmark sequence creation
        start_time = time.time()
        sequences = preprocessor.texts_to_sequences(clean_texts[:100])  # Subset for speed
        sequence_time = time.time() - start_time
        
        assert sequence_time < 5  # Should complete within 5 seconds
        assert len(sequences) == 100
    
    @pytest.mark.integration
    def test_pipeline_reproducibility(self, test_config, temp_directory):
        """Test that the pipeline produces reproducible results."""
        import numpy as np
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        # Create test data
        test_texts = [
            "Great work by Modi government",
            "Congress has better policies",
            "Election results are pending",
            "BJP wins in the state",
            "Opposition raises concerns"
        ]
        
        # Run pipeline twice with same configuration
        results1 = self._run_mock_pipeline(test_config, test_texts, temp_directory, "run1")
        results2 = self._run_mock_pipeline(test_config, test_texts, temp_directory, "run2")
        
        # Compare results - should be identical
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1['original_text'] == r2['original_text']
            assert r1['clean_text'] == r2['clean_text']
            # Note: Predictions might differ due to model randomness in real scenario
    
    def _run_mock_pipeline(self, config, texts, temp_dir, run_id):
        """Helper method to run a mocked pipeline."""
        # Mock preprocessor
        preprocessor = TextPreprocessor(config)
        
        # Process texts
        clean_texts = [preprocessor.clean_text(text) for text in texts]
        
        # Mock predictions (deterministic for testing)
        predictions = []
        for i, (original, clean) in enumerate(zip(texts, clean_texts)):
            # Simple deterministic prediction based on text length
            prob = 0.6 + (len(clean) % 10) * 0.04  # Between 0.6 and 0.96
            sentiment = 'positive' if prob > 0.5 else 'negative'
            
            predictions.append({
                'original_text': original,
                'clean_text': clean,
                'prediction': 1 if sentiment == 'positive' else 0,
                'probability': prob,
                'sentiment': sentiment
            })
        
        return predictions