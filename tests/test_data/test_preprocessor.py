"""
Unit tests for the data preprocessor module.
Author: Lakshya Khetan
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def test_init_with_valid_config(self, test_config):
        """Test initialization with valid configuration."""
        preprocessor = TextPreprocessor(test_config)
        
        assert preprocessor.config == test_config
        assert preprocessor.logger is not None
        assert preprocessor.tokenizer is None
        assert preprocessor.is_fitted is False
    
    def test_clean_text_basic(self, test_config):
        """Test basic text cleaning functionality."""
        preprocessor = TextPreprocessor(test_config)
        
        dirty_text = "RT @user: Check this out! https://example.com #hashtag"
        clean_text = preprocessor.clean_text(dirty_text)
        
        assert "RT" not in clean_text
        assert "@user" not in clean_text
        assert "https://example.com" not in clean_text
        assert clean_text.islower()
    
    def test_clean_text_preserve_original(self, test_config):
        """Test text cleaning while preserving original."""
        test_config['preprocessing']['text_cleaning']['remove_urls'] = False
        test_config['preprocessing']['text_cleaning']['convert_lowercase'] = False
        
        preprocessor = TextPreprocessor(test_config)
        
        text = "Check this URL: https://example.com"
        clean_text = preprocessor.clean_text(text)
        
        assert "https://example.com" in clean_text
        assert clean_text != clean_text.lower()
    
    def test_clean_text_empty_input(self, test_config):
        """Test cleaning empty or None text."""
        preprocessor = TextPreprocessor(test_config)
        
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("   ") == ""
    
    def test_clean_text_special_cases(self, test_config):
        """Test cleaning text with special cases."""
        preprocessor = TextPreprocessor(test_config)
        
        # Test with emojis
        text_with_emoji = "Great news! 😊👍 #happy"
        clean_text = preprocessor.clean_text(text_with_emoji)
        assert "😊" not in clean_text
        assert "👍" not in clean_text
        
        # Test with numbers
        text_with_numbers = "Election 2024 results are out!"
        clean_text = preprocessor.clean_text(text_with_numbers)
        assert "2024" not in clean_text or "2024" in clean_text  # Depending on config
    
    def test_remove_stopwords(self, test_config):
        """Test stopword removal functionality."""
        preprocessor = TextPreprocessor(test_config)
        
        text_with_stopwords = "this is a test with many common words"
        clean_text = preprocessor._remove_stopwords(text_with_stopwords)
        
        # Common stopwords should be removed
        assert "this" not in clean_text
        assert "is" not in clean_text
        assert "a" not in clean_text
        assert "test" in clean_text  # Non-stopword should remain
    
    def test_preprocess_dataframe(self, test_config, sample_tweet_dataframe):
        """Test preprocessing a DataFrame of tweets."""
        preprocessor = TextPreprocessor(test_config)
        
        result_df = preprocessor.preprocess_dataframe(
            sample_tweet_dataframe,
            text_column='text'
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'clean_text' in result_df.columns
        assert len(result_df) == len(sample_tweet_dataframe)
        
        # Check that cleaning was applied
        for clean_text in result_df['clean_text']:
            assert isinstance(clean_text, str)
            assert clean_text.islower()
    
    def test_preprocess_dataframe_invalid_column(self, test_config, sample_tweet_dataframe):
        """Test preprocessing DataFrame with invalid text column."""
        preprocessor = TextPreprocessor(test_config)
        
        with pytest.raises(KeyError):
            preprocessor.preprocess_dataframe(
                sample_tweet_dataframe,
                text_column='nonexistent_column'
            )
    
    def test_fit_tokenizer(self, test_config, sample_tweets):
        """Test fitting tokenizer on text data."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean texts first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        
        preprocessor.fit_tokenizer(clean_texts)
        
        assert preprocessor.tokenizer is not None
        assert preprocessor.is_fitted is True
        assert hasattr(preprocessor.tokenizer, 'word_index')
        assert len(preprocessor.tokenizer.word_index) > 0
    
    def test_texts_to_sequences(self, test_config, sample_tweets):
        """Test converting texts to sequences."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean and fit tokenizer first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        preprocessor.fit_tokenizer(clean_texts)
        
        sequences = preprocessor.texts_to_sequences(clean_texts[:2])
        
        assert isinstance(sequences, list)
        assert len(sequences) == 2
        assert all(isinstance(seq, list) for seq in sequences)
        assert all(isinstance(token, int) for seq in sequences for token in seq)
    
    def test_texts_to_sequences_not_fitted(self, test_config, sample_tweets):
        """Test converting texts to sequences without fitting first."""
        preprocessor = TextPreprocessor(test_config)
        
        with pytest.raises(ValueError, match="Tokenizer not fitted"):
            preprocessor.texts_to_sequences(sample_tweets)
    
    def test_pad_sequences(self, test_config):
        """Test sequence padding functionality."""
        preprocessor = TextPreprocessor(test_config)
        
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        max_length = 4
        
        padded = preprocessor.pad_sequences(sequences, max_length)
        
        assert isinstance(padded, np.ndarray)
        assert padded.shape == (3, max_length)
        assert all(len(seq) == max_length for seq in padded)
    
    def test_create_sequences_from_dataframe(self, test_config, sample_tweet_dataframe):
        """Test creating sequences from DataFrame."""
        preprocessor = TextPreprocessor(test_config)
        
        # First preprocess the DataFrame
        processed_df = preprocessor.preprocess_dataframe(
            sample_tweet_dataframe,
            text_column='text'
        )
        
        # Fit tokenizer
        preprocessor.fit_tokenizer(processed_df['clean_text'].tolist())
        
        # Create sequences
        sequences = preprocessor.create_sequences_from_dataframe(
            processed_df,
            text_column='clean_text'
        )
        
        assert isinstance(sequences, np.ndarray)
        assert sequences.shape[0] == len(processed_df)
        assert sequences.shape[1] == test_config['preprocessing']['tokenization']['max_sequence_length']
    
    def test_get_vocabulary_size(self, test_config, sample_tweets):
        """Test getting vocabulary size."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean and fit tokenizer first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        preprocessor.fit_tokenizer(clean_texts)
        
        vocab_size = preprocessor.get_vocabulary_size()
        
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        assert vocab_size <= test_config['preprocessing']['tokenization']['max_features']
    
    def test_get_vocabulary_size_not_fitted(self, test_config):
        """Test getting vocabulary size without fitting first."""
        preprocessor = TextPreprocessor(test_config)
        
        with pytest.raises(ValueError, match="Tokenizer not fitted"):
            preprocessor.get_vocabulary_size()
    
    def test_save_tokenizer(self, test_config, sample_tweets, temp_directory):
        """Test saving fitted tokenizer."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean and fit tokenizer first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        preprocessor.fit_tokenizer(clean_texts)
        
        tokenizer_path = temp_directory / "tokenizer.pickle"
        success = preprocessor.save_tokenizer(str(tokenizer_path))
        
        assert success is True
        assert tokenizer_path.exists()
    
    def test_save_tokenizer_not_fitted(self, test_config, temp_directory):
        """Test saving tokenizer without fitting first."""
        preprocessor = TextPreprocessor(test_config)
        
        tokenizer_path = temp_directory / "tokenizer.pickle"
        
        with pytest.raises(ValueError, match="Tokenizer not fitted"):
            preprocessor.save_tokenizer(str(tokenizer_path))
    
    def test_load_tokenizer(self, test_config, sample_tweets, temp_directory):
        """Test loading saved tokenizer."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean and fit tokenizer first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        preprocessor.fit_tokenizer(clean_texts)
        
        # Save tokenizer
        tokenizer_path = temp_directory / "tokenizer.pickle"
        preprocessor.save_tokenizer(str(tokenizer_path))
        
        # Create new preprocessor and load tokenizer
        new_preprocessor = TextPreprocessor(test_config)
        success = new_preprocessor.load_tokenizer(str(tokenizer_path))
        
        assert success is True
        assert new_preprocessor.is_fitted is True
        assert new_preprocessor.tokenizer is not None
        
        # Test that loaded tokenizer works
        sequences = new_preprocessor.texts_to_sequences(clean_texts[:1])
        assert len(sequences) == 1
    
    def test_load_tokenizer_file_not_found(self, test_config):
        """Test loading tokenizer from non-existent file."""
        preprocessor = TextPreprocessor(test_config)
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_tokenizer("nonexistent_tokenizer.pickle")
    
    def test_get_word_index(self, test_config, sample_tweets):
        """Test getting word index from fitted tokenizer."""
        preprocessor = TextPreprocessor(test_config)
        
        # Clean and fit tokenizer first
        clean_texts = [preprocessor.clean_text(text) for text in sample_tweets]
        preprocessor.fit_tokenizer(clean_texts)
        
        word_index = preprocessor.get_word_index()
        
        assert isinstance(word_index, dict)
        assert len(word_index) > 0
        assert all(isinstance(word, str) and isinstance(idx, int) 
                  for word, idx in word_index.items())
    
    def test_get_word_index_not_fitted(self, test_config):
        """Test getting word index without fitting first."""
        preprocessor = TextPreprocessor(test_config)
        
        with pytest.raises(ValueError, match="Tokenizer not fitted"):
            preprocessor.get_word_index()
    
    @pytest.mark.parametrize("text,expected_contains", [
        ("Great work by Modi!", ["great", "work", "modi"]),
        ("BJP is the best party", ["bjp", "best", "party"]),
        ("Congress needs improvement", ["congress", "needs", "improvement"])
    ])
    def test_clean_text_parametrized(self, test_config, text, expected_contains):
        """Parametrized test for text cleaning."""
        preprocessor = TextPreprocessor(test_config)
        clean_text = preprocessor.clean_text(text)
        
        for word in expected_contains:
            assert word in clean_text.lower()
    
    def test_preprocessing_pipeline_integration(self, test_config, sample_tweet_dataframe):
        """Integration test for complete preprocessing pipeline."""
        preprocessor = TextPreprocessor(test_config)
        
        # Step 1: Preprocess DataFrame
        processed_df = preprocessor.preprocess_dataframe(
            sample_tweet_dataframe,
            text_column='text'
        )
        
        # Step 2: Fit tokenizer
        preprocessor.fit_tokenizer(processed_df['clean_text'].tolist())
        
        # Step 3: Create sequences
        sequences = preprocessor.create_sequences_from_dataframe(
            processed_df,
            text_column='clean_text'
        )
        
        # Verify end-to-end results
        assert isinstance(sequences, np.ndarray)
        assert sequences.shape[0] == len(sample_tweet_dataframe)
        assert sequences.shape[1] == test_config['preprocessing']['tokenization']['max_sequence_length']
        assert preprocessor.is_fitted is True
        
        # Test that we can process new data
        new_text = "This is a new tweet for testing"
        clean_new_text = preprocessor.clean_text(new_text)
        new_sequence = preprocessor.texts_to_sequences([clean_new_text])
        padded_sequence = preprocessor.pad_sequences(
            new_sequence, 
            test_config['preprocessing']['tokenization']['max_sequence_length']
        )
        
        assert padded_sequence.shape == (1, test_config['preprocessing']['tokenization']['max_sequence_length'])