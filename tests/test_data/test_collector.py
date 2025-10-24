"""
Unit tests for the data collector module.
Author: Lakshya Khetan
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.collector import TwitterDataCollector


class TestTwitterDataCollector:
    """Test cases for TwitterDataCollector class."""
    
    def test_init_with_valid_config(self, test_config):
        """Test initialization with valid configuration."""
        collector = TwitterDataCollector(test_config)
        
        assert collector.config == test_config
        assert collector.logger is not None
        assert collector.data is not None
        assert isinstance(collector.data, list)
    
    def test_init_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        invalid_config = {'twitter': {}}  # Missing required keys
        
        with pytest.raises(KeyError):
            TwitterDataCollector(invalid_config)
    
    @patch('src.data.collector.tweepy.Client')
    def test_setup_api_connection_success(self, mock_client, test_config):
        """Test successful API connection setup."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        collector = TwitterDataCollector(test_config)
        result = collector._setup_api_connection()
        
        assert result is True
        mock_client.assert_called_once()
    
    @patch('src.data.collector.tweepy.Client')
    def test_search_tweets_success(self, mock_client, test_config, mock_twitter_api):
        """Test successful tweet search."""
        mock_client.return_value = mock_twitter_api
        
        collector = TwitterDataCollector(test_config)
        collector.api = mock_twitter_api
        
        # Mock search response
        mock_tweet = Mock()
        mock_tweet.id = 123
        mock_tweet.text = "Test tweet"
        mock_tweet.author_id = 456
        mock_tweet.created_at = datetime.now()
        mock_tweet.lang = 'en'
        mock_tweet.public_metrics = {
            'like_count': 10,
            'retweet_count': 5,
            'reply_count': 2
        }
        
        mock_response = Mock()
        mock_response.data = [mock_tweet]
        mock_twitter_api.search_recent_tweets.return_value = mock_response
        
        result = collector.search_tweets(
            keywords=["modi", "bjp"],
            count=10,
            lang='en'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'id' in result.columns
        assert 'text' in result.columns
    
    @patch('src.data.collector.tweepy.Client')
    def test_search_tweets_no_results(self, mock_client, test_config):
        """Test tweet search with no results."""
        mock_client_instance = Mock()
        mock_client_instance.search_recent_tweets.return_value = Mock(data=None)
        mock_client.return_value = mock_client_instance
        
        collector = TwitterDataCollector(test_config)
        collector.api = mock_client_instance
        
        result = collector.search_tweets(
            keywords=["nonexistent_keyword"],
            count=10
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    @patch('src.data.collector.tweepy.Client')
    def test_search_tweets_api_error(self, mock_client, test_config):
        """Test tweet search with API error."""
        mock_client_instance = Mock()
        mock_client_instance.search_recent_tweets.side_effect = Exception("API Error")
        mock_client.return_value = mock_client_instance
        
        collector = TwitterDataCollector(test_config)
        collector.api = mock_client_instance
        
        with pytest.raises(Exception):
            collector.search_tweets(
                keywords=["test"],
                count=10
            )
    
    def test_save_data_csv(self, test_config, sample_tweet_dataframe, temp_directory):
        """Test saving data to CSV file."""
        collector = TwitterDataCollector(test_config)
        collector.data = sample_tweet_dataframe.to_dict('records')
        
        file_path = temp_directory / "test_tweets.csv"
        result = collector.save_data(str(file_path), format='csv')
        
        assert result is True
        assert file_path.exists()
        
        # Verify saved data
        saved_df = pd.read_csv(file_path)
        assert len(saved_df) == len(sample_tweet_dataframe)
    
    def test_save_data_json(self, test_config, sample_tweet_dataframe, temp_directory):
        """Test saving data to JSON file."""
        collector = TwitterDataCollector(test_config)
        collector.data = sample_tweet_dataframe.to_dict('records')
        
        file_path = temp_directory / "test_tweets.json"
        result = collector.save_data(str(file_path), format='json')
        
        assert result is True
        assert file_path.exists()
        
        # Verify saved data can be loaded
        import json
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == len(sample_tweet_dataframe)
    
    def test_save_data_invalid_format(self, test_config):
        """Test saving data with invalid format."""
        collector = TwitterDataCollector(test_config)
        
        with pytest.raises(ValueError):
            collector.save_data("test.txt", format='invalid')
    
    def test_load_data_csv(self, test_config, sample_tweet_dataframe, temp_directory):
        """Test loading data from CSV file."""
        # Save sample data first
        file_path = temp_directory / "test_tweets.csv"
        sample_tweet_dataframe.to_csv(file_path, index=False)
        
        collector = TwitterDataCollector(test_config)
        result = collector.load_data(str(file_path), format='csv')
        
        assert result is True
        assert len(collector.data) == len(sample_tweet_dataframe)
    
    def test_load_data_json(self, test_config, sample_tweet_dataframe, temp_directory):
        """Test loading data from JSON file."""
        # Save sample data first
        file_path = temp_directory / "test_tweets.json"
        sample_tweet_dataframe.to_json(file_path, orient='records')
        
        collector = TwitterDataCollector(test_config)
        result = collector.load_data(str(file_path), format='json')
        
        assert result is True
        assert len(collector.data) == len(sample_tweet_dataframe)
    
    def test_load_data_file_not_found(self, test_config):
        """Test loading data from non-existent file."""
        collector = TwitterDataCollector(test_config)
        
        with pytest.raises(FileNotFoundError):
            collector.load_data("nonexistent.csv", format='csv')
    
    def test_filter_by_language(self, test_config):
        """Test filtering tweets by language."""
        collector = TwitterDataCollector(test_config)
        
        # Mock data with different languages
        collector.data = [
            {'id': 1, 'text': 'English tweet', 'lang': 'en'},
            {'id': 2, 'text': 'Hindi tweet', 'lang': 'hi'},
            {'id': 3, 'text': 'Another English tweet', 'lang': 'en'}
        ]
        
        result = collector.filter_by_language('en')
        assert result is True
        assert len(collector.data) == 2
        assert all(tweet['lang'] == 'en' for tweet in collector.data)
    
    def test_filter_by_date_range(self, test_config):
        """Test filtering tweets by date range."""
        collector = TwitterDataCollector(test_config)
        
        # Mock data with different dates
        collector.data = [
            {'id': 1, 'created_at': '2023-01-01', 'text': 'Old tweet'},
            {'id': 2, 'created_at': '2023-06-01', 'text': 'Recent tweet'},
            {'id': 3, 'created_at': '2023-12-01', 'text': 'New tweet'}
        ]
        
        result = collector.filter_by_date_range('2023-05-01', '2023-12-31')
        assert result is True
        assert len(collector.data) == 2
    
    def test_get_data_stats(self, test_config, sample_tweet_dataframe):
        """Test getting data statistics."""
        collector = TwitterDataCollector(test_config)
        collector.data = sample_tweet_dataframe.to_dict('records')
        
        stats = collector.get_data_stats()
        
        assert 'total_tweets' in stats
        assert 'unique_users' in stats
        assert stats['total_tweets'] == len(sample_tweet_dataframe)
    
    def test_clear_data(self, test_config, sample_tweet_dataframe):
        """Test clearing collected data."""
        collector = TwitterDataCollector(test_config)
        collector.data = sample_tweet_dataframe.to_dict('records')
        
        assert len(collector.data) > 0
        
        collector.clear_data()
        assert len(collector.data) == 0
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_integration_search_and_save(self, test_config, temp_directory):
        """Integration test for searching and saving tweets."""
        # This would require actual API credentials
        # Mock for testing purposes
        with patch('src.data.collector.tweepy.Client') as mock_client:
            mock_instance = Mock()
            mock_tweet = Mock()
            mock_tweet.id = 123
            mock_tweet.text = "Integration test tweet"
            mock_tweet.author_id = 456
            mock_tweet.created_at = datetime.now()
            mock_tweet.lang = 'en'
            mock_tweet.public_metrics = {'like_count': 0, 'retweet_count': 0}
            
            mock_response = Mock()
            mock_response.data = [mock_tweet]
            mock_instance.search_recent_tweets.return_value = mock_response
            mock_client.return_value = mock_instance
            
            collector = TwitterDataCollector(test_config)
            collector.api = mock_instance
            
            # Search tweets
            df = collector.search_tweets(keywords=["test"], count=1)
            assert len(df) > 0
            
            # Save data
            file_path = temp_directory / "integration_test.csv"
            result = collector.save_data(str(file_path))
            assert result is True
            assert file_path.exists()