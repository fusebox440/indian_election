"""
Twitter Data Collection Module
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Handles data collection from Twitter API using Tweepy.
"""

import os
import pandas as pd
import tweepy
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from pathlib import Path

from ..utils.config import config
from .preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataCollector:
    """Collects and processes tweets from Twitter API."""
    
    def __init__(self):
        """Initialize Twitter API client and preprocessor."""
        self.api = self._setup_twitter_api()
        self.preprocessor = TextPreprocessor()
        
        # CSV columns for tweet data
        self.columns = [
            'id', 'created_at', 'source', 'original_text', 'clean_text', 
            'sentiment', 'polarity', 'subjectivity', 'lang',
            'favorite_count', 'retweet_count', 'original_author', 
            'possibly_sensitive', 'hashtags', 'user_mentions', 
            'place', 'place_coord_boundaries'
        ]
    
    def _setup_twitter_api(self) -> tweepy.API:
        """Setup Twitter API client with credentials."""
        twitter_config = config.get_twitter_config()
        
        # Validate credentials
        required_keys = ['consumer_key', 'consumer_secret', 'access_token', 'access_token_secret']
        for key in required_keys:
            if not twitter_config.get(key):
                raise ValueError(f"Twitter API credential missing: {key}")
        
        # Setup authentication
        auth = tweepy.OAuthHandler(
            twitter_config['consumer_key'],
            twitter_config['consumer_secret']
        )
        auth.set_access_token(
            twitter_config['access_token'],
            twitter_config['access_token_secret']
        )
        
        # Create API client
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Verify credentials
        try:
            api.verify_credentials()
            logger.info("Twitter API authentication successful")
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API authentication failed: {e}")
            raise
        
        return api
    
    def collect_tweets(
        self,
        keywords: str,
        output_file: str,
        max_tweets: int = 10000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Collect tweets based on keywords and save to CSV.
        
        Args:
            keywords: Search keywords/hashtags
            output_file: Path to output CSV file
            max_tweets: Maximum number of tweets to collect
            start_date: Start date for tweet collection (YYYY-MM-DD)
            end_date: End date for tweet collection (YYYY-MM-DD)
            
        Returns:
            DataFrame containing collected tweets
        """
        logger.info(f"Starting tweet collection for keywords: {keywords}")
        
        # Load existing data if file exists
        output_path = Path(output_file)
        if output_path.exists():
            df = pd.read_csv(output_path)
            logger.info(f"Loaded {len(df)} existing tweets from {output_path}")
        else:
            df = pd.DataFrame(columns=self.columns)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get search parameters from config
        search_config = config.get('data_collection.search_params', {})
        count = search_config.get('count', 100)
        pages = min(max_tweets // count, search_config.get('pages', 50))
        
        collected_count = 0
        
        try:
            # Search tweets using cursor
            for page in tweepy.Cursor(
                self.api.search_tweets,
                q=keywords,
                count=count,
                include_rts=search_config.get('include_rts', False),
                lang=search_config.get('lang', 'en'),
                since=start_date,
                until=end_date,
                result_type=search_config.get('result_type', 'recent')
            ).pages(pages):
                
                for status in page:
                    # Skip if tweet already exists
                    if status.created_at.isoformat() in df['created_at'].values:
                        continue
                    
                    # Process tweet
                    tweet_data = self._process_tweet(status)
                    if tweet_data:
                        # Add to dataframe
                        df = pd.concat([df, pd.DataFrame([tweet_data])], ignore_index=True)
                        collected_count += 1
                        
                        if collected_count >= max_tweets:
                            break
                
                # Save progress periodically
                if collected_count % 1000 == 0:
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved progress: {collected_count} tweets collected")
                
                if collected_count >= max_tweets:
                    break
                
                # Rate limiting pause
                time.sleep(1)
        
        except tweepy.TweepyException as e:
            logger.error(f"Error collecting tweets: {e}")
        
        # Final save
        df.to_csv(output_path, index=False)
        logger.info(f"Tweet collection completed. Total tweets: {len(df)}")
        
        return df
    
    def _process_tweet(self, status) -> Optional[Dict[str, Any]]:
        """
        Process a single tweet and extract relevant information.
        
        Args:
            status: Tweepy status object
            
        Returns:
            Dictionary containing processed tweet data
        """
        try:
            # Skip non-English tweets
            if status.lang != 'en':
                return None
            
            # Clean tweet text
            original_text = status.full_text if hasattr(status, 'full_text') else status.text
            clean_text = self.preprocessor.clean_text(original_text)
            
            # Get sentiment analysis
            sentiment_data = self.preprocessor.analyze_sentiment(clean_text)
            
            # Extract hashtags and mentions
            hashtags = ", ".join([hashtag['text'] for hashtag in status.entities['hashtags']])
            mentions = ", ".join([mention['screen_name'] for mention in status.entities['user_mentions']])
            
            # Get location data
            location = getattr(status.user, 'location', '')
            coordinates = None
            if hasattr(status, 'place') and status.place:
                try:
                    coordinates = [
                        coord for loc in status.place.bounding_box.coordinates 
                        for coord in loc
                    ]
                except (AttributeError, TypeError):
                    coordinates = None
            
            # Create tweet data dictionary
            tweet_data = {
                'id': status.id,
                'created_at': status.created_at.isoformat(),
                'source': status.source,
                'original_text': original_text,
                'clean_text': clean_text,
                'sentiment': str(sentiment_data['sentiment']),
                'polarity': sentiment_data['polarity'],
                'subjectivity': sentiment_data['subjectivity'],
                'lang': status.lang,
                'favorite_count': status.favorite_count,
                'retweet_count': status.retweet_count,
                'original_author': status.user.screen_name,
                'possibly_sensitive': getattr(status, 'possibly_sensitive', None),
                'hashtags': hashtags,
                'user_mentions': mentions,
                'place': location,
                'place_coord_boundaries': coordinates
            }
            
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error processing tweet {status.id}: {e}")
            return None
    
    def collect_party_data(
        self, 
        party: str, 
        output_dir: str,
        max_tweets: int = 10000
    ) -> pd.DataFrame:
        """
        Collect tweets for a specific political party.
        
        Args:
            party: Party name ('bjp' or 'congress')
            output_dir: Directory to save output files
            max_tweets: Maximum tweets to collect
            
        Returns:
            DataFrame containing collected tweets
        """
        # Get party keywords from config
        keywords = config.get(f'data_collection.keywords.{party.lower()}')
        if not keywords:
            raise ValueError(f"Keywords not found for party: {party}")
        
        # Set output file path
        output_file = Path(output_dir) / f"{party.lower()}_tweets.csv"
        
        # Get date range from config
        date_config = config.get('data_collection.date_range', {})
        start_date = date_config.get('start_date')
        end_date = date_config.get('end_date')
        
        # Collect tweets
        return self.collect_tweets(
            keywords=keywords,
            output_file=str(output_file),
            max_tweets=max_tweets,
            start_date=start_date,
            end_date=end_date
        )
    
    def collect_all_parties(self, output_dir: str, max_tweets: int = 10000) -> Dict[str, pd.DataFrame]:
        """
        Collect tweets for all configured political parties.
        
        Args:
            output_dir: Directory to save output files
            max_tweets: Maximum tweets per party
            
        Returns:
            Dictionary mapping party names to DataFrames
        """
        parties = ['bjp', 'congress']
        results = {}
        
        for party in parties:
            logger.info(f"Collecting tweets for party: {party}")
            try:
                df = self.collect_party_data(party, output_dir, max_tweets)
                results[party] = df
                logger.info(f"Successfully collected {len(df)} tweets for {party}")
            except Exception as e:
                logger.error(f"Failed to collect tweets for {party}: {e}")
                results[party] = pd.DataFrame()
        
        return results