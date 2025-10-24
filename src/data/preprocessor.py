"""
Text Preprocessing Module
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Handles text cleaning, preprocessing, and sentiment analysis for tweets.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import logging

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import preprocessor as p

# Keras/TensorFlow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing and sentiment analysis for tweets."""
    
    def __init__(self):
        """Initialize preprocessor with NLTK data and configuration."""
        self._setup_nltk()
        self.config = config.get('preprocessing', {})
        self._setup_emoticons()
        self._setup_regex_patterns()
        self.tokenizer = None
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up NLTK: {e}")
            self.stop_words = set()
    
    def _setup_emoticons(self):
        """Setup emoticon patterns for cleaning."""
        # Happy emoticons
        self.emoticons_happy = {
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
        }
        
        # Sad emoticons
        self.emoticons_sad = {
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
        }
        
        # Combine all emoticons
        self.emoticons = self.emoticons_happy.union(self.emoticons_sad)
    
    def _setup_regex_patterns(self):
        """Setup regex patterns for text cleaning."""
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        # URL pattern
        self.url_pattern = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))')
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        
        # Hashtag pattern (for removal if configured)
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        # RT pattern
        self.rt_pattern = re.compile(r'\bRT\b')
    
    def clean_text(self, text: str) -> str:
        """
        Clean tweet text based on configuration settings.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Start with original text
        cleaned_text = text
        
        # Remove URLs
        if self.config.get('text_cleaning.remove_urls', True):
            cleaned_text = self.url_pattern.sub('', cleaned_text)
        
        # Remove mentions
        if self.config.get('text_cleaning.remove_mentions', True):
            cleaned_text = self.mention_pattern.sub('', cleaned_text)
        
        # Remove hashtags (but keep the text)
        if self.config.get('text_cleaning.remove_hashtags', False):
            cleaned_text = self.hashtag_pattern.sub('', cleaned_text)
        
        # Remove RT
        if self.config.get('text_cleaning.remove_rt', True):
            cleaned_text = self.rt_pattern.sub('', cleaned_text)
        
        # Remove emojis
        if self.config.get('text_cleaning.remove_emojis', True):
            cleaned_text = self.emoji_pattern.sub('', cleaned_text)
        
        # Remove special characters and colons
        cleaned_text = re.sub(r':', '', cleaned_text)
        cleaned_text = re.sub(r'‚Ä¶', '', cleaned_text)
        
        # Replace consecutive non-ASCII characters with space
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
        
        # Convert to lowercase
        if self.config.get('text_cleaning.convert_lowercase', True):
            cleaned_text = cleaned_text.lower()
        
        # Tokenize for further processing
        try:
            word_tokens = word_tokenize(cleaned_text)
        except:
            word_tokens = cleaned_text.split()
        
        # Filter tokens
        filtered_tokens = []
        min_length = self.config.get('text_cleaning.min_word_length', 2)
        
        for token in word_tokens:
            # Skip stopwords
            if (self.config.get('text_cleaning.remove_stopwords', True) and 
                token.lower() in self.stop_words):
                continue
            
            # Skip emoticons
            if token in self.emoticons:
                continue
            
            # Skip punctuation
            if (self.config.get('text_cleaning.remove_punctuation', True) and 
                token in string.punctuation):
                continue
            
            # Skip short tokens
            if len(token) < min_length:
                continue
            
            # Keep only alphabetic characters if configured
            if re.match(r'^[a-zA-Z]+$', token):
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Binary classification based on threshold
            threshold = self.config.get('sentiment_labeling.threshold', 0.0)
            binary_sentiment = 1 if sentiment.polarity > threshold else 0
            
            return {
                'sentiment': sentiment,
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'binary_sentiment': binary_sentiment
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': None,
                'polarity': 0.0,
                'subjectivity': 0.0,
                'binary_sentiment': 0
            }
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess a DataFrame containing tweets.
        
        Args:
            df: DataFrame with tweet data
            text_column: Name of column containing tweet text
            
        Returns:
            DataFrame with processed text and sentiment
        """
        logger.info(f"Preprocessing {len(df)} tweets")
        
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text
        processed_df['clean_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Analyze sentiment
        sentiment_results = processed_df['clean_text'].apply(self.analyze_sentiment)
        
        processed_df['polarity'] = sentiment_results.apply(lambda x: x['polarity'])
        processed_df['subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
        processed_df['binary_sentiment'] = sentiment_results.apply(lambda x: x['binary_sentiment'])
        
        # Filter out empty texts
        processed_df = processed_df[processed_df['clean_text'].str.strip() != '']
        
        logger.info(f"Preprocessing completed. {len(processed_df)} tweets remaining")
        
        return processed_df
    
    def create_tokenizer(self, texts: List[str], max_features: Optional[int] = None) -> Tokenizer:
        """
        Create and fit tokenizer on texts.
        
        Args:
            texts: List of texts to fit tokenizer
            max_features: Maximum number of features (words) to keep
            
        Returns:
            Fitted tokenizer
        """
        if max_features is None:
            max_features = self.config.get('tokenization.max_features', 10000)
        
        oov_token = self.config.get('tokenization.oov_token', '<OOV>')
        
        tokenizer = Tokenizer(
            num_words=max_features,
            oov_token=oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        tokenizer.fit_on_texts(texts)
        self.tokenizer = tokenizer
        
        logger.info(f"Tokenizer created with vocabulary size: {len(tokenizer.word_index)}")
        
        return tokenizer
    
    def texts_to_sequences(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert texts to padded sequences.
        
        Args:
            texts: List of texts to convert
            max_length: Maximum sequence length
            
        Returns:
            Padded sequences array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call create_tokenizer first.")
        
        if max_length is None:
            max_length = self.config.get('tokenization.max_sequence_length', 1000)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=max_length,
            padding='post',
            truncating='post'
        )
        
        logger.info(f"Converted {len(texts)} texts to sequences of length {max_length}")
        
        return padded_sequences
    
    def balance_dataset(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'binary_sentiment',
        strategy: str = 'undersample'
    ) -> pd.DataFrame:
        """
        Balance dataset by adjusting class distribution.
        
        Args:
            df: DataFrame to balance
            target_column: Column containing target labels
            strategy: Balancing strategy ('undersample', 'oversample', or 'none')
            
        Returns:
            Balanced DataFrame
        """
        if strategy == 'none':
            return df
        
        # Get class counts
        class_counts = df[target_column].value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        if strategy == 'undersample':
            # Undersample majority class
            min_count = class_counts.min()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_df = df[df[target_column] == class_label].sample(
                    n=min_count, 
                    random_state=42
                )
                balanced_dfs.append(class_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif strategy == 'oversample':
            # Oversample minority class
            max_count = class_counts.max()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_df = df[df[target_column] == class_label]
                if len(class_df) < max_count:
                    # Oversample with replacement
                    oversampled = class_df.sample(
                        n=max_count, 
                        replace=True, 
                        random_state=42
                    )
                    balanced_dfs.append(oversampled)
                else:
                    balanced_dfs.append(class_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_class_counts = balanced_df[target_column].value_counts()
        logger.info(f"Balanced class distribution: {new_class_counts.to_dict()}")
        
        return balanced_df