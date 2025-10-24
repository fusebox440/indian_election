"""
Prediction Module
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Handles making predictions on new data using trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json

from tensorflow import keras
from ..data.preprocessor import TextPreprocessor
from ..utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Handles sentiment prediction on new tweet data."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model
            tokenizer_path: Path to saved tokenizer (optional)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        
        self._load_model()
        if tokenizer_path:
            self._load_tokenizer()
    
    def _load_model(self):
        """Load trained model from file."""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer from file."""
        try:
            with open(self.tokenizer_path, 'r') as f:
                tokenizer_json = f.read()
            
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            self.preprocessor.tokenizer = self.tokenizer
            
            logger.info(f"Tokenizer loaded successfully from {self.tokenizer_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")
            raise
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess text
        clean_text = self.preprocessor.clean_text(text)
        
        if not clean_text.strip():
            return {
                'original_text': text,
                'clean_text': clean_text,
                'prediction': 0,
                'probability': 0.5,
                'sentiment': 'neutral'
            }
        
        # Convert to sequence
        if self.tokenizer is None:
            # If no tokenizer available, create one for this text
            self.preprocessor.create_tokenizer([clean_text])
        
        sequence = self.preprocessor.texts_to_sequences([clean_text])
        
        # Make prediction
        probability = self.model.predict(sequence)[0][0]
        prediction = 1 if probability > 0.5 else 0
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        return {
            'original_text': text,
            'clean_text': clean_text,
            'prediction': int(prediction),
            'probability': float(probability),
            'sentiment': sentiment
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Making predictions for {len(texts)} texts")
        
        # Preprocess all texts
        clean_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(clean_texts) if text.strip()]
        valid_texts = [clean_texts[i] for i in valid_indices]
        
        if not valid_texts:
            # Return neutral predictions for all texts
            return [
                {
                    'original_text': text,
                    'clean_text': clean_texts[i],
                    'prediction': 0,
                    'probability': 0.5,
                    'sentiment': 'neutral'
                }
                for i, text in enumerate(texts)
            ]
        
        # Create tokenizer if needed
        if self.tokenizer is None:
            self.preprocessor.create_tokenizer(valid_texts)
        
        # Convert to sequences
        sequences = self.preprocessor.texts_to_sequences(valid_texts)
        
        # Make predictions
        probabilities = self.model.predict(sequences).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        # Prepare results
        results = []
        valid_idx = 0
        
        for i, text in enumerate(texts):
            if i in valid_indices:
                prob = float(probabilities[valid_idx])
                pred = int(predictions[valid_idx])
                sentiment = 'positive' if pred == 1 else 'negative'
                valid_idx += 1
            else:
                prob = 0.5
                pred = 0
                sentiment = 'neutral'
            
            results.append({
                'original_text': text,
                'clean_text': clean_texts[i],
                'prediction': pred,
                'probability': prob,
                'sentiment': sentiment
            })
        
        logger.info(f"Predictions completed for {len(texts)} texts")
        
        return results
    
    def predict_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text',
        output_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Predict sentiment for texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_column: Name of column containing text data
            output_columns: List of column names for output
            
        Returns:
            DataFrame with prediction results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        logger.info(f"Making predictions for DataFrame with {len(df)} rows")
        
        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts)
        
        # Create results DataFrame
        result_df = df.copy()
        
        # Default output columns
        if output_columns is None:
            output_columns = ['clean_text', 'prediction', 'probability', 'sentiment']
        
        # Add prediction columns
        for col in output_columns:
            if col in predictions[0]:
                result_df[col] = [pred[col] for pred in predictions]
        
        return result_df

class ElectionPredictor:
    """Specialized predictor for election sentiment analysis."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize election predictor.
        
        Args:
            model_path: Path to saved model
            tokenizer_path: Path to saved tokenizer
        """
        self.predictor = SentimentPredictor(model_path, tokenizer_path)
        self.party_keywords = config.get('data_collection.keywords', {})
    
    def predict_party_sentiment(
        self, 
        party_data: pd.DataFrame, 
        party_name: str,
        text_column: str = 'clean_text'
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a political party's tweets.
        
        Args:
            party_data: DataFrame containing party tweets
            party_name: Name of the party
            text_column: Column containing tweet text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {party_name}: {len(party_data)} tweets")
        
        # Make predictions
        predictions_df = self.predictor.predict_dataframe(party_data, text_column)
        
        # Calculate statistics
        total_tweets = len(predictions_df)
        positive_tweets = (predictions_df['prediction'] == 1).sum()
        negative_tweets = (predictions_df['prediction'] == 0).sum()
        
        positive_percentage = (positive_tweets / total_tweets) * 100
        negative_percentage = (negative_tweets / total_tweets) * 100
        
        avg_probability = predictions_df['probability'].mean()
        
        results = {
            'party': party_name,
            'total_tweets': total_tweets,
            'positive_tweets': int(positive_tweets),
            'negative_tweets': int(negative_tweets),
            'positive_percentage': float(positive_percentage),
            'negative_percentage': float(negative_percentage),
            'average_probability': float(avg_probability),
            'predictions': predictions_df
        }
        
        logger.info(f"Sentiment analysis completed for {party_name}")
        logger.info(f"Positive: {positive_tweets} ({positive_percentage:.1f}%)")
        logger.info(f"Negative: {negative_tweets} ({negative_percentage:.1f}%)")
        
        return results
    
    def compare_parties(
        self, 
        party_data: Dict[str, pd.DataFrame],
        text_column: str = 'clean_text'
    ) -> Dict[str, Any]:
        """
        Compare sentiment across multiple parties.
        
        Args:
            party_data: Dictionary mapping party names to DataFrames
            text_column: Column containing tweet text
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing sentiment across {len(party_data)} parties")
        
        party_results = {}
        comparison_data = []
        
        # Analyze each party
        for party_name, data in party_data.items():
            results = self.predict_party_sentiment(data, party_name, text_column)
            party_results[party_name] = results
            
            comparison_data.append({
                'Party': party_name,
                'Total Tweets': results['total_tweets'],
                'Positive Tweets': results['positive_tweets'],
                'Negative Tweets': results['negative_tweets'],
                'Positive %': results['positive_percentage'],
                'Negative %': results['negative_percentage'],
                'Avg Probability': results['average_probability']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Determine predicted winner (highest positive percentage)
        winner = comparison_df.loc[comparison_df['Positive %'].idxmax(), 'Party']
        
        comparison_results = {
            'party_results': party_results,
            'comparison_table': comparison_df,
            'predicted_winner': winner,
            'analysis_summary': self._generate_summary(comparison_df)
        }
        
        logger.info(f"Party comparison completed. Predicted winner: {winner}")
        
        return comparison_results
    
    def _generate_summary(self, comparison_df: pd.DataFrame) -> str:
        """Generate analysis summary text."""
        total_tweets = comparison_df['Total Tweets'].sum()
        avg_positive = comparison_df['Positive %'].mean()
        
        winner = comparison_df.loc[comparison_df['Positive %'].idxmax()]
        runner_up = comparison_df.loc[comparison_df['Positive %'].nlargest(2).index[1]]
        
        summary = f"""
Election Sentiment Analysis Summary:
- Total tweets analyzed: {total_tweets:,}
- Average positive sentiment: {avg_positive:.1f}%
- Predicted winner: {winner['Party']} ({winner['Positive %']:.1f}% positive)
- Runner-up: {runner_up['Party']} ({runner_up['Positive %']:.1f}% positive)
- Margin: {winner['Positive %'] - runner_up['Positive %']:.1f} percentage points

The analysis suggests a close contest with {winner['Party']} having a slight advantage
based on positive sentiment in social media discussions.
        """.strip()
        
        return summary
    
    def save_predictions(
        self, 
        results: Dict[str, Any], 
        output_path: str
    ):
        """
        Save prediction results to files.
        
        Args:
            results: Results from compare_parties
            output_path: Base path for output files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_file = output_dir / "party_comparison.csv"
        results['comparison_table'].to_csv(comparison_file, index=False)
        
        # Save individual party predictions
        for party_name, party_result in results['party_results'].items():
            party_file = output_dir / f"{party_name}_predictions.csv"
            party_result['predictions'].to_csv(party_file, index=False)
        
        # Save summary
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(results['analysis_summary'])
        
        # Save full results as JSON
        json_file = output_dir / "full_results.json"
        
        # Prepare JSON-serializable data
        json_results = {
            'comparison_table': results['comparison_table'].to_dict('records'),
            'predicted_winner': results['predicted_winner'],
            'analysis_summary': results['analysis_summary'],
            'party_stats': {
                party: {
                    'party': stats['party'],
                    'total_tweets': stats['total_tweets'],
                    'positive_tweets': stats['positive_tweets'],
                    'negative_tweets': stats['negative_tweets'],
                    'positive_percentage': stats['positive_percentage'],
                    'negative_percentage': stats['negative_percentage'],
                    'average_probability': stats['average_probability']
                }
                for party, stats in results['party_results'].items()
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Prediction results saved to {output_dir}")