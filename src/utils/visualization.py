"""
Visualization Utilities
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Provides visualization functions for sentiment analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from wordcloud import WordCloud

from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """Creates visualizations for sentiment analysis results."""
    
    def __init__(self):
        """Initialize visualizer with configuration."""
        viz_config = config.get('visualization', {})
        
        # Set style
        plt.style.use(viz_config.get('style', 'seaborn'))
        
        # Set default parameters
        self.figure_size = viz_config.get('figure_size', [12, 8])
        self.dpi = viz_config.get('dpi', 300)
        self.color_palette = viz_config.get('color_palette', 'Set2')
        self.font_size = viz_config.get('font_size', 12)
        self.save_format = viz_config.get('save_format', 'png')
        
        # Set font size
        plt.rcParams.update({'font.size': self.font_size})
    
    def plot_sentiment_distribution(
        self, 
        df: pd.DataFrame, 
        sentiment_column: str = 'binary_sentiment',
        title: str = "Sentiment Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sentiment distribution as a bar chart.
        
        Args:
            df: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment labels
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Count sentiments
        sentiment_counts = df[sentiment_column].value_counts()
        labels = ['Negative', 'Positive']
        colors = sns.color_palette(self.color_palette, len(labels))
        
        # Create bar plot
        bars = ax.bar(labels, [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)], 
                     color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')
        ax.set_ylabel('Number of Tweets')
        ax.set_xlabel('Sentiment')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sentiment distribution plot saved to {save_path}")
        
        return fig
    
    def plot_party_comparison(
        self, 
        comparison_df: pd.DataFrame,
        title: str = "Party Sentiment Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison plot for different parties.
        
        Args:
            comparison_df: DataFrame with party comparison data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        parties = comparison_df['Party']
        positive_pct = comparison_df['Positive %']
        negative_pct = comparison_df['Negative %']
        
        colors = sns.color_palette(self.color_palette, len(parties))
        
        # Positive sentiment comparison
        bars1 = ax1.bar(parties, positive_pct, color=colors)
        ax1.set_title('Positive Sentiment by Party', fontweight='bold')
        ax1.set_ylabel('Percentage of Positive Tweets')
        ax1.set_ylim(0, max(positive_pct) * 1.1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Negative sentiment comparison
        bars2 = ax2.bar(parties, negative_pct, color=colors, alpha=0.7)
        ax2.set_title('Negative Sentiment by Party', fontweight='bold')
        ax2.set_ylabel('Percentage of Negative Tweets')
        ax2.set_ylim(0, max(negative_pct) * 1.1)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Party comparison plot saved to {save_path}")
        
        return fig
    
    def plot_training_history(
        self, 
        history: Dict[str, List[float]],
        title: str = "Model Training History",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot model training history.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=self.dpi)
        
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Plot accuracy
        ax1.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
        if 'val_accuracy' in history:
            ax1.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
        
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(epochs, history['loss'], 'bo-', label='Training Loss')
        if 'val_loss' in history:
            ax2.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray,
        class_names: List[str] = ['Negative', 'Positive'],
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return fig
    
    def create_wordcloud(
        self, 
        texts: List[str],
        title: str = "Word Cloud",
        save_path: Optional[str] = None,
        max_words: int = 100,
        background_color: str = 'white'
    ) -> plt.Figure:
        """
        Create a word cloud from text data.
        
        Args:
            texts: List of texts
            title: Plot title
            save_path: Path to save the plot
            max_words: Maximum number of words to display
            background_color: Background color
            
        Returns:
            Matplotlib figure
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis'
        ).generate(combined_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Word cloud saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(
        self, 
        results: Dict[str, Any],
        output_dir: str
    ):
        """
        Create a comprehensive visualization report.
        
        Args:
            results: Results from election prediction analysis
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating comprehensive visualization report in {output_dir}")
        
        # Party comparison plot
        comparison_fig = self.plot_party_comparison(
            results['comparison_table'],
            save_path=str(output_path / 'party_comparison.png')
        )
        plt.close(comparison_fig)
        
        # Individual party sentiment distributions
        for party_name, party_result in results['party_results'].items():
            sentiment_fig = self.plot_sentiment_distribution(
                party_result['predictions'],
                title=f"{party_name.upper()} Sentiment Distribution",
                save_path=str(output_path / f'{party_name}_sentiment_distribution.png')
            )
            plt.close(sentiment_fig)
            
            # Word cloud for each party
            if 'clean_text' in party_result['predictions'].columns:
                wordcloud_fig = self.create_wordcloud(
                    party_result['predictions']['clean_text'].tolist(),
                    title=f"{party_name.upper()} Word Cloud",
                    save_path=str(output_path / f'{party_name}_wordcloud.png')
                )
                plt.close(wordcloud_fig)
        
        logger.info("Comprehensive visualization report created successfully")

class ModelVisualizer:
    """Specialized visualizer for model performance."""
    
    def __init__(self):
        """Initialize model visualizer."""
        self.base_visualizer = SentimentVisualizer()
    
    def plot_model_comparison(
        self, 
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of different models.
        
        Args:
            comparison_df: DataFrame with model comparison data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.base_visualizer.figure_size, 
                              dpi=self.base_visualizer.dpi)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df['Model']))
        width = 0.2
        
        colors = sns.color_palette(self.base_visualizer.color_palette, len(metrics))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, comparison_df[metric], width, 
                  label=metric, color=colors[i])
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.base_visualizer.save_format, 
                       dpi=self.base_visualizer.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig