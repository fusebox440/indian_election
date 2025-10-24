"""
Machine Learning Models Module
Author: Lakshya Khetan
Email: lakshyaketan00@gmail.com

Contains model architectures for sentiment analysis including GloVe and LSTM models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import json

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, Flatten, 
    LSTM, Bidirectional, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ..utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for sentiment analysis models."""
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model for configuration lookup
        """
        self.model_name = model_name
        self.config = config.get_model_config(model_name)
        self.model = None
        self.history = None
        self.tokenizer = None
        
    def build_model(self) -> keras.Model:
        """Build the model architecture. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        training_config = self.config.get('training', {})
        
        optimizer = Adam(learning_rate=training_config.get('learning_rate', 0.001))
        loss = training_config.get('loss', 'binary_crossentropy')
        metrics = training_config.get('metrics', ['accuracy'])
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer: {optimizer.__class__.__name__}")
    
    def get_callbacks(self, model_path: str) -> List[keras.callbacks.Callback]:
        """Get training callbacks."""
        callbacks = []
        
        training_config = self.config.get('training', {})
        
        # Early stopping
        if training_config.get('early_stopping', True):
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=training_config.get('patience', 5),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_path: Optional[str] = None
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_path: Path to save best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        training_config = self.config.get('training', {})
        
        # Set default model path
        if model_path is None:
            model_paths = config.get_model_paths()
            model_path = f"{model_paths['saved']}/{self.model_name}_best.h5"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare validation data
        if X_val is None or y_val is None:
            validation_split = training_config.get('validation_split', 0.2)
            validation_data = None
        else:
            validation_split = None
            validation_data = (X_val, y_val)
        
        # Train model
        logger.info(f"Starting training for {self.model_name}")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=training_config.get('batch_size', 32),
            epochs=training_config.get('epochs', 10),
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=self.get_callbacks(model_path),
            verbose=1
        )
        
        logger.info(f"Training completed for {self.model_name}")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_prob.tolist()
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (binary predictions, prediction probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict(X)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

class GloVeModel(BaseModel):
    """GloVe embedding-based sentiment analysis model."""
    
    def __init__(self):
        """Initialize GloVe model."""
        super().__init__('glove')
        self.embedding_matrix = None
        self.word_index = None
    
    def load_glove_embeddings(self, embeddings_path: str, word_index: Dict[str, int]) -> np.ndarray:
        """
        Load GloVe embeddings and create embedding matrix.
        
        Args:
            embeddings_path: Path to GloVe embeddings file
            word_index: Word index from tokenizer
            
        Returns:
            Embedding matrix
        """
        logger.info(f"Loading GloVe embeddings from {embeddings_path}")
        
        embedding_dim = self.config.get('embedding_dim', 100)
        embeddings_index = {}
        
        # Load embeddings
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        except FileNotFoundError:
            logger.error(f"GloVe embeddings file not found: {embeddings_path}")
            raise
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors")
        
        # Create embedding matrix
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        self.embedding_matrix = embedding_matrix
        self.word_index = word_index
        
        logger.info(f"Created embedding matrix of shape {embedding_matrix.shape}")
        
        return embedding_matrix
    
    def build_model(self) -> keras.Model:
        """Build GloVe-based model architecture."""
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix not loaded. Call load_glove_embeddings() first.")
        
        # Model architecture config
        arch_config = self.config.get('architecture', {})
        sequence_length = self.config.get('sequence_length', 1000)
        vocab_size, embedding_dim = self.embedding_matrix.shape
        
        # Build model
        inputs = Input(shape=(sequence_length,), dtype='int32')
        
        # Embedding layer
        embedding = Embedding(
            vocab_size,
            embedding_dim,
            weights=[self.embedding_matrix],
            input_length=sequence_length,
            trainable=self.config.get('trainable', False)
        )(inputs)
        
        # Dense layers
        x = embedding
        for units in arch_config.get('dense_layers', [100, 50, 50]):
            x = Dense(
                units,
                activation=arch_config.get('activation', 'relu'),
                kernel_regularizer=regularizers.l2(arch_config.get('l2_regularization', 0.002))
            )(x)
            x = Dropout(arch_config.get('dropout_rate', 0.5))(x)
        
        # Flatten before final layers
        x = Flatten()(x)
        
        # Additional dense layers after flattening
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(1, activation=arch_config.get('final_activation', 'sigmoid'))(x)
        
        self.model = Model(inputs, outputs)
        
        logger.info("GloVe model architecture built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
        
        return self.model

class LSTMModel(BaseModel):
    """Bidirectional LSTM sentiment analysis model."""
    
    def __init__(self):
        """Initialize LSTM model."""
        super().__init__('lstm')
    
    def build_model(self) -> keras.Model:
        """Build LSTM model architecture."""
        # Model config
        embedding_dim = self.config.get('embedding_dim', 128)
        lstm_units = self.config.get('lstm_units', 64)
        max_features = self.config.get('max_features', 10000)
        sequence_length = self.config.get('sequence_length', 1000)
        
        # Architecture config
        arch_config = self.config.get('architecture', {})
        
        # Build model
        model = Sequential([
            # Embedding layer
            Embedding(
                max_features,
                embedding_dim,
                input_length=sequence_length
            ),
            
            # Bidirectional LSTM
            Bidirectional(LSTM(
                lstm_units,
                dropout=arch_config.get('dropout', 0.3),
                recurrent_dropout=arch_config.get('recurrent_dropout', 0.3),
                return_sequences=False
            )),
            
            # Dense layers
            *[Dense(units, activation='relu') for units in arch_config.get('dense_layers', [64, 32])],
            
            # Output layer
            Dense(1, activation=arch_config.get('final_activation', 'sigmoid'))
        ])
        
        self.model = model
        
        logger.info("LSTM model architecture built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
        
        return self.model

class ModelFactory:
    """Factory class for creating models."""
    
    @staticmethod
    def create_model(model_type: str) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('glove' or 'lstm')
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'glove':
            return GloVeModel()
        elif model_type.lower() == 'lstm':
            return LSTMModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class ModelTrainer:
    """Handles training and evaluation of multiple models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.results = {}
    
    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a model.
        
        Args:
            model_type: Type of model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            **kwargs: Additional arguments for model
            
        Returns:
            Training and evaluation results
        """
        logger.info(f"Training {model_type} model")
        
        # Create model
        model = ModelFactory.create_model(model_type)
        
        # Handle GloVe-specific setup
        if model_type.lower() == 'glove':
            if 'word_index' not in kwargs or 'embeddings_path' not in kwargs:
                raise ValueError("GloVe model requires 'word_index' and 'embeddings_path'")
            
            model.load_glove_embeddings(kwargs['embeddings_path'], kwargs['word_index'])
        
        # Train model
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        results = model.evaluate(X_test, y_test)
        results['training_history'] = {
            'loss': history.history['loss'],
            'accuracy': history.history.get('accuracy', []),
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', [])
        }
        
        # Store model and results
        self.models[model_type] = model
        self.results[model_type] = results
        
        logger.info(f"{model_type} model training completed")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['classification_report']['1']['precision'],
                'Recall': results['classification_report']['1']['recall'],
                'F1-Score': results['classification_report']['1']['f1-score']
            })
        
        return pd.DataFrame(comparison_data).round(4)