"""
LSTM Model for SOC Estimation
Battery Management System Project

Uses TensorFlow/Keras LSTM for sequence-based SOC prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class LSTMEstimator:
    def __init__(self, seq_length=50, n_features=3, lstm_units=64, dropout_rate=0.2):
        """
        Initialize LSTM Estimator for SOC prediction.
        
        Args:
            seq_length: Number of time steps in input sequence
            n_features: Number of input features (Current, Voltage, Temperature)
            lstm_units: Number of LSTM units per layer
            dropout_rate: Dropout rate for regularization
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build and compile LSTM model architecture."""
        self.model = Sequential([
            # First LSTM layer with return sequences for stacking
            LSTM(self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.seq_length, self.n_features)),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            
            # Dense layers for output
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # SOC output (0-1 range after scaling)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(self.model.summary())
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=64, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features [samples, seq_length, n_features]
            y_train: Training targets [samples]
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Maximum training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Predict SOC values.
        
        Args:
            X: Input features [samples, seq_length, n_features]
            
        Returns:
            Predicted SOC values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Tuple of (loss, mae)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save(self, filepath):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick test of model architecture
    print("Testing LSTM Model Architecture...")
    
    lstm = LSTMEstimator(seq_length=50, n_features=3)
    lstm.build_model()
    
    # Test with dummy data
    dummy_X = np.random.randn(100, 50, 3)
    dummy_y = np.random.rand(100)
    
    print(f"\nInput shape: {dummy_X.shape}")
    print(f"Output shape: {dummy_y.shape}")
    
    # Quick training test (2 epochs only)
    lstm.train(dummy_X, dummy_y, epochs=2, verbose=1)
    
    predictions = lstm.predict(dummy_X[:5])
    print(f"\nSample predictions: {predictions}")
    print("\nLSTM Model test completed successfully!")
