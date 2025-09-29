import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


class RainfallLSTM:
    def __init__(self, seq_length=12, n_features=1):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None

    def build_model(self, units1=64, units2=32, dense_units=16, dropout_rate=0.2):
        """Build LSTM model according to proposal specifications"""
        self.model = Sequential(
            [
                LSTM(
                    units1,
                    return_sequences=True,
                    input_shape=(self.seq_length, self.n_features),
                ),
                Dropout(dropout_rate),
                LSTM(units2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(dense_units, activation="relu"),
                Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        print("LSTM model built successfully with proposal specifications")
        return self.model

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        patience=10,
        save_path="models/rainfall_lstm.h5",
    ):
        """Train the model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True),
        ]

        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )

        print("Model training completed")
        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def load_model(self, model_path):
        """Load saved model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss}")
        return loss
