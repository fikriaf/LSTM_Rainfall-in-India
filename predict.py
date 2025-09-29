from data_loader import DataLoader
from model import RainfallLSTM
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Initialize data loader
    loader = DataLoader()

    # Preprocess data
    ts_data = loader.preprocess_data(subdivision="ANDAMAN & NICOBAR ISLANDS")

    # Create sequences
    seq_length = 12
    X, y = loader.create_sequences(ts_data, seq_length=seq_length)

    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)

    # Load trained model
    lstm_model = RainfallLSTM(seq_length=seq_length)
    lstm_model.load_model("models/rainfall_lstm.h5")

    # Make predictions
    predictions = lstm_model.predict(X_test)

    # Inverse transform predictions and actual values
    predictions_inv = loader.inverse_transform(predictions)
    y_test_inv = loader.inverse_transform(y_test)

    # Calculate metrics
    mse = np.mean((predictions_inv - y_test_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_inv - y_test_inv))

    # Calculate R² score
    ss_res = np.sum((y_test_inv - predictions_inv) ** 2)
    ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual Rainfall", color="blue")
    plt.plot(predictions_inv, label="Predicted Rainfall", color="red")
    plt.title("Rainfall Prediction vs Actual")
    plt.xlabel("Time Steps")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plt.savefig("plots/predictions.png")
    plt.show()

    print("Prediction and evaluation completed!")


if __name__ == "__main__":
    main()
