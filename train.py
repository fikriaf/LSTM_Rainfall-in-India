from data_loader import DataLoader
from model import RainfallLSTM
import matplotlib.pyplot as plt


def main():
    # Initialize data loader
    loader = DataLoader()

    # Preprocess data for a subdivision
    ts_data = loader.preprocess_data(subdivision="ANDAMAN & NICOBAR ISLANDS")

    # Create sequences
    seq_length = 30  # Use 30 days of data to predict next day
    X, y = loader.create_sequences(ts_data, seq_length=seq_length)

    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)

    # Build model
    lstm_model = RainfallLSTM(seq_length=seq_length)
    lstm_model.build_model()  # Use default parameters as per proposal

    # Train model
    history = lstm_model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        batch_size=32,
        patience=10,
        save_path="models/rainfall_lstm.h5",
    )

    # Evaluate
    loss = lstm_model.evaluate(X_test, y_test)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/training_history.png")
    plt.show()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
