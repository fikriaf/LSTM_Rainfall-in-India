"""
Main script for Rainfall Prediction in India using LSTM
Modular implementation based on the project proposal
"""

import argparse
from data_loader import DataLoader
from model import RainfallLSTM
import os


def run_training(subdivision="ANDAMAN & NICOBAR ISLANDS", seq_length=30):
    """Run the training pipeline"""
    print("Starting training pipeline...")

    # Data loading and preprocessing
    loader = DataLoader()
    ts_data = loader.preprocess_data(subdivision=subdivision)
    X, y = loader.create_sequences(ts_data, seq_length=seq_length)
    X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)

    # Model building and training
    lstm_model = RainfallLSTM(seq_length=seq_length)
    lstm_model.build_model(units1=64, units2=32, dense_units=16, dropout_rate=0.2)

    lstm_model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        batch_size=32,
        patience=10,
        save_path="models/rainfall_lstm.h5",
    )

    print("Training completed!")


def run_prediction(subdivision="ANDAMAN & NICOBAR ISLANDS", seq_length=30):
    """Run the prediction pipeline"""
    print("Starting prediction pipeline...")

    # Data loading
    loader = DataLoader()
    ts_data = loader.preprocess_data(subdivision=subdivision)
    X, y = loader.create_sequences(ts_data, seq_length=seq_length)
    _, X_test, _, y_test = loader.split_data(X, y, test_size=0.2)

    # Load model and predict
    lstm_model = RainfallLSTM(seq_length=seq_length)
    lstm_model.load_model("models/rainfall_lstm.h5")

    predictions = lstm_model.predict(X_test)
    loss = lstm_model.evaluate(X_test, y_test)

    print(f"Prediction completed with test loss: {loss}")


def main():
    parser = argparse.ArgumentParser(description="Rainfall Prediction using LSTM")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "both"],
        default="both",
        help="Mode to run",
    )
    parser.add_argument(
        "--subdivision",
        default="ANDAMAN & NICOBAR ISLANDS",
        help="Subdivision to use for prediction",
    )
    parser.add_argument(
        "--seq_length", type=int, default=30, help="Sequence length for LSTM"
    )

    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        run_training(args.subdivision, args.seq_length)

    if args.mode in ["predict", "both"]:
        if not os.path.exists("models/rainfall_lstm.h5"):
            print("Model not found! Please run training first.")
            return
        run_prediction(args.subdivision, args.seq_length)


if __name__ == "__main__":
    main()
