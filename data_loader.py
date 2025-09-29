import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import calendar


class DataLoader:
    def __init__(self, data_path="dataset/rainfall in india 1901-2015.csv"):
        self.data_path = data_path
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """Load the rainfall dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        print(f"Dataset loaded with shape: {self.data.shape}")
        return self.data

    def preprocess_data(self, subdivision="ANDAMAN & NICOBAR ISLANDS"):
        """Preprocess data for a specific subdivision and convert to daily"""
        if self.data is None:
            self.load_data()

        # Filter by subdivision
        sub_data = self.data[self.data["SUBDIVISION"] == subdivision].copy()

        # Drop unnecessary columns
        monthly_cols = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
        sub_data = sub_data[["YEAR"] + monthly_cols]

        # Handle missing values
        sub_data = sub_data.fillna(method="ffill").fillna(method="bfill")

        # Convert monthly to daily using simple division (actual days per month)
        daily_data = []
        for _, row in sub_data.iterrows():
            year = int(row["YEAR"])
            for month_idx, month in enumerate(monthly_cols):
                month_num = month_idx + 1
                rainfall = row[month]

                # Get actual number of days in month
                days_in_month = calendar.monthrange(year, month_num)[1]
                daily_rainfall = rainfall / days_in_month

                # Generate daily entries
                for day in range(1, days_in_month + 1):
                    date = pd.Timestamp(year=year, month=month_num, day=day)
                    daily_data.append({"date": date, "rainfall": daily_rainfall})

        ts_data = pd.DataFrame(daily_data)
        ts_data = ts_data.sort_values("date").reset_index(drop=True)

        return ts_data

    def create_sequences(self, data, seq_length=12):
        """Create sequences for LSTM input"""
        rainfall_values = data["rainfall"].values.reshape(-1, 1)

        # Normalize
        scaled_data = self.scaler.fit_transform(rainfall_values)

        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i : i + seq_length])
            y.append(scaled_data[i + seq_length])

        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def inverse_transform(self, scaled_values):
        """Inverse transform scaled values back to original scale"""
        return self.scaler.inverse_transform(scaled_values)
