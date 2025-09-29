import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set dataset path dari folder lokal
dataset_path = "dataset"

print("=== EKSPLORASI DATA CURAH HUJAN INDIA ===")
print("Sesuai Proposal: Prediksi Curah Hujan Harian di India Menggunakan LSTM")
print()

# 1. Load dataset
print("1. Loading dataset...")
try:
    # Cek file apa saja yang ada di dataset
    import os

    files = os.listdir(dataset_path)
    print(f"Files dalam dataset: {files}")

    # Load file CSV yang benar (rainfall in india 1901-2015.csv)
    csv_files = [f for f in files if "rainfall in india 1901-2015.csv" in f]
    if csv_files:
        data_file = csv_files[0]  # load file yang benar
        df = pd.read_csv(os.path.join(dataset_path, data_file))
        print(f"Dataset berhasil dimuat dari: {data_file}")
        print(f"Shape dataset: {df.shape}")
    else:
        print("File 'rainfall in india 1901-2015.csv' tidak ditemukan!")
        exit()

    files = os.listdir(dataset_path)
    print(f"Files dalam dataset: {files}")

    # Load file CSV (asumsi ada file rainfall data)
    csv_files = [f for f in files if f.endswith(".csv")]
    if csv_files:
        data_file = csv_files[0]  # ambil file CSV pertama
        df = pd.read_csv(os.path.join(dataset_path, data_file))
        print(f"Dataset berhasil dimuat dari: {data_file}")
        print(f"Shape dataset: {df.shape}")
    else:
        print("Tidak ada file CSV ditemukan!")
        exit()

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Informasi dasar dataset
print("\n2. Informasi Dasar Dataset:")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"Kolom-kolom: {list(df.columns)}")

# 3. Preview data
print("\n3. Preview Data (5 baris pertama):")
print(df.head())

print("\n4. Info Dataset:")
print(df.info())

print("\n5. Statistik Deskriptif:")
print(df.describe())

# 6. Missing values
print("\n6. Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\n7. Tipe Data:")
print(df.dtypes)

# Simpan hasil eksplorasi
print("\n=== EKSPLORASI DATA SELESAI ===")
print("Lanjut ke tahap preprocessing data...")
