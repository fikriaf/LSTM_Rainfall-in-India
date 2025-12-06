# 📘 PENJELASAN LENGKAP NOTEBOOK COMPARISON 13 MODELS
## File: comparison_13_models.ipynb

**Tujuan:** Membandingkan performa 13 model LSTM dengan konfigurasi berbeda untuk menemukan hyperparameter terbaik.

---

# CELL 1: IMPORT LIBRARIES

```python
import pandas as pd                    # Manipulasi data tabular (DataFrame)
import numpy as np                     # Operasi numerik dan array
import matplotlib.pyplot as plt        # Visualisasi grafik dasar
import seaborn as sns                  # Visualisasi statistik lanjutan
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import calendar                        # Untuk menghitung jumlah hari per bulan
import warnings
warnings.filterwarnings('ignore')      # Sembunyikan warning messages

# Set style visualisasi
sns.set_style('whitegrid')             # Grid putih untuk plot
plt.rcParams['figure.figsize'] = (14, 8)  # Ukuran default plot 14x8 inch

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
```

**Penjelasan Import:**

| Library | Fungsi |
|---------|--------|
| `pandas` | Membaca CSV, manipulasi DataFrame |
| `numpy` | Operasi array, perhitungan numerik |
| `matplotlib.pyplot` | Membuat grafik bar, line, scatter |
| `seaborn` | Styling plot yang lebih bagus |
| `MinMaxScaler` | Normalisasi data ke range 0-1 |
| `LabelEncoder` | Mengubah nama wilayah (string) → angka |
| `mean_squared_error` | Menghitung MSE |
| `mean_absolute_error` | Menghitung MAE |
| `r2_score` | Menghitung R² (koefisien determinasi) |
| `load_model` | Memuat model .h5 yang sudah ditraining |
| `calendar` | Mengetahui jumlah hari dalam bulan tertentu |

**Output:**
```
Libraries imported successfully!
TensorFlow version: 2.20.0
```

---

# CELL 2: KONFIGURASI MODEL

```python
# Path dataset (kosong = folder yang sama dengan notebook)
DATASET_PATH = ""

# Daftar 13 file model yang akan dibandingkan
MODEL_FILES = [
    'exp_baseline_(adam,_lr=0.001,_bs=64).h5',      # Model 1: Baseline
    'exp_deep_architecture_(128-64-32).h5',         # Model 2: Arsitektur dalam
    'exp_high_dropout_(0.5).h5',                    # Model 3: Dropout tinggi
    'exp_high_lr_(adam,_lr=0.01).h5',               # Model 4: Learning rate tinggi
    'exp_large_batch_(bs=128).h5',                  # Model 5: Batch besar
    'exp_long_sequence_(60_days).h5',               # Model 6: Sequence panjang
    'exp_low_lr_(adam,_lr=0.0001).h5',              # Model 7: Learning rate rendah
    'exp_no_dropout.h5',                            # Model 8: Tanpa dropout
    'exp_rmsprop_optimizer.h5',                     # Model 9: RMSprop optimizer
    'exp_sgd_optimizer.h5',                         # Model 10: SGD optimizer
    'exp_short_sequence_(15_days).h5',              # Model 11: Sequence pendek
    'exp_simple_architecture_(32-16-8).h5',         # Model 12: Arsitektur sederhana
    'exp_small_batch_(bs=32).h5'                    # Model 13: Batch kecil
]

# Nama pendek untuk visualisasi
MODEL_NAMES = [
    'Baseline', 'Deep Arch', 'High Dropout', 'High LR', 'Large Batch',
    'Long Seq (60d)', 'Low LR', 'No Dropout', 'RMSprop', 'SGD',
    'Short Seq (15d)', 'Simple Arch', 'Small Batch'
]

print(f"Total models to compare: {len(MODEL_FILES)}")
```

**Penjelasan 13 Eksperimen:**

| No | Model | Konfigurasi | Tujuan Eksperimen |
|----|-------|-------------|-------------------|
| 1 | Baseline | Adam, lr=0.001, bs=64 | Model referensi standar |
| 2 | Deep Arch | 128-64-32 units | Test: lebih dalam = lebih baik? |
| 3 | High Dropout | dropout=0.5 | Test: regularisasi agresif |
| 4 | High LR | lr=0.01 | Test: learning rate 10x lebih tinggi |
| 5 | Large Batch | bs=128 | Test: batch 2x lebih besar |
| 6 | Long Seq | 60 hari | Test: konteks temporal 2x lebih panjang |
| 7 | Low LR | lr=0.0001 | Test: learning rate 10x lebih rendah |
| 8 | No Dropout | dropout=0.0 | Test: tanpa regularisasi |
| 9 | RMSprop | RMSprop optimizer | Test: optimizer alternatif untuk RNN |
| 10 | SGD | SGD optimizer | Test: optimizer klasik |
| 11 | Short Seq | 15 hari | Test: konteks temporal lebih pendek |
| 12 | Simple Arch | 32-16-8 units | Test: arsitektur lebih sederhana |
| 13 | Small Batch | bs=32 | Test: batch lebih kecil |

**Output:**
```
Total models to compare: 13
```

---

# CELL 3: LOAD DAN PREPROCESS DATA

```python
def load_and_preprocess_data():
    """Load dan preprocess data"""
    print("Loading dataset...")
    data = pd.read_csv(DATASET_PATH + 'rainfall in india 1901-2015.csv')
    print(f"Dataset shape: {data.shape}")
```

**`pd.read_csv()`** membaca file CSV ke DataFrame.
- **Dataset shape: (4116, 19)** = 4116 baris × 19 kolom
- Baris = kombinasi subdivision × tahun (36 wilayah × ~115 tahun)
- Kolom = SUBDIVISION, YEAR, JAN-DEC, ANNUAL, dll.

```python
    monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
```

**`monthly_cols`** = daftar 12 kolom curah hujan bulanan yang akan diproses.

```python
    # Encode subdivision
    encoder = LabelEncoder()
    data['SUB_ID'] = encoder.fit_transform(data['SUBDIVISION'])
```

**LabelEncoder** mengubah nama wilayah (string) menjadi angka:
```
'ANDAMAN & NICOBAR ISLANDS' → 0
'ARUNACHAL PRADESH' → 1
'ASSAM & MEGHALAYA' → 2
...
'WEST UTTAR PRADESH' → 35
```
**Mengapa perlu?** Machine learning butuh input numerik, bukan string.

```python
    # Filter dan preprocess
    data_filtered = data[['SUBDIVISION', 'SUB_ID', 'YEAR'] + monthly_cols]
    data_filtered = data_filtered.fillna(method='ffill').fillna(method='bfill')
```

**Handle Missing Values:**
- `fillna(method='ffill')` = Forward Fill: isi nilai kosong dengan nilai sebelumnya
- `fillna(method='bfill')` = Backward Fill: isi nilai kosong dengan nilai sesudahnya

**Contoh:**
```
Sebelum: [10, NaN, NaN, 15, NaN]
ffill:   [10, 10,  10,  15, 15]
bfill:   [10, 10,  10,  15, 15]  (jika masih ada NaN di awal)
```

```python
    # Convert ke daily data
    daily_data = []
    subdivisions = data_filtered['SUBDIVISION'].unique()
    
    print(f"Converting to daily data for {len(subdivisions)} subdivisions...")
    for sub in subdivisions:
        sub_df = data_filtered[data_filtered['SUBDIVISION'] == sub]
        sub_id = sub_df['SUB_ID'].iloc[0]
        for _, row in sub_df.iterrows():
            year = int(row['YEAR'])
            for month_idx, month in enumerate(monthly_cols):
                month_num = month_idx + 1
                rainfall = row[month]
                days_in_month = calendar.monthrange(year, month_num)[1]
                daily_rainfall = rainfall / days_in_month
                for day in range(1, days_in_month + 1):
                    date = pd.Timestamp(year=year, month=month_num, day=day)
                    daily_data.append({
                        'SUBDIVISION': sub,
                        'SUB_ID': sub_id,
                        'date': date,
                        'rainfall': daily_rainfall
                    })
```

**Konversi Bulanan → Harian:**

**Rumus:** `daily_rainfall = monthly_rainfall / days_in_month`

**Contoh:**
```
Januari 2015: 310 mm total, 31 hari
daily_rainfall = 310 / 31 = 10 mm/hari

Februari 2015: 56 mm total, 28 hari
daily_rainfall = 56 / 28 = 2 mm/hari
```

**`calendar.monthrange(year, month)[1]`** mengembalikan jumlah hari dalam bulan:
- `calendar.monthrange(2015, 1)[1]` = 31 (Januari)
- `calendar.monthrange(2015, 2)[1]` = 28 (Februari non-kabisat)
- `calendar.monthrange(2016, 2)[1]` = 29 (Februari kabisat)

```python
    ts_data = pd.DataFrame(daily_data)
    ts_data = ts_data.sort_values(['SUB_ID', 'date']).reset_index(drop=True)
    
    print(f"Daily data shape: {ts_data.shape}")
    return ts_data

# Execute function
ts_data = load_and_preprocess_data()
print("\nSample data:")
print(ts_data.head())
```

**`sort_values(['SUB_ID', 'date'])`** mengurutkan data:
1. Pertama berdasarkan SUB_ID (wilayah)
2. Kemudian berdasarkan date (tanggal)

**Output:**
```
Loading dataset...
Dataset shape: (4116, 19)
Converting to daily data for 36 subdivisions...
Daily data shape: (1503342, 4)

Sample data:
                 SUBDIVISION  SUB_ID       date  rainfall
0  ANDAMAN & NICOBAR ISLANDS       0 1901-01-01  1.587097
1  ANDAMAN & NICOBAR ISLANDS       0 1901-01-02  1.587097
2  ANDAMAN & NICOBAR ISLANDS       0 1901-01-03  1.587097
3  ANDAMAN & NICOBAR ISLANDS       0 1901-01-04  1.587097
4  ANDAMAN & NICOBAR ISLANDS       0 1901-01-05  1.587097
```

---

# CELL 4: CREATE SEQUENCES FUNCTION

```python
def create_sequences(data, seq_length=30):
    """Create sequences untuk testing"""
    X_all, y_all = [], []
    scaler = MinMaxScaler(feature_range=(0, 1))
```

**Inisialisasi:**
- `X_all` = list untuk menyimpan input sequences
- `y_all` = list untuk menyimpan target values
- `MinMaxScaler(feature_range=(0, 1))` = normalisasi ke range 0-1

```python
    subdivisions = data['SUBDIVISION'].unique()
    print(f"Creating sequences with length {seq_length}...")
    
    for sub in subdivisions:
        sub_df = data[data['SUBDIVISION'] == sub]
        rainfall_values = sub_df['rainfall'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(rainfall_values)
```

**`reshape(-1, 1)`** mengubah array 1D menjadi 2D:
```
Sebelum: [10.5, 8.2, 15.3, 12.1]        # Shape: (4,)
Sesudah: [[10.5], [8.2], [15.3], [12.1]] # Shape: (4, 1)
```
**Mengapa perlu?** MinMaxScaler membutuhkan input 2D.

**`fit_transform()`** = `fit()` + `transform()`:
1. `fit()`: Hitung min dan max dari data
2. `transform()`: Terapkan rumus scaling

**Rumus MinMaxScaler:**
```
x_scaled = (x - x_min) / (x_max - x_min)
```

**Contoh:**
```
Data: [100, 200, 300, 400, 500] mm
min=100, max=500

Scaled: [0.0, 0.25, 0.5, 0.75, 1.0]
```

```python
        for i in range(len(scaled_data) - seq_length):
            X_all.append(scaled_data[i:i+seq_length])
            y_all.append(scaled_data[i+seq_length])
```

**Sliding Window untuk Membuat Sequences:**

```
Data: [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, ...]
seq_length = 3

Iterasi 1 (i=0): X = [d1, d2, d3] → y = d4
Iterasi 2 (i=1): X = [d2, d3, d4] → y = d5
Iterasi 3 (i=2): X = [d3, d4, d5] → y = d6
...
```

**Dengan seq_length=30 (default):**
```
Sequence 1: X = [hari 1-30]  → y = hari 31
Sequence 2: X = [hari 2-31]  → y = hari 32
Sequence 3: X = [hari 3-32]  → y = hari 33
...
```

```python
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"Total sequences: {X_all.shape[0]}")
    return X_all, y_all, scaler
```

**Output Shapes:**
- `X_all.shape`: `(1502262, 30, 1)` = 1.5 juta sequences, masing-masing 30 timesteps × 1 feature
- `y_all.shape`: `(1502262, 1)` = 1.5 juta target values

---

# CELL 5: PREPARE TEST DATA

```python
# Create sequences untuk seq_length = 30 (default)
X, y, scaler = create_sequences(ts_data, seq_length=30)

# Split data (80-20)
split_idx = int(len(X) * 0.8)
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"\nTest set size: {len(X_test)} samples")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
```

**Data Split 80-20:**
```
Total: 1,502,262 sequences
├── Training (80%): 1,201,809 sequences (tidak digunakan di notebook ini)
└── Testing (20%):    300,453 sequences (untuk evaluasi)
```

**Mengapa `shuffle=False` (implisit)?**
- Untuk time series, urutan waktu harus terjaga
- Data test = data masa depan (setelah data training)

**Output:**
```
Creating sequences with length 30...
Total sequences: 1502262

Test set size: 300453 samples
Test set shape: X=(300453, 30, 1), y=(300453, 1)
```

---

# CELL 6: EVALUATE ALL MODELS

```python
def evaluate_model(model_path, X_test, y_test, scaler, seq_length=30):
    """Load model dan evaluasi performanya"""
    try:
        # Load model dari file .h5
        model = load_model(model_path)
```

**`load_model()`** memuat model Keras yang sudah disimpan:
- File .h5 berisi: arsitektur model + weights + optimizer state
- Tidak perlu training ulang, langsung bisa predict

```python
        # Predict
        y_pred_scaled = model.predict(X_test, verbose=0)
```

**`model.predict()`** menghasilkan prediksi:
- Input: `X_test` dengan shape `(300453, 30, 1)`
- Output: `y_pred_scaled` dengan shape `(300453, 1)`
- `verbose=0`: tidak tampilkan progress bar

```python
        # Inverse transform (kembalikan ke skala asli)
        y_test_original = scaler.inverse_transform(y_test)
        y_pred_original = scaler.inverse_transform(y_pred_scaled)
```

**`inverse_transform()`** mengembalikan data dari skala 0-1 ke skala asli (mm):
```
Scaled:   [0.0, 0.25, 0.5, 0.75, 1.0]
Original: [100, 200, 300, 400, 500] mm
```

**Rumus:**
```
x_original = x_scaled × (x_max - x_min) + x_min
```

```python
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_original,
            'actuals': y_test_original
        }
    
    except Exception as e:
        print(f"Error loading {model_path}: {str(e)}")
        return None
```

**Penjelasan 4 Metrics:**

### MSE (Mean Squared Error)
```
MSE = (1/n) × Σ(y_actual - y_pred)²
```
- Memberikan penalti besar untuk error besar (karena dikuadratkan)
- Satuan: mm² (kuadrat dari satuan asli)

### RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
- Satuan sama dengan data asli (mm)
- Lebih mudah diinterpretasi
- **Semakin kecil = semakin baik**

### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|y_actual - y_pred|
```
- Rata-rata kesalahan absolut
- Robust terhadap outlier (tidak dikuadratkan)
- **Semakin kecil = semakin baik**

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
   = 1 - Σ(y_actual - y_pred)² / Σ(y_actual - y_mean)²
```
- Range: 0-1 (bisa negatif jika sangat buruk)
- R² = 1.0: prediksi sempurna
- R² = 0.0: sama dengan prediksi rata-rata
- **Semakin besar = semakin baik**

```python
# Evaluate semua model
print("="*60)
print("EVALUATING ALL MODELS")
print("="*60)

results = []

for model_file, model_name in zip(MODEL_FILES, MODEL_NAMES):
    print(f"\nEvaluating: {model_name}")
    
    # Handle special cases untuk sequence length berbeda
    if 'long_sequence' in model_file:
        X_temp, y_temp, scaler_temp = create_sequences(ts_data, seq_length=60)
        split_idx_temp = int(len(X_temp) * 0.8)
        X_test_temp = X_temp[split_idx_temp:]
        y_test_temp = y_temp[split_idx_temp:]
        result = evaluate_model(model_file, X_test_temp, y_test_temp, scaler_temp, seq_length=60)
    elif 'short_sequence' in model_file:
        X_temp, y_temp, scaler_temp = create_sequences(ts_data, seq_length=15)
        split_idx_temp = int(len(X_temp) * 0.8)
        X_test_temp = X_temp[split_idx_temp:]
        y_test_temp = y_temp[split_idx_temp:]
        result = evaluate_model(model_file, X_test_temp, y_test_temp, scaler_temp, seq_length=15)
    else:
        result = evaluate_model(model_file, X_test, y_test, scaler)
```

**Special Handling untuk Sequence Length Berbeda:**

Model dengan sequence length berbeda membutuhkan data test yang sesuai:
- `long_sequence` (60 hari): buat sequences baru dengan `seq_length=60`
- `short_sequence` (15 hari): buat sequences baru dengan `seq_length=15`
- Model lainnya: gunakan default `seq_length=30`

**Mengapa perlu?** Input shape harus cocok dengan model:
- Model 60 hari: input shape `(batch, 60, 1)`
- Model 30 hari: input shape `(batch, 30, 1)`
- Model 15 hari: input shape `(batch, 15, 1)`

```python
    if result:
        results.append({
            'Model': model_name,
            'MSE': result['mse'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'R²': result['r2']
        })
        print(f"  ✓ MSE: {result['mse']:.6f}")
        print(f"  ✓ RMSE: {result['rmse']:.6f}")
        print(f"  ✓ MAE: {result['mae']:.6f}")
        print(f"  ✓ R²: {result['r2']:.6f}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print("\n" + "="*60)
print("EVALUATION COMPLETED!")
print("="*60)
```

**Output (contoh):**
```
============================================================
EVALUATING ALL MODELS
============================================================

Evaluating: Baseline
  ✓ MSE: 0.432443
  ✓ RMSE: 0.657604
  ✓ MAE: 0.311985
  ✓ R²: 0.956401

Evaluating: Deep Arch
  ✓ MSE: 0.496209
  ✓ RMSE: 0.704421
  ✓ MAE: 0.312941
  ✓ R²: 0.949972
...
```

---

# CELL 7: DISPLAY RESULTS TABLE

```python
print("\n" + "="*60)
print("HASIL PERBANDINGAN (Sorted by RMSE)")
print("="*60)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")
```

**`results_df.to_string(index=False)`** menampilkan DataFrame tanpa index.

**`to_csv()`** menyimpan hasil ke file CSV untuk dokumentasi.

**Output:**
```
============================================================
HASIL PERBANDINGAN (Sorted by RMSE)
============================================================
          Model      MSE     RMSE      MAE       R²
     No Dropout 0.252173 0.502168 0.106950 0.974576
         Low LR 0.280062 0.529209 0.112614 0.971764
        High LR 0.397693 0.630629 0.263816 0.959904
    Large Batch 0.400676 0.632989 0.251193 0.959604
    Simple Arch 0.419859 0.647965 0.269471 0.957669
       Baseline 0.432443 0.657604 0.311985 0.956401
        RMSprop 0.473277 0.687951 0.332657 0.952284
Short Seq (15d) 0.488989 0.699277 0.398983 0.950686
      Deep Arch 0.496209 0.704421 0.312941 0.949972
            SGD 0.538383 0.733746 0.252078 0.945720
    Small Batch 0.605848 0.778363 0.390826 0.938918
 Long Seq (60d) 0.624977 0.790555 0.375408 0.937024
   High Dropout 2.517743 1.586740 0.889334 0.746159

✓ Results saved to 'model_comparison_results.csv'
```

**Interpretasi Hasil:**
- **No Dropout** = Model terbaik (RMSE terendah: 0.502)
- **High Dropout** = Model terburuk (RMSE tertinggi: 1.587)
- R² > 0.95 untuk sebagian besar model = prediksi sangat baik

---

# CELL 8: VISUALISASI BAR CHARTS

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['RMSE', 'MAE', 'MSE', 'R²']
colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx // 2, idx % 2]
    
    # Sort berdasarkan metric
    if metric == 'R²':
        sorted_df = results_df.sort_values(metric, ascending=False)  # Higher is better
    else:
        sorted_df = results_df.sort_values(metric, ascending=True)   # Lower is better
    
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.8)
    
    # Highlight best model (bar pertama setelah sorting)
    bars[0].set_color('green')
    bars[0].set_alpha(1.0)
    
    ax.set_xlabel(metric)
    ax.set_title(f'{metric} Comparison (Best = Green)')
    ax.invert_yaxis()  # Model terbaik di atas

plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Penjelasan Visualisasi:**

**`plt.subplots(2, 2)`** membuat grid 2×2 untuk 4 metrics.

**`ax.barh()`** membuat horizontal bar chart:
- Sumbu Y = nama model
- Sumbu X = nilai metric

**Sorting:**
- RMSE, MAE, MSE: `ascending=True` (lower is better)
- R²: `ascending=False` (higher is better)

**`bars[0].set_color('green')`** mewarnai bar terbaik dengan hijau.

**`ax.invert_yaxis()`** membalik sumbu Y agar model terbaik di atas.

---

# KESIMPULAN DARI HASIL PERBANDINGAN

## Ranking Model (Terbaik ke Terburuk):

| Rank | Model | RMSE | R² | Insight |
|------|-------|------|-----|---------|
| 1 | No Dropout | 0.502 | 0.975 | Tanpa regularisasi terbaik |
| 2 | Low LR | 0.529 | 0.972 | Learning rate rendah stabil |
| 3 | High LR | 0.631 | 0.960 | LR tinggi masih bagus |
| 4 | Large Batch | 0.633 | 0.960 | Batch besar efektif |
| 5 | Simple Arch | 0.648 | 0.958 | Arsitektur sederhana cukup |
| 6 | Baseline | 0.658 | 0.956 | Referensi standar |
| ... | ... | ... | ... | ... |
| 13 | High Dropout | 1.587 | 0.746 | Dropout 0.5 terlalu agresif |

## Temuan Penting:

1. **No Dropout terbaik** → Dataset besar (1.5 juta samples) tidak butuh regularisasi kuat
2. **High Dropout terburuk** → Dropout 0.5 terlalu agresif, model underfitting
3. **Deep Architecture tidak lebih baik** → Lebih kompleks ≠ lebih baik
4. **Sequence 30 hari optimal** → Cocok untuk pola curah hujan bulanan
5. **Adam optimizer konsisten** → Lebih baik dari RMSprop dan SGD

---

**Dokumen ini menjelaskan SEMUA cell dalam notebook comparison_13_models.ipynb dengan penjelasan langsung di konteks kode.**
