# 📊 PENJELASAN COMPARISON 13 MODELS
## Perbandingan 13 Variasi Model LSTM

---

## 1. TUJUAN NOTEBOOK

Notebook ini membandingkan **13 model LSTM** dengan konfigurasi berbeda untuk menemukan kombinasi hyperparameter terbaik dalam prediksi curah hujan.

---

## 2. DAFTAR 13 MODEL YANG DIBANDINGKAN

| No | Model | Konfigurasi | Tujuan Eksperimen |
|----|-------|-------------|-------------------|
| 1 | **Baseline** | Adam, lr=0.001, bs=64 | Model referensi standar |
| 2 | **Deep Architecture** | 128-64-32 units | Apakah lebih dalam = lebih baik? |
| 3 | **High Dropout** | dropout=0.5 | Regularisasi agresif |
| 4 | **High LR** | lr=0.01 | Learning rate tinggi |
| 5 | **Large Batch** | batch_size=128 | Batch besar |
| 6 | **Long Sequence** | 60 hari | Konteks temporal lebih panjang |
| 7 | **Low LR** | lr=0.0001 | Learning rate rendah |
| 8 | **No Dropout** | dropout=0.0 | Tanpa regularisasi |
| 9 | **RMSprop** | RMSprop optimizer | Optimizer alternatif |
| 10 | **SGD** | SGD optimizer | Optimizer klasik |
| 11 | **Short Sequence** | 15 hari | Konteks temporal pendek |
| 12 | **Simple Architecture** | 32-16-8 units | Arsitektur sederhana |
| 13 | **Small Batch** | batch_size=32 | Batch kecil |

---

## 3. FUNGSI-FUNGSI UTAMA

### A. `load_and_preprocess_data()`
**Fungsi:** Memuat dan memproses dataset curah hujan.

**Langkah-langkah:**
```
1. Baca CSV → DataFrame
2. Encode SUBDIVISION → angka (LabelEncoder)
3. Handle missing values (forward fill, backward fill)
4. Konversi data bulanan → harian
```

**Konversi Bulanan ke Harian:**
```python
daily_rainfall = monthly_rainfall / days_in_month
```

**Contoh:**
```
Januari 2015: 310 mm (31 hari)
Daily = 310 / 31 = 10 mm/hari
```

---

### B. `create_sequences(data, seq_length)`
**Fungsi:** Membuat pasangan input-output untuk supervised learning.

**Proses:**
```
Data: [d1, d2, d3, d4, d5, d6, ...]

seq_length = 3:
  X[0] = [d1, d2, d3] → y[0] = d4
  X[1] = [d2, d3, d4] → y[1] = d5
  X[2] = [d3, d4, d5] → y[2] = d6
```

**Variasi Sequence Length:**
| Model | seq_length | Input Shape |
|-------|------------|-------------|
| Short Sequence | 15 | (15, 1) |
| Default | 30 | (30, 1) |
| Long Sequence | 60 | (60, 1) |

---

### C. `evaluate_model(model_path, X_test, y_test, scaler)`
**Fungsi:** Mengevaluasi performa model yang sudah ditraining.

**Langkah:**
```
1. Load model dari file .h5
2. Prediksi: y_pred = model.predict(X_test)
3. Inverse transform (kembalikan ke skala asli)
4. Hitung metrics: MSE, RMSE, MAE, R²
```

**Inverse Transform:**
```python
# Dari skala 0-1 kembali ke mm
y_original = scaler.inverse_transform(y_scaled)
```

---

## 4. METRICS EVALUASI

### A. MSE (Mean Squared Error)
```
MSE = (1/n) × Σ(y_actual - y_pred)²
```
- Penalti besar untuk error besar
- Satuan: mm² (kuadrat)

### B. RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
- Satuan sama dengan data asli (mm)
- Lebih mudah diinterpretasi

### C. MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|y_actual - y_pred|
```
- Robust terhadap outlier
- Rata-rata kesalahan absolut

### D. R² Score
```
R² = 1 - (SS_res / SS_tot)
```
- Range: 0-1 (bisa negatif jika sangat buruk)
- 1.0 = prediksi sempurna
- 0.0 = sama dengan prediksi rata-rata

---

## 5. NORMALISASI UNTUK RANKING

**Tujuan:** Membandingkan metrics dengan skala berbeda secara adil.

**Rumus Min-Max Normalization:**
```
x_norm = (x - x_min) / (x_max - x_min)
```

**Untuk R² (diinvert karena higher is better):**
```
R²_norm = (R²_max - R²) / (R²_max - R²_min)
```

**Average Score:**
```
Avg_Score = (RMSE_norm + MAE_norm + MSE_norm + R²_norm) / 4
```
- Semakin rendah = semakin baik

---

## 6. ANALISIS HYPERPARAMETER

### A. Learning Rate
```
┌─────────────────────────────────────────────────────┐
│  High LR (0.01)    → Konvergen cepat, tidak stabil  │
│  Default (0.001)   → Keseimbangan optimal           │
│  Low LR (0.0001)   → Stabil, tapi lambat            │
└─────────────────────────────────────────────────────┘
```

### B. Batch Size
```
┌─────────────────────────────────────────────────────┐
│  Small (32)   → Gradient noisy, regularization      │
│  Medium (64)  → Keseimbangan                        │
│  Large (128)  → Gradient stable, training cepat     │
└─────────────────────────────────────────────────────┘
```

### C. Dropout Rate
```
┌─────────────────────────────────────────────────────┐
│  No Dropout (0.0)  → Tanpa regularisasi             │
│  Default (0.2)     → Regularisasi ringan            │
│  High (0.5)        → Regularisasi agresif           │
└─────────────────────────────────────────────────────┘
```

### D. Architecture Depth
```
┌─────────────────────────────────────────────────────┐
│  Simple (32-16-8)    → ~7,500 parameters            │
│  Baseline (64-32-16) → ~30,000 parameters           │
│  Deep (128-64-32)    → ~120,000 parameters          │
└─────────────────────────────────────────────────────┘
```

### E. Sequence Length
```
┌─────────────────────────────────────────────────────┐
│  Short (15 days)  → Konteks terbatas                │
│  Default (30 days)→ Pola bulanan                    │
│  Long (60 days)   → Pola 2 bulan                    │
└─────────────────────────────────────────────────────┘
```

### F. Optimizer
```
┌─────────────────────────────────────────────────────┐
│  Adam     → Adaptif, konvergen cepat                │
│  RMSprop  → Baik untuk RNN/LSTM                     │
│  SGD      → Klasik, butuh tuning manual             │
└─────────────────────────────────────────────────────┘
```

---

## 7. ALUR KERJA NOTEBOOK

```
┌─────────────────────────────────────────────────────┐
│  1. IMPORT LIBRARIES                                │
│     └── pandas, numpy, tensorflow, sklearn          │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  2. LOAD & PREPROCESS DATA                          │
│     └── CSV → Daily data → Scaled sequences         │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  3. CREATE TEST SET                                 │
│     └── 80% train, 20% test                         │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  4. EVALUATE 13 MODELS                              │
│     └── Load .h5 → Predict → Calculate metrics      │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  5. VISUALIZE RESULTS                               │
│     └── Bar charts, ranking plots                   │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  6. STATISTICAL ANALYSIS                            │
│     └── Mean, median, std, improvement %            │
└─────────────────────────────────────────────────────┘
```

---

## 8. VISUALISASI YANG DIHASILKAN

### A. Bar Charts (4 Metrics)
```
┌──────────────────────────────────────┐
│  RMSE Comparison    │  MAE Comparison │
│  ████████ Model A   │  ██████ Model A │
│  ██████ Model B     │  ████ Model B   │
├──────────────────────────────────────┤
│  MSE Comparison     │  R² Comparison  │
│  ████████ Model A   │  ██████ Model A │
│  ██████ Model B     │  ████ Model B   │
└──────────────────────────────────────┘
```
- Hijau = Model terbaik
- Sorted dari terbaik ke terburuk

### B. Overall Ranking
```
Normalized scores (lower = better):
┌────────────────────────────────────────┐
│ Model    │ RMSE │ MAE │ MSE │ R² │ Avg │
├────────────────────────────────────────┤
│ Best     │ 0.0  │ 0.0 │ 0.0 │ 0.0│ 0.0 │
│ ...      │ ...  │ ... │ ... │ ...│ ... │
│ Worst    │ 1.0  │ 1.0 │ 1.0 │ 1.0│ 1.0 │
└────────────────────────────────────────┘
```

---

## 9. OUTPUT FILES

| File | Isi |
|------|-----|
| `model_comparison_results.csv` | Tabel hasil semua metrics |
| `model_comparison_plots.png` | 4 bar charts metrics |
| `model_ranking.png` | Overall ranking plot |

---

## 10. STATISTICAL SUMMARY

**Metrics yang dihitung:**
```python
# Untuk setiap metric (RMSE, R², dll):
Mean   = rata-rata semua model
Median = nilai tengah
Std    = standar deviasi (variasi)
Min    = nilai terbaik
Max    = nilai terburuk
```

**Performance Improvement:**
```python
improvement = ((worst_rmse - best_rmse) / worst_rmse) × 100%
```
- Menunjukkan seberapa besar perbedaan model terbaik vs terburuk

---

## 11. KESIMPULAN EKSPERIMEN

### Temuan Umum:

| Aspek | Temuan |
|-------|--------|
| **Dropout** | No Dropout sering lebih baik (dataset besar) |
| **Learning Rate** | 0.001 (default Adam) optimal |
| **Batch Size** | 64 memberikan keseimbangan baik |
| **Architecture** | Baseline (64-32-16) cukup optimal |
| **Sequence Length** | 30 hari cocok untuk pola bulanan |
| **Optimizer** | Adam konsisten terbaik |

### Insight Penting:
1. **Lebih kompleks ≠ lebih baik** - Deep architecture bisa overfitting
2. **Regularisasi berlebihan berbahaya** - High dropout menurunkan performa
3. **Learning rate kritis** - Terlalu tinggi/rendah sama-sama buruk
4. **Sequence length harus sesuai pola data** - 30 hari optimal untuk curah hujan

---

## 12. CARA MEMBACA HASIL

### Contoh Output:
```
🏆 BEST MODEL
Model: No Dropout
  • RMSE: 0.502234
  • MAE:  0.107012
  • MSE:  0.252239
  • R²:   0.974612
```

**Interpretasi:**
- RMSE 0.50 mm = rata-rata error prediksi ~0.5 mm
- R² 0.97 = model menjelaskan 97% variasi data
- Model tanpa dropout terbaik karena dataset cukup besar

---

**Notebook ini memberikan framework sistematis untuk membandingkan berbagai konfigurasi model LSTM dan menemukan hyperparameter optimal untuk prediksi curah hujan.**
