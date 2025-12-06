# 🗺️ PENJELASAN MAPPING MODELS
## Perbandingan Model & Visualisasi Geografis

---

## 1. TUJUAN NOTEBOOK

Notebook ini melakukan:
1. **Perbandingan mendalam** antara 2 model (No Dropout vs Baseline)
2. **Analisis statistik** prediksi kedua model
3. **Analisis residual** untuk evaluasi kualitas model
4. **Visualisasi geografis** hasil prediksi pada peta India

---

## 2. MODEL YANG DIBANDINGKAN

| Model | File | Arsitektur | Dropout |
|-------|------|------------|---------|
| No Dropout | `exp_no_dropout.h5` | LSTM 64-32 | 0.0 |
| Baseline | `exp_baseline_(adam,_lr=0.001,_bs=64).h5` | LSTM 64-32 | 0.2 |

---

## 3. FUNGSI-FUNGSI UTAMA

### A. `monthly_to_daily(df)`
**Fungsi:** Mengkonversi data curah hujan bulanan menjadi harian.

**Logika:**
```python
daily_rainfall = monthly_rainfall / days_in_month
```

**Contoh:**
```
Januari: 310 mm / 31 hari = 10 mm/hari
Februari: 56 mm / 28 hari = 2 mm/hari
```

**Hari per Bulan:**
| Bulan | Hari |
|-------|------|
| Jan, Mar, Mei, Jul, Agu, Okt, Des | 31 |
| Apr, Jun, Sep, Nov | 30 |
| Feb | 28 |

---

### B. `create_sequences(data, seq_len)`
**Fungsi:** Membuat sliding window untuk input LSTM.

**Ilustrasi:**
```
Data: [1, 2, 3, 4, 5, 6, 7, 8]
seq_len = 3

X[0] = [1, 2, 3] → y[0] = 4
X[1] = [2, 3, 4] → y[1] = 5
X[2] = [3, 4, 5] → y[2] = 6
...
```

---

### C. `calculate_metrics(y_true, y_pred, model_name)`
**Fungsi:** Menghitung metrics evaluasi model.

**Metrics yang dihitung:**

| Metric | Rumus | Interpretasi |
|--------|-------|--------------|
| MSE | `(1/n) × Σ(y - ŷ)²` | Error kuadrat rata-rata |
| RMSE | `√MSE` | Akar MSE, satuan sama dengan data |
| MAE | `(1/n) × Σ\|y - ŷ\|` | Error absolut rata-rata |
| R² | `1 - (SS_res/SS_tot)` | Proporsi variasi yang dijelaskan |

---

### D. `additional_metrics(y_true, y_pred, model_name)`
**Fungsi:** Menghitung metrics tambahan.

| Metric | Rumus | Interpretasi |
|--------|-------|--------------|
| MAPE | `(1/n) × Σ\|y - ŷ\|/y × 100%` | Persentase error rata-rata |
| Explained Variance | `1 - Var(y - ŷ)/Var(y)` | Variasi yang dijelaskan |

---

## 4. ANALISIS STATISTIK

### A. T-Test (Paired)
**Fungsi:** Menguji apakah prediksi kedua model berbeda secara signifikan.

**Rumus:**
```
t = (d̄) / (s_d / √n)

d̄ = rata-rata perbedaan
s_d = standar deviasi perbedaan
n = jumlah sampel
```

**Interpretasi:**
| p-value | Kesimpulan |
|---------|------------|
| < 0.05 | Prediksi berbeda signifikan |
| ≥ 0.05 | Prediksi tidak berbeda signifikan |

---

### B. Error Analysis
**Rumus:**
```
error = y_actual - y_predicted
mean_error = rata-rata(error)
std_error = standar_deviasi(error)
```

**Interpretasi:**
- Mean error ≈ 0 → Model tidak bias
- Std error kecil → Prediksi konsisten

---

## 5. ANALISIS RESIDUAL

### A. Histogram Residual
**Tujuan:** Memeriksa distribusi error.

```
Ideal: Distribusi normal (bell curve)
       Centered di 0
```

**Interpretasi:**
- Simetris → Model tidak bias
- Skewed → Ada pola yang tidak tertangkap

---

### B. Q-Q Plot (Quantile-Quantile)
**Tujuan:** Memeriksa normalitas residual.

```
Ideal: Titik-titik mengikuti garis diagonal

     ●●●●●
   ●●●
  ●●
 ●●
●●
─────────────
```

**Interpretasi:**
- Titik di garis → Residual normal
- Titik melengkung → Residual tidak normal

---

### C. ACF (Autocorrelation Function)
**Tujuan:** Memeriksa apakah residual berkorelasi dengan dirinya sendiri.

**Rumus:**
```
ACF(k) = Σ(e_t × e_{t+k}) / Σ(e_t²)

k = lag (jarak waktu)
e = residual
```

**Interpretasi:**
- ACF dalam batas → Residual independen (baik)
- ACF di luar batas → Ada pola temporal yang tidak tertangkap

---

### D. Heteroscedasticity Check
**Tujuan:** Memeriksa apakah variasi error konstan.

```
Plot: Residual vs Predictions

Ideal (Homoscedastic):     Tidak Ideal (Heteroscedastic):
    ●  ●  ●  ●                    ●
  ●  ●  ●  ●  ●                 ● ●
────────────────              ●  ●  ●
  ●  ●  ●  ●  ●             ●   ●   ●
    ●  ●  ●  ●            ●    ●    ●
                        ────────────────
```

**Interpretasi:**
- Spread konstan → Model baik
- Spread membesar → Perlu transformasi data

---

## 6. VISUALISASI GEOGRAFIS

### A. Choropleth Map
**Definisi:** Peta yang menggunakan warna untuk menunjukkan nilai variabel di setiap wilayah.

**Komponen:**
```
┌─────────────────────────────────────┐
│  PETA INDIA                         │
│  ┌─────┐ ┌─────┐                    │
│  │█████│ │░░░░░│  █ = Curah hujan   │
│  │█████│ │░░░░░│      tinggi        │
│  └─────┘ └─────┘  ░ = Curah hujan   │
│                       rendah        │
│  [Legend: 0 ──── 3000 mm]           │
└─────────────────────────────────────┘
```

---

### B. Mapping Subdivision ke State
**Masalah:** Data curah hujan menggunakan "Subdivision", shapefile menggunakan "State".

**Solusi:** Dictionary mapping

```python
subdivision_to_state = {
    'COASTAL ANDHRA PRADESH': 'ANDHRA PRADESH',
    'RAYALSEEMA': 'ANDHRA PRADESH',
    'GANGETIC WEST BENGAL': 'WEST BENGAL',
    ...
}
```

**Contoh Mapping:**
| Subdivision | State |
|-------------|-------|
| COASTAL KARNATAKA | KARNATAKA |
| NORTH INTERIOR KARNATAKA | KARNATAKA |
| SOUTH INTERIOR KARNATAKA | KARNATAKA |

---

### C. Scaling Factor
**Tujuan:** Menyesuaikan prediksi model (mm/hari) ke skala tahunan.

**Rumus:**
```
scaling_factor = pred_mean / y_test_mean
state_prediction = state_avg_real × scaling_factor
```

**Contoh:**
```
pred_mean = 5.2 mm/hari
y_test_mean = 5.0 mm/hari
scaling_factor = 5.2 / 5.0 = 1.04

Kerala real = 2500 mm/tahun
Kerala pred = 2500 × 1.04 = 2600 mm/tahun
```

---

## 7. LIBRARY YANG DIGUNAKAN

### A. GeoPandas
**Fungsi:** Membaca dan memanipulasi data geospasial.

```python
import geopandas as gpd
gdf = gpd.read_file('shapefile.shp')
gdf.plot(column='value', cmap='Blues')
```

### B. SciPy Stats
**Fungsi:** Uji statistik.

```python
from scipy.stats import ttest_rel, probplot
t_stat, p_value = ttest_rel(pred1, pred2)
```

### C. Statsmodels
**Fungsi:** Analisis time series.

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, ax=ax)
```

---

## 8. OUTPUT VISUALISASI

### A. Comparison Plots
```
┌─────────────────────────────────────────────────┐
│  Plot 1: Time Series          Plot 2: Scatter   │
│  ─── Actual                   ● No Dropout      │
│  ─── No Dropout               ● Baseline        │
│  ─── Baseline                 -- Perfect Fit    │
└─────────────────────────────────────────────────┘
```

### B. Residual Analysis (6 plots)
```
┌─────────────────────────────────────────────────┐
│  Histogram    │  Histogram    │  Q-Q Plot       │
│  No Dropout   │  Baseline     │  No Dropout     │
├─────────────────────────────────────────────────┤
│  Q-Q Plot     │  ACF          │  ACF            │
│  Baseline     │  No Dropout   │  Baseline       │
└─────────────────────────────────────────────────┘
```

### C. Geographic Maps
```
┌─────────────────────────────────────────────────┐
│  Peta 1: Real Data    │  Peta 2: Prediction     │
│  (1901-2015 ANNUAL)   │  (Model Output)         │
│  [Greens colormap]    │  [Blues colormap]       │
└─────────────────────────────────────────────────┘
```

---

## 9. ANALISIS KEKUATAN & KELEMAHAN

### Model No Dropout
| Aspek | Keterangan |
|-------|------------|
| **Kekuatan** | Akurasi tinggi (R² = 0.9746), MAE/RMSE rendah |
| **Kelemahan** | Risiko overfitting pada data baru |
| **Cocok untuk** | Dataset besar, prediksi akurat |

### Model Baseline
| Aspek | Keterangan |
|-------|------------|
| **Kekuatan** | Lebih robust, generalisasi baik |
| **Kelemahan** | Akurasi sedikit lebih rendah |
| **Cocok untuk** | Data noisy, deployment production |

---

## 10. ALUR KERJA NOTEBOOK

```
┌─────────────────────────────────────────────────┐
│  1. LOAD MODELS                                 │
│     └── exp_no_dropout.h5, exp_baseline.h5      │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  2. PREPROCESS DATA                             │
│     └── Monthly → Daily → Scaled → Sequences    │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  3. MAKE PREDICTIONS                            │
│     └── model.predict() → inverse_transform()   │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  4. CALCULATE METRICS                           │
│     └── MSE, RMSE, MAE, R², MAPE, EV            │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  5. STATISTICAL TESTS                           │
│     └── T-test, Error analysis                  │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  6. RESIDUAL ANALYSIS                           │
│     └── Histogram, Q-Q, ACF, Heteroscedasticity │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  7. GEOGRAPHIC VISUALIZATION                    │
│     └── Choropleth maps dengan GeoPandas        │
└─────────────────────────────────────────────────┘
```

---

## 11. FILES OUTPUT

| File | Deskripsi |
|------|-----------|
| `comparison_2_models_map.png` | Peta perbandingan prediksi 2 model |
| `gambar_5_11_distribusi_geografis_performa.png` | Distribusi R² per subdivisi |

---

## 12. KESIMPULAN

### Hasil Perbandingan:
| Metric | No Dropout | Baseline | Winner |
|--------|------------|----------|--------|
| RMSE | 0.5022 | 0.5234 | No Dropout |
| MAE | 0.1070 | 0.1156 | No Dropout |
| R² | 0.9746 | 0.9564 | No Dropout |

### Rekomendasi:
- **Untuk akurasi maksimal:** Gunakan No Dropout
- **Untuk robustness:** Gunakan Baseline dengan dropout
- **Untuk improvement:** Pertimbangkan ensemble kedua model

---

**Notebook ini memberikan analisis komprehensif untuk membandingkan model LSTM dan memvisualisasikan hasil prediksi secara geografis pada peta India.**
