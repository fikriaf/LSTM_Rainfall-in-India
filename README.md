# Prediksi Curah Hujan Harian di India Menggunakan LSTM

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square)](https://matplotlib.org/)

## Deskripsi Proyek

Implementasi model **Long Short-Term Memory (LSTM)** untuk prediksi curah hujan harian di India menggunakan data historis 115 tahun (1901-2015). Proyek ini mencakup 13 eksperimen komprehensif dengan berbagai konfigurasi hyperparameter untuk menemukan arsitektur optimal.

### Highlights

- Dataset: 1.5+ juta data point harian dari 36 subdivisi meteorologi India
- 13 eksperimen dengan variasi learning rate, batch size, optimizer, arsitektur, dan dropout
- Model terbaik mencapai **R² Score 0.9746** dan **RMSE 0.5022 mm**
- Analisis komprehensif performa model dengan visualisasi interaktif

## Dataset

**Sumber**: [Rainfall in India 1901-2015](https://www.kaggle.com/datasets) - Kaggle

**Karakteristik**:
- Periode: 1901-2015 (115 tahun)
- Cakupan: 36 subdivisi meteorologi di India
- Total data points: 1,503,342 (setelah transformasi harian)
- Format: Time series curah hujan bulanan ditransformasi ke harian

## Arsitektur Model

### Model Baseline

```
Input Layer: (sequence_length=30, features=1)
├── LSTM Layer 1: 64 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 2: 32 units
├── Dropout: 0.2
├── Dense Layer: 16 units, activation='relu'
└── Output Layer: 1 unit

Total Parameters: 29,857 (116.63 KB)
```

### Konfigurasi Training

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 100 (dengan early stopping patience=10)
- **Train/Test Split**: 80/20 (kronologis)

## Hasil Eksperimen

### Perbandingan 13 Model

![Model Ranking](model_ranking.png)

| Rank | Model | RMSE | MAE | R² Score |
|------|-------|------|-----|----------|
| 1 | No Dropout | 0.5022 | 0.1070 | **0.9746** |
| 2 | Low LR (0.0001) | 0.5292 | 0.1126 | 0.9718 |
| 3 | High LR (0.01) | 0.6306 | 0.2638 | 0.9599 |
| 4 | Large Batch (128) | 0.6330 | 0.2512 | 0.9596 |
| 5 | Simple Arch (32-16-8) | 0.6480 | 0.2695 | 0.9577 |
| 6 | Baseline | 0.6576 | 0.3120 | 0.9564 |
| 7 | RMSprop | 0.6880 | 0.3327 | 0.9523 |
| 8 | Short Seq (15d) | 0.6993 | 0.3990 | 0.9507 |
| 9 | Deep Arch (128-64-32) | 0.7044 | 0.3129 | 0.9500 |
| 10 | SGD | 0.7337 | 0.2521 | 0.9457 |
| 11 | Small Batch (16) | 0.7784 | 0.3908 | 0.9389 |
| 12 | Long Seq (60d) | 0.7906 | 0.3754 | 0.9370 |
| 13 | High Dropout (0.5) | 1.5867 | 0.8893 | 0.7462 |

### Visualisasi Performa

![Model Comparison](model_comparison_plots.png)

### Distribusi Geografis Performa

![Geographic Distribution](gambar_5_11_distribusi_geografis_performa.png)

### Perbandingan 2 Model Terbaik

![Top 2 Models Comparison](comparison_2_models_map.png)

## Insight Utama

### 1. Pengaruh Dropout
- Model **tanpa dropout** menunjukkan performa terbaik
- Dataset besar (1.5M sequences) mencegah overfitting
- Dropout berlebihan (0.5) justru merugikan performa

### 2. Pengaruh Learning Rate
- **Low LR (0.0001)**: Konvergensi stabil, performa terbaik kedua
- **High LR (0.01)**: Konvergensi cepat namun kurang stabil
- Learning rate rendah optimal untuk dataset kompleks

### 3. Pengaruh Batch Size
- **Large batch (128)**: Training cepat, performa baik
- **Small batch (16)**: Gradient noise berlebihan, performa menurun
- Batch size 64-128 optimal untuk dataset besar

### 4. Pengaruh Arsitektur
- **Simple architecture** (32-16-8) mengejutkan dengan performa baik
- **Deep architecture** (128-64-32) tidak selalu lebih baik
- Kapasitas model harus disesuaikan dengan kompleksitas data

### 5. Pengaruh Sequence Length
- **30 hari**: Optimal untuk menangkap pola bulanan
- **15 hari**: Context window terlalu pendek
- **60 hari**: Menambah noise, performa menurun

## Struktur Proyek

```
.
├── project-deep-learning-lstm-rainfall-in-india.ipynb  # Notebook utama
├── comparison_13_models.ipynb                          # Analisis komparasi
├── mapping_models.ipynb                                # Visualisasi geografis
├── rainfall in india 1901-2015.csv                     # Dataset utama
├── district wise rainfall normal.csv                   # Data normal curah hujan
├── El-Nino.csv                                         # Data El Niño
├── model_comparison_results.csv                        # Hasil eksperimen
├── exp_baseline_(adam,_lr=0.001,_bs=64).h5            # Model baseline
├── exp_no_dropout.h5                                   # Model terbaik
├── exp_low_lr_(adam,_lr=0.0001).h5                    # Model rank 2
├── exp_high_lr_(adam,_lr=0.01).h5                     # Eksperimen high LR
├── exp_large_batch_(bs=128).h5                        # Eksperimen large batch
├── exp_small_batch_(bs=32).h5                         # Eksperimen small batch
├── exp_simple_architecture_(32-16-8).h5               # Arsitektur simple
├── exp_deep_architecture_(128-64-32).h5               # Arsitektur deep
├── exp_high_dropout_(0.5).h5                          # Eksperimen high dropout
├── exp_rmsprop_optimizer.h5                           # Optimizer RMSprop
├── exp_sgd_optimizer.h5                               # Optimizer SGD
├── exp_short_sequence_(15_days).h5                    # Sequence 15 hari
├── exp_long_sequence_(60_days).h5                     # Sequence 60 hari
├── model_ranking.png                                   # Visualisasi ranking
├── model_comparison_plots.png                          # Plot komparasi
├── gambar_5_11_distribusi_geografis_performa.png      # Distribusi geografis
├── comparison_2_models_map.png                         # Komparasi 2 model
├── LAPORAN_PROGRESS_DEEP_LEARNING.md                  # Laporan lengkap
├── PRESENTASI_PROGRESS_DEEP_LEARNING.pdf              # Slide presentasi
└── README.md                                           # File ini
```

## Instalasi

### Requirements

```bash
pip install tensorflow==2.20.0
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install jupyter notebook
```

### Clone Repository

```bash
git clone <repository-url>
cd lstm-rainfall-prediction-india
```

### Download Dataset

```bash
python download_dataset.py
```

## Penggunaan

### 1. Training Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

### 2. Load Model Terbaik

```python
from tensorflow.keras.models import load_model

# Load model terbaik (No Dropout)
model = load_model('exp_no_dropout.h5')

# Prediksi
predictions = model.predict(X_test)
```

### 3. Evaluasi

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Hitung metrik
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.4f} mm")
print(f"MAE: {mae:.4f} mm")
print(f"R² Score: {r2:.4f}")
```

## Metrik Evaluasi

- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar MSE, dalam satuan mm
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Proporsi variance yang dijelaskan (0-1, semakin tinggi semakin baik)

## Teknologi

- **Python 3.8+**: Bahasa pemrograman utama
- **TensorFlow 2.20.0**: Framework deep learning
- **Keras**: High-level API untuk neural networks
- **NumPy**: Komputasi numerik
- **Pandas**: Manipulasi dan analisis data
- **Scikit-learn**: Preprocessing dan metrik evaluasi
- **Matplotlib & Seaborn**: Visualisasi data

## Kontribusi

Kontribusi sangat diterima! Silakan buat pull request atau buka issue untuk:

- Perbaikan bug
- Penambahan fitur baru
- Peningkatan dokumentasi
- Eksperimen arsitektur baru

## Roadmap

- [ ] Implementasi Bidirectional LSTM
- [ ] Attention mechanism untuk interpretabilitas
- [ ] CNN-LSTM hybrid architecture
- [ ] Feature engineering dengan data eksternal (El Niño, temperature)
- [ ] Ensemble methods dari top-5 model
- [ ] Web application untuk prediksi interaktif
- [ ] REST API untuk model serving
- [ ] Deployment ke cloud (AWS/GCP/Heroku)

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

## Referensi

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.
2. Kumar, D., et al. (2022). LSTM-based rainfall prediction using spatial and temporal features. Journal of Hydrology, 608, 127634.
3. Poornima, S., & Pushpalatha, M. (2022). Rainfall prediction using deep learning techniques. Environmental Science and Pollution Research, 29(45), 67835-67856.
4. Mishra, V., et al. (2021). Deep learning for improved global precipitation forecasting. Nature Communications, 12(1), 1-12.

## Kontak

Untuk pertanyaan atau kolaborasi, silakan hubungi melalui:

- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

---

**Dibuat dengan** ❤️ **menggunakan TensorFlow dan Keras**

*Proyek Deep Learning - Prediksi Curah Hujan India*
