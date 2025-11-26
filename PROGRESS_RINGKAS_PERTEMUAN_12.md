# LAPORAN PROGRESS PENELITIAN
## Prediksi Curah Hujan Harian di India Menggunakan LSTM

**Peneliti:** Fikri Armia Fahmi | **Program Studi:** Informatika | **Pembimbing:** Dr. Ida Nurhaida, M.T  
**Periode:** Semester Genap 2024/2025 | **Pertemuan:** 12 | **Kategori:** Deep Learning - Time Series Forecasting

---

## 1. TUJUAN PENELITIAN

Mengembangkan model LSTM untuk prediksi curah hujan harian di India dengan target: (1) R² minimal 0.97 dan RMSE maksimal 0.55 mm melalui eksperimen 13 konfigurasi, (2) integrasi faktor eksternal (ENSO, temperatur, kelembapan), (3) aplikasi praktis untuk pertanian, peringatan dini bencana, dan manajemen sumber daya air.

## 2. DATASET

**Sumber:** Rainfall in India 1901-2015 (Kaggle) mencakup 36 subdivisi meteorologi dan 641 distrik selama 115 tahun. **Data:** 4,116 observasi bulanan ditransformasi menjadi 1,503,342 observasi harian dengan sliding window 30 hari, menghasilkan 1,201,809 training sequences (80%) dan 300,453 test sequences (20%).

**Masalah Awal:** Transformasi data bulanan ke harian menggunakan distribusi uniform, missing values kurang dari 1%, distribusi skewed dengan variabilitas tinggi antar wilayah (165-11,777 mm/tahun), ketidakseimbangan spasial di wilayah timur laut.

## 3. METODE YANG SUDAH DIKERJAKAN

**Preprocessing:** Transformasi temporal bulanan ke harian, normalisasi MinMaxScaler per-subdivisi (0-1), encoding 36 subdivisi, sequence creation sliding window 30 hari, train-test split kronologis 80:20.

**Model Development:** Arsitektur baseline LSTM 64-32-16 units dengan dropout 0.2 (29,857 parameters), optimizer Adam (lr=0.001), loss function MSE, early stopping patience 10 epochs.

**Eksperimen 13 Konfigurasi:** Variasi learning rate (0.0001, 0.001, 0.01), batch size (16, 32, 64, 128), dropout rate (0.0, 0.2, 0.5), optimizer (Adam, RMSprop, SGD), arsitektur (Simple 32-16-8, Baseline 64-32-16, Deep 128-64-32), sequence length (15, 30, 60 hari).

**Analisis Lanjutan:** Klasifikasi 3 kategori menggunakan terciles, korelasi ENSO-temperatur-kelembapan, clustering K-Means 641 distrik (4 zona iklim), deteksi anomali 45 distrik (7%), trend analysis 115 tahun.

## 4. HASIL SEMENTARA

**Performa Model Terbaik (No Dropout):**

| Metrik | Nilai | Target | Status |
|--------|-------|--------|--------|
| RMSE | 0.5022 mm | ≤0.55 mm | Tercapai |
| MAE | 0.1070 mm | ≤0.15 mm | Tercapai |
| R² Score | 0.9746 | ≥0.97 | Tercapai |

Peningkatan 24% dibanding baseline (RMSE 0.6576 mm).

**Ranking Top 5 Model:**

| Rank | Model | RMSE (mm) | R² Score |
|------|-------|-----------|----------|
| 1 | No Dropout | 0.5022 | 0.9746 |
| 2 | Low LR (0.0001) | 0.5292 | 0.9718 |
| 3 | High LR (0.01) | 0.6306 | 0.9599 |
| 4 | Large Batch (64) | 0.6330 | 0.9596 |
| 5 | Simple Arch | 0.6480 | 0.9577 |

**Temuan Kunci:** Dataset besar tidak memerlukan dropout agresif, learning rate rendah optimal untuk konvergensi stabil, batch size besar efisien, optimizer Adam/RMSprop superior, arsitektur moderat memberikan keseimbangan terbaik.

**Analisis Lanjutan:** Klasifikasi Random Forest 3 kategori terciles (Low <913.8mm: 212 distrik, Medium 913.8-1346.8mm: 217 distrik, High >1346.8mm: 212 distrik). Korelasi ENSO r=-0.067 (El Niño 1303mm/tahun, La Niña 1472mm/tahun). Clustering 4 zona iklim: Kering <800mm, Moderat 800-1500mm, Basah 1500-2500mm, Sangat Basah >2500mm. Deteksi anomali 45 distrik (7%), tertinggi Tamenglong 7229mm/tahun. Trend 1901-2015 slope -0.21mm/tahun, penurunan nyata setelah 1990-an.

## 5. KENDALA YANG DIHADAPI

**Data:** Transformasi uniform tidak menangkap variabilitas harian aktual, underprediksi kejadian ekstrem >20mm, fitur terbatas tanpa variabel meteorologi lain. **Komputasi:** Training CPU 150-250 detik/eksperimen, tidak dapat grid search ekstensif. **Metodologis:** Tidak memanfaatkan korelasi spasial, tidak ada confidence interval, interpretabilitas terbatas. **Validasi:** Belum diuji pengguna akhir dan data di luar periode 1901-2015.

## 6. RENCANA LANJUTAN

**Peningkatan Model:** Ensemble top-5 model, attention mechanisms, bidirectional LSTM, probabilistic forecasting Monte Carlo Dropout. **Integrasi Fitur:** Variabel meteorologi (temperatur, kelembapan, tekanan), ENSO Index aktual, spatial features, multi-task learning. **Aplikasi:** Web app Streamlit/Flask, dashboard monitoring, RESTful API, mobile prototype peringatan dini. **Dokumentasi:** Laporan akhir komprehensif, presentasi UAS, repository GitHub, paper draft. **Target:** R² ≥0.98, RMSE ≤0.45mm, web app 3+ fitur, laporan 100+ halaman.

## 7. KESIMPULAN

Model LSTM mencapai performa terdepan (R²=0.9746, RMSE=0.5022mm) melampaui target. Eksperimen 13 konfigurasi mengidentifikasi model tanpa dropout terbaik untuk dataset besar. Analisis multi-domain (klasifikasi, ENSO, clustering, anomali, trend) selesai. Kontribusi: metodologi optimasi sistematis, akurasi tinggi arsitektur efisien, framework aplikasi pertanian-bencana, analisis 115 tahun data India. Progress 85% (Model Development Complete), remaining 15% (Application Development).

---

*Dokumen progress ringkas Pertemuan 12. Detail lengkap: LAPORAN_PROGRESS_DEEP_LEARNING_REVISED.md*
