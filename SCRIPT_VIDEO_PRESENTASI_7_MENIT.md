# SCRIPT VIDEO PRESENTASI - 7 MENIT
## Prediksi Curah Hujan Harian di India Menggunakan LSTM

**Durasi Total: 7 menit (420 detik)**

---

## OPENING (30 detik)

Selamat pagi/siang/sore. Saya Fikri Armia Fahmi dari Program Studi Informatika. Hari ini saya akan mempresentasikan progress penelitian saya tentang Prediksi Curah Hujan Harian di India Menggunakan Long Short-Term Memory atau LSTM.

---

## SLIDE 1: LATAR BELAKANG (40 detik)

Mengapa prediksi curah hujan penting? Di India, sektor pertanian menyumbang 18% PDB dengan populasi 1.4 miliar jiwa. Curah hujan di India sangat bervariasi karena dipengaruhi sistem monsun, ENSO, dan perubahan iklim. Ini menciptakan tantangan besar dalam perencanaan sumber daya air dan mitigasi bencana.

Solusi yang saya tawarkan adalah menggunakan deep learning dengan LSTM. LSTM mampu menangkap dependensi temporal jangka panjang dan menunjukkan performa superior dibanding metode tradisional seperti ARIMA, dengan peningkatan akurasi 15-25%.

---

## SLIDE 2-3: RUMUSAN MASALAH & TUJUAN (30 detik)

Penelitian ini menjawab lima pertanyaan utama: bagaimana merancang arsitektur LSTM optimal, pengaruh hiperparameter, performa dibanding konfigurasi lain, integrasi faktor eksternal, dan aplikasi praktisnya.

Target utama adalah mengimplementasikan LSTM dengan 13 konfigurasi berbeda, mencapai R² minimal 0.97, dan mengembangkan aplikasi praktis untuk pertanian dan peringatan dini bencana.

---

## SLIDE 4: DATASET (30 detik)

Dataset yang digunakan adalah Rainfall in India dari Kaggle, mencakup periode 115 tahun dari 1901 hingga 2015 di 36 subdivisi meteorologi. Data original berisi 4,116 observasi bulanan yang ditransformasi menjadi 1.5 juta observasi harian. Data dibagi 80% untuk training dan 20% untuk testing dengan sequence length 30 hari.

---

## SLIDE 5-6: ARSITEKTUR & EKSPERIMEN (40 detik)

Arsitektur baseline menggunakan dua layer LSTM dengan 64 dan 32 units, dropout 0.2, dan dense layer 16 units. Total parameter sekitar 30 ribu.

Saya melakukan 13 eksperimen dengan variasi learning rate, batch size, dropout rate, optimizer, arsitektur, dan sequence length. Metrik evaluasi yang digunakan adalah RMSE, MAE, dan R² Score.

---

## SLIDE 7: HASIL EKSPERIMEN (45 detik)

Hasil menunjukkan model terbaik adalah konfigurasi No Dropout dengan RMSE 0.5022 mm dan R² 0.9746. Ini berarti model mampu menjelaskan 97.46% varians dalam data. Model ini memberikan peningkatan 24% dibanding baseline.

Ranking kedua adalah Low Learning Rate dengan RMSE 0.5292, diikuti High Learning Rate, Large Batch Size, dan Simple Architecture. Yang menarik, model tanpa dropout justru memberikan performa terbaik.

---

## SLIDE 8: ANALISIS HIPERPARAMETER (40 detik)

Temuan kunci dari eksperimen hiperparameter: Pertama, dropout ternyata tidak diperlukan untuk dataset besar ini. No dropout memberikan performa terbaik, sementara high dropout malah menurunkan performa drastis.

Kedua, learning rate rendah memberikan konvergensi yang lebih stabil. Ketiga, batch size besar lebih optimal untuk efisiensi. Dan keempat, optimizer Adam dan RMSprop jauh lebih baik dibanding SGD.

---

## SLIDE 9: ANALISIS PREDIKSI (10 detik)

Model terbaik menunjukkan clustering yang ketat pada scatter plot dengan R² 0.9746. Time series menunjukkan model mampu menangkap pola musiman dengan baik, meskipun ada sedikit underprediksi pada nilai ekstrem.

## SLIDE 10: PERFORMA GEOGRAFIS (10 detik)

Bisa dilihat pada gambar, Secara geografis, model konsisten dengan R² di atas 0.95 di mayoritas subdivisi. Performa terbaik di wilayah dengan variabilitas moderat, sementara wilayah pesisir sedikit lebih menantang.

---

## SLIDE 11-12: APLIKASI MULTI-DOMAIN (15 detik)

Penelitian ini tidak hanya fokus pada akurasi model, tapi juga aplikasi praktis. Pertama, transformasi ke klasifikasi 3 kategori menggunakan terciles: Low di bawah 913 mm, Medium 913-1346 mm, dan High di atas 1346 mm. Distribusinya seimbang dengan Random Forest classifier.

Kedua, analisis korelasi faktor eksternal menunjukkan ENSO berkorelasi kuat dengan curah hujan. El Niño (curah hujan turn rata-rata 1303 mm), sementara La Niña curahhujan naik (1472 mm).

Ketiga, sistem peringatan dini dikembangkan dengan lead time 3-7 hari untuk banjir dan 2-4 minggu untuk kekeringan, dengan potensi peningkatan efisiensi air 10-15%.

---
## SLIDE 13: APLIKASI MULTI-DOMAIN (10 detik)

Nah slide ini menjadi bukti nyata berupa metriks dan scatter plot untuk klasifikasi dan pernyataan bahwa ENSO menunjukkan hubungan positif terhadap curah hujan tahunan.

## SLIDE 14: APLIKASI MULTI-DOMAIN (15 detik)

Selanjutnya, di slide ini menggambarkan detail terkait ENSO, YAITU Rata-rata lima tahunan menunjukkan El Niño menurunkan curah hujan, sedangkan La Niña meningkatkannya. Curah hujan tertinggi terjadi saat La Niña (1472 mm), lalu Normal (1426 mm), dan terendah pada El Niño (1303 mm). Tahun ekstrem seperti El Niño 2002 (92 mm) dan Normal 1961 (5554 mm) menegaskan pola ini. Meski korelasi tidak linear, pengaruh ENSO tetap konsisten dan penting bagi prediksi jangka panjang.

## SLIDE 15: CLUSTERING & DETEKSI ANOMALI (20 detik)

Analisis musiman menunjukkan pola yang jelas: monsun Juni-September memiliki curah hujan tertinggi, sementara Desember-Februari paling kering. Meghalaya mencatat curah hujan tertinggi 3682 mm per tahun.

Clustering K-Means mengidentifikasi 4 zona iklim dari 641 distrik: kering, moderat, basah, dan sangat basah.

## SLIDE 16: DETEKSI ANOMALI (25 detik)

Deteksi anomali menemukan 45 distrik atau 7% dengan curah hujan ekstrem. Distrik paling ekstrem adalah Tamenglong di Manipur dengan 7229 mm per tahun.

Trend historis 115 tahun menunjukkan penurunan linear 0.21 mm per tahun, dengan periode paling basah di 1930-1960 dan penurunan nyata setelah 1990-an, mengindikasikan dampak perubahan iklim.

---

## SLIDE 17-18: KETERBATASAN & RENCANA (30 detik)

Keterbatasan utama adalah transformasi data yang menggunakan distribusi uniform, underprediksi pada kejadian ekstrem, dan training hanya di CPU.

Rencana pengembangan mencakup ensemble methods, attention mechanisms, integrasi fitur eksternal, dan deployment aplikasi web serta mobile untuk petani dan pembuat kebijakan.

---

## SLIDE 19: KONTRIBUSI & KESIMPULAN (40 detik)

Kontribusi penelitian ini ada di tiga aspek: metodologis dengan studi hiperparameter komprehensif, teknis dengan mencapai R² di atas 0.97, dan praktis dengan framework untuk pertanian dan peringatan dini.

Kesimpulannya, model LSTM berhasil dikembangkan dengan performa terdepan. 13 eksperimen mengidentifikasi konfigurasi optimal dengan temuan kunci bahwa dataset besar tidak memerlukan dropout agresif. Framework aplikasi siap untuk pertanian, peringatan dini, dan manajemen air.

---

## CLOSING (10 detik)

Sekian dari saya, terima kasih atas perhatiannya.

---

