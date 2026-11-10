 Klasifikasi Citra Pakaian dengan CNN dari Scratch
 
Proyek UTS mata kuliah **Deep Learning**, Program Studi S-1 Teknik Informatika, Universitas Mikroskil.
 
Model **Convolutional Neural Network (CNN) dasar yang dibangun dari awal** (tanpa *transfer learning* dan tanpa bobot terlatih) menggunakan **TensorFlow/Keras** dan **`tf.data`** untuk mengklasifikasikan foto produk ke dalam **11 jenis pakaian**. Model juga diekspor ke ONNX dan dijalankan langsung di browser melalui aplikasi web.
 
- Aplikasi web: `ISI ALAMAT VERCEL`
- Repository aplikasi web: https://github.com/MILAS1818/klasifikasi-pakaian
- Repository notebook dan dasbor: https://github.com/MILAS1818/KLASIFIKASI-CITRA-PAKAIAN-DENGAN-CNN-DARI-SCRATCH
## Anggota Kelompok
 
| NIM | Nama |
| --- | --- |
| 231111989 | Nicholas Jogi Nathanael |
| 231112641 | Rafela Wilmina Saputri |
| 231112230 | Athaya Naufal Brasta |
| 231112695 | Sukses Lumaban Gaol |
| 231112407 | Precylia Tan |
 
## Dataset
 
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) oleh P. Aggarwal (Kaggle), berisi foto produk katalog Myntra berukuran sekitar 80 × 60 piksel beserta metadata `styles.csv`.
 
- Label: kolom `articleType`, hanya kategori `masterCategory = Apparel`
- Kelas dengan kurang dari 400 gambar dibuang, kelas besar dibatasi maksimal 2.000 gambar
- Hasil akhir: **11.508 gambar, 11 kelas**: Bra, Briefs, Dresses, Jeans, Kurtas, Sarees, Shirts, Shorts, Tops, Trousers, Tshirts
- Pembagian *stratified* 70 / 15 / 15: 8.055 latih, 1.726 validasi, 1.727 uji
Dataset diunduh otomatis di dalam notebook memakai `kagglehub` (dapat meminta login Kaggle).
 
## Arsitektur dan Pelatihan
 
```
Input 80x60x3
 -> 3 blok: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool 2x2 -> Dropout 0.25   (filter 32, 64, 128)
 -> GlobalAveragePooling
 -> Dense 256 -> BatchNorm -> ReLU -> Dropout 0.5
 -> Dense 11 (softmax)
```
 
Total parameter: **324.971**.
 
| Pengaturan | Nilai |
| --- | --- |
| Loss | Categorical cross-entropy, label smoothing 0.1 |
| Optimizer | Adam, learning rate awal 1e-3 (RMSprop tersedia lewat `CONFIG["optimizer"]`) |
| Learning rate scheduling | `ReduceLROnPlateau` (faktor 0.5, patience 3, minimum 1e-6) |
| Early stopping | patience 8, `restore_best_weights=True` |
| Batch size / epoch maks. | 64 / 60 |
| Augmentasi (hanya data latih) | flip horizontal, kecerahan, kontras, saturasi, rotasi hingga 10° |
| Seed | 42 |
 
## Hasil pada Data Uji
 
| Metrik | Nilai |
| --- | --- |
| Accuracy | 92,59% |
| Macro F1 | 0,9319 |
| Weighted F1 | 0,9256 |
 
Kesalahan terbanyak terjadi pada kelas yang mirip secara visual, terutama Tshirts dengan Tops. Rincian lengkap ada pada laporan.
 
## Struktur Repository
 
Repository aplikasi web:
 
```
klasifikasi-pakaian/
├── index.html        # HTML, CSS, dan JavaScript aplikasi
├── vercel.json       # pengaturan cache untuk folder model
└── model/
    ├── model.onnx        # model hasil konversi tf2onnx (opset 13)
    └── class_names.json  # daftar 11 kelas
```
 
Repository notebook:
 
```
├── <notebook>.ipynb  # pelatihan, evaluasi, dasbor interaktif
└── outputs/          # dibuat otomatis oleh notebook
    ├── cnn_scratch_pakaian.keras
    ├── class_names.json
    ├── ringkasan_hasil.csv
    ├── classification_report.csv
    └── history.json
```
 
## Cara Menjalankan Notebook
 
1. Buka notebook di **Google Colab** atau Kaggle Notebook.
2. Aktifkan GPU (Colab: *Runtime > Change runtime type > GPU*).
3. Jalankan seluruh sel secara berurutan (*Run all*). Bagian 1 sampai 8 harus dijalankan lebih dulu sebelum dasbor (Bagian 12).
4. Hasil model dan laporan tersimpan di folder `outputs/`.
Pustaka utama: TensorFlow 2.x, NumPy, pandas, scikit-learn, matplotlib, seaborn, Pillow, kagglehub, ipywidgets.
 
Catatan: hasil pelatihan di GPU dapat sedikit berbeda antarjalan walaupun *seed* sudah dikunci, sehingga angka akurasi bisa bergeser tipis.
 
### Dasbor interaktif
 
Bagian 12 notebook menampilkan dasbor `ipywidgets` dengan tujuh tab: Ringkasan, Performa Kelas, Confusion Matrix, Galeri Prediksi, Uji Gambar, Analisis Kesalahan, dan Konfigurasi.
 
## Cara Menjalankan Aplikasi Web
 
Aplikasi berupa situs statis, jadi cukup dilayani oleh server HTTP sederhana (membuka `index.html` langsung dari berkas tidak akan memuat model).
 
```bash
git clone https://github.com/MILAS1818/klasifikasi-pakaian.git
cd klasifikasi-pakaian
python -m http.server 8000
```
 
Buka `http://localhost:8000`. Tunggu status **Model siap**, pilih satu foto (JPG, PNG, atau WebP), lalu tekan **Prediksi**.
 
Cara kerja di browser:
 
1. `model.onnx` dan `class_names.json` dimuat, lalu sesi ONNX Runtime Web (versi 1.20.1, dari CDN jsDelivr) dibuat.
2. Foto diubah ke ukuran 60 × 80 piksel, nilai piksel dibagi 255, lalu disusun menjadi tensor `[1, 80, 60, 3]`.
3. Hasil *softmax* diurutkan dan lima kelas teratas ditampilkan dalam diagram batang.
Seluruh proses berjalan di perangkat pengguna; foto tidak dikirim ke server.
 
## Ekspor ke ONNX
 
```python
model = keras.models.load_model("outputs/cnn_scratch_pakaian.keras")
model.export("saved_model")
```
 
```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model saved_model --output model.onnx --opset 13
```
 
Hasil ONNX dibandingkan dengan model Keras pada masukan acak untuk memastikan konversi tidak mengubah perilaku model.
 
## Deployment
 
Repository terhubung ke Vercel dengan *Framework Preset* **Other**. *Build Command*, *Output Directory*, dan *Install Command* dikosongkan karena aplikasi berupa berkas statis. Setiap *push* ke *branch* `main` otomatis diterbitkan ulang.
 
## Keterbatasan
 
- Data berupa foto produk berlatar polos beresolusi rendah; foto dengan orang, latar ramai, atau beberapa pakaian sekaligus lebih sering salah.
- Belum ada deteksi *out-of-distribution*: foto yang bukan pakaian tetap diberi salah satu dari 11 kelas.
- Hasil berasal dari satu kali pelatihan (seed 42), belum ada eksperimen *ablation* maupun multi-seed.
- *Resize* di browser memakai algoritma yang sedikit berbeda dari `tf.image.resize`, sehingga persentase keyakinan dapat bergeser tipis.
