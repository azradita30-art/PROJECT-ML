# Klasifikasi Penyakit Genetik Menggunakan Machine Learning

## Deskripsi Project

Project ini bertujuan membangun model *machine learning* untuk mengklasifikasikan kategori penyakit genetik berdasarkan karakteristik pasien, riwayat keluarga, hasil pemeriksaan laboratorium, dan biomarker genetik. Permasalahan yang digunakan adalah **klasifikasi multikelas** dengan variabel target `Disease`.

Tahapan analisis yang dilakukan meliputi:

1. *Business understanding*
2. *Data understanding*
3. *Exploratory Data Analysis* (EDA)
4. *Data preprocessing*
5. *Data construction*
6. Pemodelan menggunakan beberapa algoritma klasifikasi
7. *Hyperparameter tuning*
8. Interpretasi hasil menggunakan *feature importance* dan SHAP
9. Evaluasi model

Model yang dibandingkan dalam project ini adalah:

- **XGBoost Classifier**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

Project ini dibuat untuk keperluan tugas mata kuliah **Machine Learning**.

## Anggota Tim

| No. | Nama | NIM |
|---|---|---|
| 1 | Azra Dita Nurcahyati | K1D024041 |
| 2 | Nayla Edenine Qohar | K1D024044 |
| 3 | Aura Salsabila | K1D024047 |
| 4 | Naufal Zaqie Al Faruq | K1D024050 |

## Sumber Dataset

Dataset yang digunakan adalah **Genetic Disease Dataset** yang berasal dari **Kaggle**.

Informasi utama dataset:

| Karakteristik | Keterangan |
|---|---|
| Nama dataset | Genetic Disease Dataset |
| Jumlah observasi | 1.000 data |
| Jumlah variabel | 13 variabel |
| Jumlah fitur prediktor | 12 fitur |
| Variabel target | `Disease` |
| Jenis masalah | Klasifikasi multikelas |
| Tipe data | Numerik dan kategorik biner |

Fitur yang digunakan dalam pemodelan:

- `Age`
- `Gender`
- `Family_History`
- `Hemoglobin`
- `Fetal_Hemoglobin`
- `RDW_CV`
- `Serum_Ferritin`
- `BRCA1_Expression`
- `p53_Mutation`
- `Sweat_Chloride`
- `Sickled_RBC_Percent`
- `IL6_Level`

Target klasifikasi adalah `Disease`, yaitu kategori penyakit genetik yang dikodekan dalam lima kelas.


## Struktur File

Struktur file yang digunakan dalam project ini dapat disusun sebagai berikut:

```text
.
├── PROJEKMLGROUPH.ipynb
├── README.md
└── data/
    └── genetic_disease_dataset.csv
```

Keterangan:

- `PROJEKMLGROUPH.ipynb` berisi kode analisis, visualisasi, preprocessing, pemodelan, tuning, evaluasi, dan interpretasi model.
- `README.md` berisi dokumentasi project.
- Folder `data/` dapat digunakan untuk menyimpan file dataset CSV. Nama file dataset dapat disesuaikan dengan file yang diunduh dari Kaggle.

## Library yang Digunakan

Project ini menggunakan beberapa library Python berikut:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

Untuk menjalankan secara lokal, library dapat diinstal dengan perintah:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Cara Menjalankan Project

### 1. Menjalankan di Google Colab

Notebook yang diberikan menggunakan fitur upload file dari Google Colab. Langkah menjalankannya:

1. Buka file `PROJEKMLGROUPH.ipynb` di Google Colab.
2. Jalankan cell import library.
3. Saat muncul perintah upload file, unggah file dataset CSV dari Kaggle.
4. Jalankan seluruh cell secara berurutan dari awal sampai akhir.
5. Periksa output EDA, preprocessing, evaluasi model, dan interpretasi hasil.

Bagian kode upload dataset pada notebook:

```python
from google.colab import files
import io

upload = files.upload()
uploaded_file_name = next(iter(upload.keys()))

df = pd.read_csv(uploaded_file_name)
df.head()
```

### 2. Menjalankan di Jupyter Notebook Lokal

Apabila notebook dijalankan secara lokal, bagian upload file dari Google Colab perlu diganti dengan pembacaan file CSV secara langsung.

Contoh:

```python
import pandas as pd

df = pd.read_csv("data/genetic_disease_dataset.csv")
df.head()
```

Setelah itu, jalankan seluruh cell notebook secara berurutan.

## Alur Analisis

### 1. Data Understanding

Pada tahap ini dilakukan pemeriksaan struktur data, tipe data, statistik deskriptif, distribusi target, dan visualisasi data.

Hasil awal menunjukkan bahwa dataset memiliki 1.000 observasi dan 13 variabel. Variabel target `Disease` memiliki lima kelas yang distribusinya relatif seimbang, sehingga tidak diperlukan teknik penyeimbangan data seperti SMOTE, oversampling, atau undersampling.

### 2. Exploratory Data Analysis

EDA dilakukan menggunakan:

- Statistik deskriptif
- Histogram
- Boxplot
- Pie chart untuk variabel kategorik biner
- Heatmap korelasi antarvariabel
- Heatmap korelasi fitur terhadap target

Beberapa variabel yang memiliki hubungan cukup kuat dengan target `Disease` adalah:

- `Sweat_Chloride`
- `Serum_Ferritin`
- `IL6_Level`
- `Sickled_RBC_Percent`
- `Hemoglobin`

### 3. Data Preprocessing

Tahap preprocessing meliputi:

- Pemeriksaan missing value
- Pemeriksaan data duplikat
- Penanganan outlier
- Standardisasi data numerik

Hasil preprocessing:

- Tidak ditemukan missing value.
- Tidak ditemukan data duplikat.
- Outlier ditangani menggunakan metode **Interquartile Range (IQR)** dengan teknik capping.
- Variabel numerik distandardisasi menggunakan **StandardScaler**.

### 4. Data Construction

Dataset dipisahkan menjadi:

- `X`: variabel prediktor
- `y`: variabel target `Disease`

Data kemudian dibagi menjadi data latih dan data uji dengan proporsi:

| Jenis Data | Jumlah |
|---|---:|
| Data latih | 800 |
| Data uji | 200 |

Pembagian data menggunakan `train_test_split` dengan `test_size=0.2` dan `random_state=42`.

### 5. Pemodelan

Model yang digunakan:

1. XGBoost Classifier
2. SVM
3. Random Forest Classifier

Setiap model diuji dalam dua skenario:

- Model awal/default
- Model setelah *hyperparameter tuning*

Hyperparameter tuning dilakukan menggunakan **GridSearchCV** dengan validasi silang 5-fold.
## Interpretasi Feature Importance

Berdasarkan hasil *feature importance* dan SHAP, variabel yang paling berpengaruh dalam klasifikasi penyakit genetik adalah:

1. `Sickled_RBC_Percent`
2. `Fetal_Hemoglobin`
3. `Sweat_Chloride`
4. `BRCA1_Expression`
5. `IL6_Level`
6. `Serum_Ferritin`

Variabel `Sickled_RBC_Percent` menjadi fitur dengan kontribusi paling besar terhadap hasil prediksi model. Hal ini menunjukkan bahwa karakteristik hematologis dan biomarker biologis memiliki peran penting dalam membedakan kategori penyakit genetik pada dataset yang digunakan.

## Ringkasan Hasil

Perbandingan performa model berdasarkan output notebook:

| Model | Accuracy | Precision Macro | Recall Macro | F1-Score Macro | ROC AUC Macro |
|---|---:|---:|---:|---:|---:|
| XGBoost Awal | 0.930 | 0.93 | 0.93 | 0.93 | 0.971515 |
| XGBoost Tuning | 0.955 | 0.96 | 0.95 | 0.96 | 0.978041 |
| SVM Awal | 0.950 | 0.95 | 0.95 | 0.95 | 0.972485 |
| SVM Tuning | 0.955 | 0.96 | 0.95 | 0.96 | 0.970340 |
| Random Forest Awal | 0.955 | 0.96 | 0.95 | 0.96 | 0.972222 |
| Random Forest Tuning | 0.950 | 0.95 | 0.95 | 0.95 | 0.973260 |

Berdasarkan evaluasi keseluruhan, model terbaik adalah **XGBoost Tuning** karena memiliki kombinasi performa paling kuat, terutama pada nilai **ROC AUC Macro** tertinggi sebesar **0.978041**, dengan **accuracy 0.955** dan **F1-score macro 0.96**.

Parameter terbaik XGBoost berdasarkan hasil tuning:

```python
{
    "colsample_bytree": 0.7,
    "learning_rate": 0.01,
    "max_depth": 7,
    "n_estimators": 50,
    "subsample": 0.9
}
```

## Kesimpulan

Project ini berhasil menerapkan workflow machine learning secara menyeluruh, mulai dari EDA, preprocessing, data construction, pemodelan, tuning, evaluasi, hingga interpretasi model. Dari beberapa algoritma yang diuji, **XGBoost dengan hyperparameter tuning** dipilih sebagai model terbaik karena menghasilkan performa evaluasi paling optimal.

Hasil model ini hanya berlaku untuk dataset yang digunakan dalam project dan **tidak dimaksudkan sebagai alat diagnosis medis langsung**. Model dapat dikembangkan lebih lanjut sebagai prototipe sistem pendukung keputusan atau alat bantu skrining awal, dengan syarat diuji kembali menggunakan data medis nyata yang lebih luas dan tervalidasi.

## Reproducibility

Agar hasil dapat direplikasi, pastikan:

- Dataset yang digunakan sama dengan dataset pada project.
- Seluruh cell notebook dijalankan secara berurutan.
- Parameter `random_state=42` tetap digunakan pada proses split data dan model.
- Library yang digunakan telah terinstal dengan benar.
- Apabila dijalankan secara lokal, kode upload Google Colab diganti dengan `pd.read_csv()` sesuai lokasi file dataset.
