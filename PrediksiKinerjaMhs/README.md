## Rayhan Akbar Pradana
## A11.2022.14085

# Prediksi Kinerja Mahasiswa

## Ringkasan dan Permasalahan

Proyek ini memahami bagaimana kinerja siswa (nilai tes) dipengaruhi oleh variabel lain seperti Jenis Kelamin, Etnis, Tingkat pendidikan orang tua pendidikan orang tua, Lunch dan kursus persiapan ujian.

## Tujuan

Memprediksi kinerja akademik mahasiswa berdasarkan beberapa fitur untuk mengidentifikasi mahasiswa yang berpotensi mengalami kesulitan akademik.

## Model

Penggunaan model regresi linier yang diperkuat dengan jaringan saraf tiruan (Neural Network).

## Alur Project

1. Pengumpulan Data : Mendownload dataset dari Kaggle.
2. Eksplorasi Data: Memvisualisasikan dan memahami data.
3. Pra-pemrosesan Data: Normalisasi dan pembersihan data.
4. Feature Engineering: Menambahkan fitur baru yang relevan.
5. Modeling: Membangun dan melatih model regresi dan neural network.
6. Evaluasi: Mengukur performa model dengan metrik seperti MAE dan RMSE.
7. Visualisasi Hasil: Menampilkan hasil prediksi dan residu.
## Dataset

![App Screenshot](./image/image.png)

Penjelasan Dataset, EDA dan Proses features Dataset
Dataset

- Sumber: Dataset Mdari Kaggle
- Deskripsi: Dataset ini mencakup informasi terperinci tentang kinerja akademik siswa di berbagai mata pelajaran, bersama dengan faktor demografis dan sosial. Dataset ini mencakup jenis kelamin (Pria atau Wanita), ras/etnis (Kelompok A, B, C, D, E), dan tingkat pendidikan orang tua (Sarjana, beberapa perguruan tinggi, sarjana, master, sarjana muda, sekolah menengah atas). Data ini juga mencakup informasi mengenai apakah siswa menerima makan siang standar atau gratis/berkurang, apakah mereka mengikuti kursus persiapan ujian, dan nilai mereka dalam matematika, membaca, dan menulis. Dataset ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi keberhasilan akademik siswa dengan memberikan pandangan komprehensif tentang latar belakang dan lingkungan pendidikan mereka.

### EDA (Exploratory Data Analysis)??

??

### Proses features Dataset

Import paket-paket yang dibutuhkan seperti numpy, pandas, seaborn, dan matplotlib.
Membaca dataset dari file CSV ke dalam DataFrame menggunakan pandas.
Memeriksa jumlah nilai kosong dan baris duplikat dalam dataset untuk memastikan data tidak ada yang hilang dan unik.
Menambahkan kolom fitur baru seperti total score yang merupakan jumlah dari nilai matematika, membaca, dan menulis.
Menambahkan kolom average yang merupakan rata-rata dari total score.
Menggunakan StandardScaler untuk menskalakan fitur numerik dalam dataset.
Membagi data menjadi set pelatihan (training) dan pengujian (testing) dengan proporsi 80-20 menggunakan train_test_split.

## Proses Learning/Modeling

1. Pra-pemrosesan Data

?????

2. Membangun Model
    Membuat model Linear Regression menggunakan sklearn.linear_model.
`from sklearn.linear_model import LinearRegression`
`model = LinearRegression(fit_intercept=True)`

3. Melatih Model
    Melatih model dengan data training.
`model = model.fit(X_train, y_train)`

4. Memprediksi dan Mengevaluasi Model
    Memprediksi nilai dengan data testing dan mengevaluasi performa model menggunakan metrik seperti mean_absolute_error, mean_squared_error, dan r2_score.
`y_pred = model.predict(X_test)`
`from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score`

`def evaluate_model(true, predicted):`
    `mae = mean_absolute_error(true, predicted)`
    `mse = mean_squared_error(true, predicted)`
    `rmse = np.sqrt(mean_squared_error(true, predicted))`
    `r2_square = r2_score(true, predicted)`
    `return mae, rmse, r2_square`

`mae, rmse, r2 = evaluate_model(y_test, y_pred)`
`print('Model performance for Training set')`
`print("- Root Mean Squared Error: {:.4f}".format(rmse))`
`print("- Mean Absolute Error: {:.4f}".format(mae))`
`print("- R2 Score: {:.4f}".format(r2))`

5. Visualisasi Hasil
    Membuat plot scatter dan regression untuk membandingkan nilai aktual dengan prediksi.

`plt.scatter(y_test, y_pred)`
`plt.xlabel('Actual')`
`plt.ylabel('Predicted')`

`sns.regplot(x=y_test, y=y_pred, ci=None, color='red')`

## Perfoma Model

??

## Diskusi Hasil dan Kesimpulan

### Hasil

??

## Kesimpulan

![App Screenshot](./image/output.png)

??