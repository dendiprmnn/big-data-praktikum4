# big-data-praktikum-4

## IMPOR LIBRARY YANG DIPERLUKAN
- from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
- from pyspark.sql.functions import col, count, when, isnull, mean, lower, to_date, year, month, quarter
- from pyspark.ml.feature import Bucketizer, VectorAssembler, MinMaxScaler, StringIndexer, OneHotEncoder
- from pyspark.ml import Pipeline
- import pyspark.sql.functions as F

## TUGAS PRA-PEMROSESAN DATA PRODUK ELEKTRONIK 

<p> print("=" * 70)
print("TUGAS: PRA-PEMROSESAN DATA PRODUK ELEKTRONIK")
print("=" * 70) </p>

### 1. MEMBUAT DATAFRAME AWAL
print("\n1. DATASET AWAL:")
print("-" * 50)

data_produk = [
    (101, 'Laptop A', 'Elektronik', 15000000, 4.5, 120, '2023-01-20', 'stok_tersedia'),
    (102, 'Smartphone B', 'Elektronik', 8000000, 4.7, 250, '2023-02-10', 'stok_tersedia'),
    (103, 'Headphone C', 'Aksesoris', 1200000, 4.2, None, '2023-02-15', 'stok_habis'),
    (104, 'Laptop A', 'Elektronik', 15000000, 4.5, 120, '2023-01-20', 'stok_tersedia'),  # Duplikat
    (105, 'Tablet D', 'Elektronik', 6500000, None, 80, '2023-03-01', 'stok_tersedia'),
    (106, 'Charger E', 'Aksesoris', 250000, -4.0, 500, '2023-03-05', 'Stok_Tersedia'),  # Rating tidak valid & Status inkonsisten
    (107, 'Smartwatch F', 'Elektronik', 3100000, 4.8, 150, '2023-04-12', 'stok_habis')
]

skema_produk = StructType([
    StructField("id_produk", IntegerType()),
    StructField("nama_produk", StringType()),
    StructField("kategori", StringType()),
    StructField("harga", IntegerType()),
    StructField("rating", FloatType()),
    StructField("terjual", IntegerType()),
    StructField("tgl_rilis", StringType()),
    StructField("status_stok", StringType())
])

df_tugas = spark.createDataFrame(data=data_produk, schema=skema_produk)
print("DataFrame awal:")
df_tugas.show()
df_tugas.printSchema()

### 2. DATA CLEANING
print("\n2. DATA CLEANING")
print("-" * 50)

#### a. Identifikasi Missing Values
print("\na. Identifikasi Missing Values:")
missing_counts = df_tugas.select([count(when(isnull(c), c)).alias(c) for c in df_tugas.columns])
missing_counts.show()

#### b. Imputasi Missing Values
print("\nb. Imputasi Missing Values:")

#### Untuk kolom 'terjual' (numerik), kita isi dengan median
median_terjual = df_tugas.approxQuantile("terjual", [0.5], 0.01)[0]
print(f"Median terjual: {median_terjual}")

#### Untuk kolom 'rating' (numerik), kita isi dengan rata-rata rating yang valid
mean_rating = df_tugas.filter(df_tugas["rating"] > 0).select(mean("rating")).collect()[0][0]
print(f"Rata-rata rating valid: {mean_rating}")

df_clean = df_tugas.na.fill({
    'terjual': int(median_terjual),
    'rating': float(mean_rating)
})

print("Data setelah imputasi missing values:")
df_clean.show()

#### c. Hapus Data Duplikat
print("\nc. Menghapus Data Duplikat:")
print(f"Jumlah baris sebelum hapus duplikat: {df_clean.count()}")

df_clean = df_clean.dropDuplicates()

print(f"Jumlah baris setelah hapus duplikat: {df_clean.count()}")
print("Data setelah hapus duplikat:")
df_clean.show()

#### d. Perbaiki Rating Tidak Valid (negatif)
print("\nd. Perbaiki Rating Tidak Valid:")
print("Rating sebelum perbaikan:")
df_clean.select("id_produk", "rating").show()

df_clean = df_clean.withColumn(
    "rating", 
    when(col("rating") < 0, 0.0).otherwise(col("rating"))
)

print("Rating setelah perbaikan (rating negatif diubah menjadi 0):")
df_clean.select("id_produk", "rating").show()

#### e. Standarisasi Kolom status_stok
print("\ne. Standarisasi Kolom status_stok:")
print("Status stok sebelum standarisasi:")
df_clean.select("id_produk", "status_stok").show()

df_clean = df_clean.withColumn(
    "status_stok", 
    lower(col("status_stok"))
)

print("Status stok setelah standarisasi (huruf kecil semua):")
df_clean.select("id_produk", "status_stok").show()

### 3. DATA TRANSFORMASI
print("\n3. DATA TRANSFORMASI")
print("-" * 50)

#### a. Diskretisasi Harga (Binning)
print("\na. Diskretisasi Harga (Binning):")

#### Definisikan kategori harga
splits_harga = [0, 5000000, 10000000, float('inf')]

bucketizer_harga = Bucketizer(
    splits=splits_harga,
    inputCol="harga",
    outputCol="kategori_harga"
)

df_transformed = bucketizer_harga.transform(df_clean)

#### Tambahkan label kategori
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def map_kategori_harga(bin_idx):
    labels = ["Murah", "Sedang", "Mahal"]
    if bin_idx < len(labels):
        return labels[int(bin_idx)]
    return "Unknown"

map_udf = udf(map_kategori_harga, StringType())
df_transformed = df_transformed.withColumn("kategori_harga_label", map_udf(col("kategori_harga")))

print("Data setelah diskretisasi harga:")
df_transformed.select("id_produk", "nama_produk", "harga", "kategori_harga", "kategori_harga_label").show()

#### b. Normalisasi Numerik (Min-Max Scaling)
print("\nb. Normalisasi Kolom Numerik (Min-Max Scaling):")

#### Kolom numerik yang akan dinormalisasi
numeric_cols = ["harga", "rating", "terjual"]

#### Buat vektor dari kolom numerik
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_numeric")

#### Normalisasi dengan MinMaxScaler
scaler = MinMaxScaler(inputCol="features_numeric", outputCol="features_normalized")

#### Pipeline untuk assembler dan scaler
pipeline = Pipeline(stages=[assembler, scaler])

#### Fit dan transform
scaler_model = pipeline.fit(df_transformed)
df_transformed = scaler_model.transform(df_transformed)

print("Data setelah normalisasi (5 baris pertama):")
df_transformed.select("id_produk", "harga", "rating", "terjual", "features_normalized").show(5)

### 4. FEATURE ENGINEERING
print("\n4. FEATURE ENGINEERING")
print("-" * 50)

#### a. Ekstraksi Fitur dari Tanggal
print("\na. Ekstraksi Fitur dari Tanggal:")

df_engineered = df_transformed.withColumn(
    "tgl_rilis_timestamp", 
    to_date(col("tgl_rilis"), "yyyy-MM-dd")
)

#### Ekstrak tahun, bulan, dan kuartal
df_engineered = df_engineered.withColumn("tahun_rilis", year(col("tgl_rilis_timestamp")))
df_engineered = df_engineered.withColumn("bulan_rilis", month(col("tgl_rilis_timestamp")))
df_engineered = df_engineered.withColumn("kuartal_rilis", quarter(col("tgl_rilis_timestamp")))

print("Data setelah ekstraksi fitur tanggal:")
df_engineered.select("id_produk", "tgl_rilis", "tahun_rilis", "bulan_rilis", "kuartal_rilis").show()

#### b. One-Hot Encoding untuk Kategori dan Status Stok
print("\nb. One-Hot Encoding untuk Variabel Kategorikal:")

#### StringIndexer untuk kategori dan status_stok
indexer_kategori = StringIndexer(inputCol="kategori", outputCol="kategori_index")
indexer_stok = StringIndexer(inputCol="status_stok", outputCol="status_stok_index")

#### OneHotEncoder
encoder = OneHotEncoder(
    inputCols=["kategori_index", "status_stok_index"],
    outputCols=["kategori_ohe", "status_stok_ohe"]
)

#### Pipeline untuk encoding
pipeline_encode = Pipeline(stages=[indexer_kategori, indexer_stok, encoder])
encoder_model = pipeline_encode.fit(df_engineered)
df_encoded = encoder_model.transform(df_engineered)

print("Data setelah One-Hot Encoding (5 baris pertama):")
df_encoded.select("id_produk", "kategori", "kategori_ohe", "status_stok", "status_stok_ohe").show(5, truncate=False)

### 5. FITUR INTERAKSI DAN FITUR LAINNYA
print("\n5. FITUR TAMBAHAN")
print("-" * 50)

#### a. Fitur interaksi: harga_per_unit_terjual
df_final = df_encoded.withColumn(
    "harga_per_unit_terjual",
    col("harga") / col("terjual")
)

#### b. Kategorisasi rating
df_final = df_final.withColumn(
    "kategori_rating",
    when(col("rating") >= 4.5, "Sangat Baik")
    .when(col("rating") >= 4.0, "Baik")
    .when(col("rating") >= 3.0, "Cukup")
    .otherwise("Kurang")
)

print("Data dengan fitur tambahan:")
df_final.select("id_produk", "nama_produk", "harga", "terjual", "harga_per_unit_terjual", "rating", "kategori_rating").show()

### 6. HASIL AKHIR
print("\n" + "=" * 70)
print("HASIL AKHIR: DATAFRAME SETELAH PRA-PEMROSESAN LENGKAP")
print("=" * 70)

#### Tampilkan 10 baris pertama dengan kolom-kolom penting
print("\nDataFrame akhir (10 baris pertama):")

#### Pilih kolom-kolom representatif untuk ditampilkan
display_cols = [
    "id_produk", "nama_produk", "kategori", "harga", 
    "rating", "terjual", "tgl_rilis", "status_stok",
    "bulan_rilis", "kategori_harga_label", "kategori_rating"
]

df_final.select(display_cols).show(10, truncate=False)

#### Tampilkan skema akhir
print("\nSkema DataFrame akhir:")
df_final.printSchema()

#### Tampilkan statistik ringkasan
print("\nStatistik Ringkasan:")
df_final.select("harga", "rating", "terjual").summary("count", "mean", "stddev", "min", "max").show()

print("\n" + "=" * 70)
print("TUGAS SELESAI: PRA-PEMROSESAN DATA PRODUK ELEKTRONIK")
print("=" * 70)
