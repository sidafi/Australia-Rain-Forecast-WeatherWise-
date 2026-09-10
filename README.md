# Australia Rainfall Prediction 🌦️🇦🇺

Project pembelajaran **Machine Learning** untuk memprediksi kemungkinan hujan di Australia menggunakan dataset **WeatherAUS**.

Aplikasi dibuat menggunakan **Python + Streamlit** dan menggunakan dua model machine learning yang telah dilatih:

* `model_raintomorrow.pkl` → prediksi hujan besok

## Features

* Input observasi cuaca
* Prediksi hujan hari ini
* Prediksi hujan besok
* Probability prediction
* Tampilan cuaca berdasarkan hasil prediksi
* Interface berbasis Streamlit

## Dataset

Dataset yang digunakan:

```text
weatherAUS_ready(1).csv
```

Dataset telah melalui proses **data cleaning dan Exploratory Data Analysis (EDA)** sebelum digunakan untuk training model.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Kemudian jalankan:

```bash
streamlit run app.py
```

Aplikasi akan tersedia di:

```text
http://localhost:8501
```

Untuk mengakses dari perangkat lain dalam jaringan yang sama:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

## Learning Objective

Project ini dibuat sebagai latihan untuk memahami alur sederhana penerapan Machine Learning:

```text
Dataset
   ↓
EDA & Data Cleaning
   ↓
Model Training
   ↓
Model Export (.pkl)
   ↓
User Input
   ↓
Prediction
   ↓
Streamlit Web Application
```

---

**Machine Learning Practice — Weather Prediction Australia**
