# Australia Rainfall Prediction 🌦️🇦🇺

A **Machine Learning learning project** to predict the possibility of rainfall in Australia using the **WeatherAUS dataset**.

This application was developed using **Python + Streamlit** and uses a trained machine learning model:

* `model_raintomorrow.pkl` → predicts whether it will rain tomorrow

## Features

* Weather observation input
* Rainfall prediction for tomorrow
* Prediction probability
* Weather visualization based on prediction results
* Interactive interface built with Streamlit

## Dataset

The dataset used in this project:

```text
weatherAUS_ready(1).csv
```

The dataset has gone through **data cleaning and Exploratory Data Analysis (EDA)** processes before being used for model training.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

To access the application from other devices on the same network:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

## Learning Objective

This project was created as a practice to understand the basic workflow of implementing a Machine Learning system:

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

**Machine Learning Practice — Australia Weather Rainfall Prediction**
