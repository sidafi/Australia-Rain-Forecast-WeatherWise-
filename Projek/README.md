# WeatherWise

WeatherWise is a Streamlit web application for predicting whether it will rain tomorrow at an Australian weather station. The application uses a trained Random Forest classifier, displays the estimated rain probability, explains the main weather factors, and stores forecast history locally in SQLite.

## Features

- Predicts tomorrow's rain condition with one trained model.
- Supports Australian weather stations and wind directions from the model schema.
- Provides manual numeric inputs and synchronized sliders.
- Validates values against the accepted model ranges.
- Displays a clear or rainy animated forecast based on the prediction.
- Shows all submitted observation conditions in a table.
- Stores completed forecasts in a local SQLite database.
- Displays history as readable tables instead of raw JSON.
- Protects history deletion with password verification and confirmation.
- Includes responsive layout rules for desktop and mobile screens.

## Machine Learning Model

The application loads `model_cuaca_rf.pkl`, a saved scikit-learn Random Forest classifier. The model's `feature_names_in_` attribute is used as the source of truth for feature construction and one-hot encoding.

The current model uses these observation inputs:

- `Location`
- `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`
- `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`
- `Humidity9am`, `Humidity3pm`
- `Pressure3pm`
- `Cloud9am`, `Cloud3pm`
- `WindGustDir`, `WindDir9am`, `WindDir3pm`
- `RainToday`

`RainTomorrow` is the prediction target and is not accepted as an input.

## Dataset

The reference dataset is available in the parent project folders:

```text
../Proses EDA/weatherAUS_ready.csv
```

The dataset contains Australian weather observations with `RainTomorrow` as the target label. The application itself uses the already-trained model artifact and does not retrain the model when it starts.

## Installation

From the `Projek` directory, create or activate a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

To access the application from another device on the same network, run Streamlit with a network-accessible address and open the computer's local IP address from the phone:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Example:

```text
http://192.168.1.10:8501
```

The computer and phone must be connected to the same network, and Windows Firewall may need to allow the Streamlit port.

## Project Structure

```text
Projek/
├── app.py                 # Streamlit application
├── model_cuaca_rf.pkl     # Trained Random Forest model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── weather.db             # Local SQLite history, created at runtime
└── assets/                # Optional local assets
```

## Prediction Flow

1. Load the saved Random Forest model.
2. Collect station, rain, temperature, humidity, pressure, cloud, and wind inputs.
3. Synchronize each manual numeric input with its slider.
4. Validate the submitted values and ensure `MinTemp` is not greater than `MaxTemp`.
5. Normalize categorical values and construct the exact model feature row.
6. Run `predict()` and `predict_proba()`.
7. Display the forecast, probability, explanation, animation, and observation table.
8. Save the successful forecast to SQLite.

## Forecast History

The application creates `weather.db` automatically beside `app.py`. Each forecast is stored in the `forecast_history` table with:

- timestamp
- location
- observation inputs
- prediction label
- prediction probability

The **Forecast history** page displays each saved forecast in an expandable table. Clearing history requires the configured administrator password and a second confirmation step.

## Security Note

Before publishing this repository publicly, change the history-clear password in `app.py` or move it to Streamlit secrets or an environment variable. Do not commit personal credentials, API tokens, or private deployment configuration.

## Troubleshooting

- Confirm `model_cuaca_rf.pkl` is in the same directory as `app.py`.
- Run the commands from the `Projek` directory.
- Install dependencies with `pip install -r requirements.txt`.
- If Streamlit is not recognized, activate the Python environment where Streamlit is installed.
- If the model feature schema changes, retrain or update the preprocessing logic so it matches `feature_names_in_`.
