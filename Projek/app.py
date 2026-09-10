from datetime import datetime
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_cuaca_rf.pkl"
SCALER_PATH = APP_DIR / "scaler_cuaca.pkl"
DATABASE_PATH = APP_DIR / "weather.db"
HISTORY_CLEAR_PASSWORD = "MuhammadGunata2006"

st.set_page_config(page_title="WeatherWise | Tomorrow's Weather Forecast", page_icon="🌧️", layout="wide")


def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location TEXT NOT NULL,
                observation_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def get_model_feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Model tidak memiliki feature_names_in_.")
    return list(names)


def get_location_options(model):
    features = get_model_feature_names(model)
    options = sorted({name.replace("Location_", "") for name in features if name.startswith("Location_")})
    return options


def get_direction_options(model):
    features = get_model_feature_names(model)
    direction_values = sorted({name.split("_", 1)[1] for name in features if name.startswith("WindGustDir_")})
    return direction_values


def infer_numeric_columns(model_features):
    numeric = []
    for name in model_features:
        if name == "RainToday":
            continue
        if name.startswith("Location_") or name.startswith("WindGustDir_") or name.startswith("WindDir9am_") or name.startswith("WindDir3pm_"):
            continue
        numeric.append(name)
    return numeric


def normalize_observation(raw_values):
    payload = dict(raw_values)
    rainfall = float(payload.get("Rainfall", 0.0))
    payload["RainToday"] = "Yes" if rainfall > 1.0 else "No"
    return payload


def build_feature_row(raw_values, model_features, scaler=None):
    values = normalize_observation(raw_values)
    row = {name: 0.0 for name in model_features}
    numeric_columns = infer_numeric_columns(model_features)

    if scaler is not None:
        numeric_dict = {column: float(values[column]) for column in numeric_columns if column in values}
        if numeric_dict:
            df = pd.DataFrame([numeric_dict])
            scaled = scaler.transform(df)
            for index, column in enumerate(numeric_dict.keys()):
                row[column] = float(scaled[0, index])
    else:
        for column in numeric_columns:
            if column in values:
                row[column] = float(values[column])

    row["RainToday"] = 1.0 if values["RainToday"] == "Yes" else 0.0

    for name in model_features:
        if name.startswith("Location_"):
            option = name.replace("Location_", "")
            row[name] = 1.0 if values.get("Location") == option else 0.0
        elif name.startswith("WindGustDir_"):
            option = name.replace("WindGustDir_", "")
            row[name] = 1.0 if values.get("WindGustDir") == option else 0.0
        elif name.startswith("WindDir9am_"):
            option = name.replace("WindDir9am_", "")
            row[name] = 1.0 if values.get("WindDir9am") == option else 0.0
        elif name.startswith("WindDir3pm_"):
            option = name.replace("WindDir3pm_", "")
            row[name] = 1.0 if values.get("WindDir3pm") == option else 0.0

    return pd.DataFrame([row], columns=model_features)


def predict_rain_tomorrow(values, model, scaler=None):
    model_features = get_model_feature_names(model)
    frame = build_feature_row(values, model_features, scaler=scaler)
    prediction_int = int(model.predict(frame)[0])
    prediction_label = "It might rain tomorrow" if prediction_int == 1 else "It probably won't rain tomorrow"

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][1])
    return prediction_label, prediction_int, probability, frame


def render_saved_history():
    st.subheader("Prediction History")
    clear_requested = st.button("Clear history", type="secondary", use_container_width=False)
    if clear_requested:
        st.session_state["request_clear_history"] = True
        st.session_state["history_clear_authenticated"] = False
        st.session_state.pop("history_clear_password_error", None)

    if st.session_state.get("request_clear_history") and not st.session_state.get("history_clear_authenticated"):
        st.info("Enter the administrator password to continue.")
        password_col, verify_col = st.columns([3, 1])
        with password_col:
            clear_password = st.text_input(
                "Password",
                type="password",
                key="history_clear_password",
                label_visibility="collapsed",
                placeholder="Enter password",
            )
        with verify_col:
            verify_password = st.button("Verify password", use_container_width=True)

        if verify_password:
            if clear_password == HISTORY_CLEAR_PASSWORD:
                st.session_state["history_clear_authenticated"] = True
                st.session_state.pop("history_clear_password_error", None)
                st.rerun()
            else:
                st.session_state["history_clear_password_error"] = "Incorrect password. History was not changed."

        if st.session_state.get("history_clear_password_error"):
            st.error(st.session_state["history_clear_password_error"])

    if st.session_state.get("history_clear_authenticated"):
        st.warning("Password accepted. This will permanently remove all saved forecasts.")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm clear", type="primary", use_container_width=True):
                with sqlite3.connect(DATABASE_PATH) as conn:
                    conn.execute("DELETE FROM forecast_history")
                    conn.commit()
                st.session_state["request_clear_history"] = False
                st.session_state["history_clear_authenticated"] = False
                st.session_state.pop("history_clear_password", None)
                st.success("Forecast history cleared.")
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state["request_clear_history"] = False
                st.session_state["history_clear_authenticated"] = False
                st.session_state.pop("history_clear_password", None)
                st.rerun()

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM forecast_history ORDER BY id DESC"
        ).fetchall()

    if not rows:
        st.info("No forecast data has been saved yet.")
        return

    for row in rows:
        observation = json.loads(row["observation_json"])
        result = json.loads(row["result_json"])
        with st.expander(f"{row['location']} • {row['created_at']}", expanded=False):
            result_col, probability_col = st.columns(2)
            with result_col:
                st.markdown("**Tomorrow forecast**")
                st.write(result["label"])
            with probability_col:
                st.markdown("**Rain probability**")
                st.write(f"{result['probability']:.2%}")

            history_table = pd.DataFrame(
                [{"Input field": field, "Value": str(value)} for field, value in observation.items()]
            )
            st.markdown("**Observation inputs**")
            st.dataframe(history_table, use_container_width=True, hide_index=True)


def save_prediction(values, result):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location TEXT NOT NULL,
                observation_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO forecast_history (created_at, location, observation_json, result_json) VALUES (?, ?, ?, ?)",
            (
                result["timestamp"],
                values["Location"],
                json.dumps(values, ensure_ascii=False),
                json.dumps({"label": result["label"], "probability": result["probability"], "value": result["value"]}, ensure_ascii=False),
            ),
        )
        conn.commit()


MODEL_VALID_RANGES = {
    "MinTemp": (-15.0, 50.0),
    "MaxTemp": (-10.0, 60.0),
    "Rainfall": (0.0, 400.0),
    "Evaporation": (0.0, 40.0),
    "Sunshine": (0.0, 15.0),
    "WindGustSpeed": (0.0, 150.0),
    "WindSpeed9am": (0.0, 100.0),
    "WindSpeed3pm": (0.0, 100.0),
    "Humidity9am": (0.0, 100.0),
    "Humidity3pm": (0.0, 100.0),
    "Pressure3pm": (970.0, 1050.0),
    "Cloud9am": (0.0, 9.0),
    "Cloud3pm": (0.0, 9.0),
}


def validate_input_ranges(values):
    errors = []
    for field, (lower, upper) in MODEL_VALID_RANGES.items():
        if field not in values:
            continue
        value = float(values[field])
        if value < lower or value > upper:
            errors.append(f"{field} must be between {lower:.1f} and {upper:.1f}.")
    if "MinTemp" in values and "MaxTemp" in values and float(values["MinTemp"]) > float(values["MaxTemp"]):
        errors.append("MinTemp cannot be greater than MaxTemp.")
    return errors


def build_reason_summary(values, prediction_label, probability):
    reasons = []
    rainfall = float(values.get("Rainfall", 0.0))
    humidity_3pm = float(values.get("Humidity3pm", 0.0))
    humidity_9am = float(values.get("Humidity9am", 0.0))
    cloud_3pm = float(values.get("Cloud3pm", 0.0))
    sunshine = float(values.get("Sunshine", 0.0))
    pressure_3pm = float(values.get("Pressure3pm", 1015.0))

    if rainfall > 1.0:
        reasons.append("Rainfall is above 1 mm, which strongly suggests rain is already occurring or likely to build up.")
    if humidity_3pm >= 70:
        reasons.append("High humidity at 3pm means the air is close to saturation, a common trigger for rain development.")
    if humidity_9am >= 70:
        reasons.append("Humidity early in the day is already elevated, which keeps moisture in the atmosphere and increases tomorrow's rain chances.")
    if cloud_3pm >= 6:
        reasons.append("Cloud cover is high, so the atmosphere is retaining moisture and sunlight is being blocked.")
    if sunshine <= 3:
        reasons.append("Sunshine is low, which reduces evaporation and usually supports a wetter pattern.")
    if pressure_3pm < 1010:
        reasons.append("Pressure is below the typical dry-weather range, which often points to unstable and wetter air masses.")
    if humidity_3pm < 45 and cloud_3pm <= 3 and sunshine > 6 and rainfall <= 1.0:
        reasons.append("The air is relatively dry and the sky is clearing, which makes a dry forecast more probable.")

    if not reasons:
        reasons.append("The current conditions are relatively balanced, so the model is leaning toward a moderate dry trend.")

    if not forecast_is_rain(prediction_label):
        reasons.insert(0, "The model sees a drier atmosphere overall, so the risk of rain tomorrow is reduced.")
    else:
        reasons.insert(0, "The model is detecting a wetter atmospheric setup, which raises the chance of rain tomorrow.")

    return " ".join(reasons[:3])


def forecast_is_rain(prediction_label):
    normalized_label = prediction_label.lower()
    dry_phrases = ("won't rain", "will not rain", "tidak hujan")
    return not any(phrase in normalized_label for phrase in dry_phrases)


def build_forecast_explanation(values, prediction_label, probability):
    probability_text = f"{probability:.0%}"
    humidity = float(values.get("Humidity3pm", 0.0))
    cloud = float(values.get("Cloud3pm", 0.0))
    pressure = float(values.get("Pressure3pm", 1015.0))
    rainfall = float(values.get("Rainfall", 0.0))

    if forecast_is_rain(prediction_label):
        outlook = f"The model estimates a {probability_text} rain probability for tomorrow."
        atmosphere = "The atmosphere is trending wetter" if humidity >= 65 or cloud >= 6 or pressure < 1010 else "Some moisture signals are present, but the setup is not strongly saturated"
    else:
        outlook = f"The model estimates a {probability_text} rain probability, so tomorrow is more likely to stay dry."
        atmosphere = "The atmosphere is relatively dry and stable" if humidity < 65 and cloud < 6 and pressure >= 1010 else "The atmosphere has some moisture, but the dry signals are stronger overall"

    recent_rain = " Recent rainfall is also included as a near-term moisture signal." if rainfall > 1.0 else " No meaningful rainfall was recorded in the submitted observation."
    return f"{outlook} {atmosphere}.{recent_rain} The probability is an estimate from the trained Random Forest, not a guarantee."


def render_result_card(title, label, probability):
    is_rain = forecast_is_rain(label)
    color = "#f38c78" if is_rain else "#67b7e7"
    probability_text = "N/A" if probability is None else f"{probability:.2%}"
    st.markdown(
        f"""
        <div style="background: rgba(21,35,53,.9); border:1px solid #35506d; border-top:4px solid {color}; border-radius:14px; padding:1.1rem; min-height:190px;">
            <div style="font-size:.7rem; letter-spacing:.15em; text-transform:uppercase; color:#9abbd6; font-weight:700;">{title}</div>
            <div style="font-size:1.7rem; font-weight:700; color:{color}; margin-top:.8rem;">{label}</div>
            <div style="margin-top:1rem; color:#dfeaf7; font-size:.85rem;">Probability: <strong>{probability_text}</strong></div>
            <div style="height:8px; background: rgba(255,255,255,.08); border-radius:999px; overflow:hidden; margin-top:1rem;">
                <div style="width:{(probability if probability is not None else 0) * 100:.1f}%; height:100%; border-radius:999px; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_city_background(location):
    city_pool = [
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80",
        "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d8?auto=format&fit=crop&w=1600&q=80",
        "https://images.unsplash.com/photo-1517836357463-d25dfeac3438?auto=format&fit=crop&w=1600&q=80",
        "https://images.unsplash.com/photo-1493246507139-91e8fad9978e?auto=format&fit=crop&w=1600&q=80",
        "https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1800&q=80",
    ]
    seed = sum(ord(ch) for ch in str(location))
    return city_pool[seed % len(city_pool)]


def estimate_tomorrow_temperature(values, prediction_label):
    rain = "Hujan" in prediction_label
    base = float(values["MaxTemp"]) * 0.7 + float(values["MinTemp"]) * 0.3
    humidity = float(values["Humidity3pm"])
    cloud = float(values["Cloud3pm"])
    wind = float(values["WindSpeed3pm"])
    delta = 0.0
    delta += (humidity - 50) * 0.03
    delta += (4 - cloud) * 0.4
    delta += (wind - 20) * 0.02
    delta += 0.9 if rain else -0.9
    return round(base + delta, 1)


def render_weather_overview(values, prediction_label, probability):
    background = get_city_background(values.get("Location", "Sydney"))
    tomorrow_temp = estimate_tomorrow_temperature(values, prediction_label)
    weather_class = "rainy" if forecast_is_rain(prediction_label) else "clear"
    summary_text = build_forecast_explanation(values, prediction_label, probability)
    st.markdown(
        f"""
        <style>
            .weather-overview {{ position: relative; overflow: hidden; border: 1px solid #2d415d; border-radius: 18px; min-height: 430px; margin: 1.2rem 0 1.5rem; background: url('{background}') center/cover no-repeat; }}
            .weather-overview::before {{ content: ''; position: absolute; inset: 0; background: linear-gradient(110deg, rgba(8,18,28,.88), rgba(10,22,35,.68) 40%, rgba(9,22,34,.25)); }}
            .weather-content {{ position: relative; z-index: 1; padding: 1.5rem 1.4rem; }}
            .weather-topline {{ color: #9ed9ff; text-transform: uppercase; letter-spacing: .14em; font-size: .7rem; font-weight: 700; }}
            .weather-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin: 1.3rem 0 1.4rem; }}
            .weather-header h2 {{ margin: .2rem 0; color: #fff; font-size: clamp(2rem, 4vw, 3.2rem); }}
            .weather-location {{ color: #dfeaf7; font-size: .82rem; letter-spacing: .1em; text-transform: uppercase; }}
            .status-pill {{ border: 1px solid rgba(142,211,255,.6); border-radius: 999px; padding: .42rem .74rem; color: #cfeeff; font-size: .68rem; font-weight: 700; }}
            .status-pill.muted {{ border-color: rgba(211,225,239,.5); color: #eaf3fd; }}
            .forecast-grid {{ display: grid; grid-template-columns: 1fr 1.2fr; gap: 1rem; }}
            .forecast-box {{ background: rgba(15,29,43,.62); border: 1px solid rgba(195,213,233,.18); border-radius: 16px; padding: 1rem; min-height: 190px; }}
            .forecast-kicker {{ color: #9cb6d0; font-size: .68rem; text-transform: uppercase; letter-spacing: .14em; }}
            .forecast-temp {{ font-size: 3rem; font-weight: 700; color: #fff; line-height: 1; margin-top: .8rem; }}
            .forecast-temp sup {{ font-size: 1rem; color: #c7d9ea; }}
            .forecast-label {{ margin-top: .7rem; color: #eef5ff; font-size: 1rem; }}
            .forecast-prob {{ margin-top: 1rem; color: #dfeaf7; font-size: .8rem; }}
            .forecast-prob strong {{ font-size: 1.2rem; color: white; }}
            .condition-row {{ display: flex; justify-content: space-between; gap: 1rem; padding: .72rem 0; border-bottom: 1px solid rgba(211,225,239,.12); color: #dfeaf7; font-size: .8rem; }}
            .condition-row:last-child {{ border-bottom: none; }}
            .weather-animated {{ position: relative; width: 100%; height: 120px; overflow: hidden; margin-top: .7rem; }}
            .sun-core {{ position: absolute; width: 44px; height: 44px; right: 26px; top: 20px; border-radius: 50%; background: #f9d45f; box-shadow: 0 0 25px rgba(249,212,95,.7); opacity: 0; }}
            .sun-ray {{ position: absolute; width: 7px; height: 22px; right: 43px; top: 3px; border-radius: 8px; background: #f9d45f; transform-origin: center 38px; opacity: 0; }}
            .ray-1 {{ transform: rotate(0deg); }} .ray-2 {{ transform: rotate(90deg); }} .ray-3 {{ transform: rotate(180deg); }} .ray-4 {{ transform: rotate(270deg); }}
            .cloud {{ position: absolute; width: 84px; height: 20px; left: 14px; top: 58px; background: rgba(188,207,224,.5); border-radius: 18px; box-shadow: 26px -12px 0 -2px rgba(200,217,234,.6), 52px -3px 0 -3px rgba(178,204,225,.54); opacity: 0; }}
            .rain-drop {{ position:absolute; width: 3px; height: 16px; border-radius: 6px; background: #67d0ff; top: 10px; box-shadow: 0 0 8px rgba(103,208,255,.6); opacity: 0; }}
            .clear .sun-core, .clear .sun-ray {{ opacity: 1; }}
            .rainy .rain-drop {{ opacity: 1; animation: rain-fall 1.2s linear infinite; }}
            .rainy .cloud {{ opacity: 0.8; }}
            @keyframes rain-fall {{ 0% {{ opacity: 0; transform: translateY(-8px); }} 20% {{ opacity: 1; }} 100% {{ opacity: 0; transform: translateY(52px); }} }}
            .reason-box {{ margin-top: 1rem; background: rgba(15,29,43,.72); border: 1px solid rgba(195,213,233,.18); border-radius: 14px; padding: 1rem; color: #dfeaf7; font-size: .88rem; line-height: 1.7; }}
            @media (max-width: 768px) {{
                .weather-overview {{ min-height: auto; border-radius: 14px; margin-top: .8rem; }}
                .weather-content {{ padding: 1.1rem .8rem; }}
                .weather-header {{ display: block; margin: 1rem 0; }}
                .weather-header h2 {{ font-size: 2.2rem; }}
                .weather-header > div:last-child {{ justify-content: flex-start !important; margin-top: .8rem; }}
                .forecast-grid {{ grid-template-columns: 1fr; gap: .75rem; }}
                .forecast-box {{ min-height: 0; padding: .85rem; }}
                .forecast-temp {{ font-size: 2.5rem; }}
                .weather-animated {{ height: 90px; }}
                .reason-box {{ font-size: .82rem; padding: .8rem; }}
            }}
        </style>
        <div class="weather-overview">
            <div class="weather-content">
                <div class="weather-topline"><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#71ef9b; margin-right:.6rem; box-shadow:0 0 10px rgba(113,239,155,.8);"></span> TOMORROW OUTLOOK</div>
                <div class="weather-header">
                    <div>
                        <div class="weather-location">Weather station</div>
                        <h2>{values.get('Location', 'Sydney')}</h2>
                    </div>
                    <div style="display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end;">
                        <span class="status-pill">AI model ready</span>
                        <span class="status-pill muted">Updated now</span>
                    </div>
                </div>
                <div class="forecast-grid">
                    <div class="forecast-box {weather_class}">
                        <div class="forecast-kicker">Tomorrow</div>
                        <div class="weather-animated">
                            <div class="sun-core"></div>
                            <div class="sun-ray ray-1"></div>
                            <div class="sun-ray ray-2"></div>
                            <div class="sun-ray ray-3"></div>
                            <div class="sun-ray ray-4"></div>
                            <div class="cloud"></div>
                            <div class="rain-drop drop-1"></div>
                            <div class="rain-drop drop-2"></div>
                            <div class="rain-drop drop-3"></div>
                            <div class="rain-drop drop-4"></div>
                        </div>
                        <div class="forecast-temp">{tomorrow_temp}<sup>°C</sup></div>
                        <div class="forecast-label">{prediction_label}</div>
                        <div class="forecast-prob">Chance: <strong>{probability:.2%}</strong></div>
                    </div>
                    <div class="forecast-box">
                        <div class="forecast-kicker">Why this forecast?</div>
                        <div class="reason-box">{summary_text}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    full_df = pd.DataFrame([
        {
            "Location": values.get("Location", "-"),
            "MinTemp": values.get("MinTemp", 0.0),
            "MaxTemp": values.get("MaxTemp", 0.0),
            "Rainfall": values.get("Rainfall", 0.0),
            "Evaporation": values.get("Evaporation", 0.0),
            "Sunshine": values.get("Sunshine", 0.0),
            "WindGustSpeed": values.get("WindGustSpeed", 0.0),
            "WindSpeed9am": values.get("WindSpeed9am", 0.0),
            "WindSpeed3pm": values.get("WindSpeed3pm", 0.0),
            "Humidity9am": values.get("Humidity9am", 0.0),
            "Humidity3pm": values.get("Humidity3pm", 0.0),
            "Pressure3pm": values.get("Pressure3pm", 0.0),
            "Cloud9am": values.get("Cloud9am", 0.0),
            "Cloud3pm": values.get("Cloud3pm", 0.0),
            "WindGustDir": values.get("WindGustDir", "N"),
            "WindDir9am": values.get("WindDir9am", "N"),
            "WindDir3pm": values.get("WindDir3pm", "N"),
            "RainToday": values.get("RainToday", "No"),
        }
    ])
    st.subheader("Observation conditions")
    st.dataframe(full_df, use_container_width=True, hide_index=True)


def sync_numeric_pair(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]


def numeric_control(label, min_value, max_value, value, step, key):
    manual_key = f"{key}_manual"
    slider_key = f"{key}_slider"
    if manual_key not in st.session_state:
        st.session_state[manual_key] = value
    if slider_key not in st.session_state:
        st.session_state[slider_key] = value

    manual_value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        format="%.1f",
        key=manual_key,
        on_change=sync_numeric_pair,
        args=(manual_key, slider_key),
    )
    slider_value = st.slider(
        f"Adjust {label}",
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=slider_key,
        on_change=sync_numeric_pair,
        args=(slider_key, manual_key),
    )
    return float(slider_value)


def main():
    initialize_database()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Australia Weather Intelligence</div>
            <h1>Rain risk forecast for tomorrow</h1>
            <p>Choose the station, enter the latest weather conditions, and the model will estimate whether the next day is likely to be wet or dry.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
            .hero { margin-bottom: 1rem; padding: 2rem 2.2rem; border-radius: 18px; border: 1px solid #2d415d; background: linear-gradient(120deg, rgba(9,18,28,.93), rgba(19,57,82,.82)), url('https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1800&q=80') center/cover; }
            .eyebrow { font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; color: #85d1ff; font-weight: 700; }
            .hero h1 { margin: .55rem 0 .45rem; color: white; font-size: clamp(2.2rem, 3vw, 3.4rem); line-height: 1.05; }
            .hero p { margin: 0; max-width: 760px; color: #dfeefb; font-size: 1rem; }
            .stApp { background: linear-gradient(180deg,#0d1726,#12253a); }
            .block-container { max-width: 1280px; }
            div[data-testid="stForm"] { background: rgba(20,34,51,.90); border: 1px solid #2d415d; border-radius: 16px; padding: 1rem 1.1rem; }
            div[data-testid="stSidebar"] { background: #111f30; }
            .tiny-tip { background: rgba(18,32,46,.9); border:1px solid #2d415d; border-radius: 14px; padding: 1rem; margin-bottom: 1rem; }
            .tiny-tip .label { color: #9ed9ff; font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
            .tiny-tip p { color: #dfeaf7; margin: .6rem 0 0; }
            .section-card { background: rgba(17,30,45,.8); border: 1px solid #2d415d; border-radius: 14px; padding: 1rem; margin: .8rem 0 1.2rem; }
            .section-card h3 { margin: .2rem 0 .4rem; color: #fff; }
            .section-card .sub { color: #bfd2e7; margin-bottom: .8rem; }
            .stButton > button { border-radius: 10px; font-weight: 700; }
            [data-testid="stDataFrame"] { overflow-x: auto; }
            @media (max-width: 768px) {
                .block-container { padding: 1rem .75rem 3rem; }
                .hero { padding: 1.35rem 1rem; border-radius: 14px; }
                .hero h1 { font-size: 2rem; }
                .hero p { font-size: .88rem; line-height: 1.55; }
                .tiny-tip { padding: .85rem; }
                .section-card { padding: .85rem; margin: .65rem 0 .9rem; }
                .section-card h3 { font-size: 1.05rem; }
                div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] { padding: .75rem; }
                div[data-testid="stSidebar"] { min-width: 240px; }
                [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .5rem; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100% !important; width: 100% !important; flex: 1 1 100% !important; }
                [data-testid="stDataFrame"] { max-width: 100%; }
                .stButton > button { min-height: 2.7rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model()
        scaler = load_scaler()
    except Exception as exc:
        st.error(f"Model failed to load: {exc}")
        st.stop()

    model_features = get_model_feature_names(model)
    location_options = get_location_options(model)
    direction_options = get_direction_options(model)

    with st.sidebar:
        st.markdown("## WeatherWise")
        st.caption("Tomorrow rain forecast")
        st.divider()
        if st.button("Forecast form", use_container_width=True):
            st.session_state["page"] = "predict"
        if st.button("Forecast history", use_container_width=True):
            st.session_state["page"] = "history"
        st.divider()
        st.markdown("### Quick guide")
        st.caption("1. Pick a weather station")
        st.caption("2. Fill in the latest observation")
        st.caption("3. Press Predict and review the forecast")
        st.caption("Tip: Rainfall > 1.0 automatically sets RainToday = Yes")

    if st.session_state.get("page") == "history":
        render_saved_history()
        return

    st.markdown(
        """
        <div class="tiny-tip">
            <div class="label">How it works</div>
            <p>All values are converted to the same format the model saw during training. The app syncs RainToday automatically, applies one-hot encoding for location and wind direction, and then runs the trained Random Forest to predict tomorrow's rain risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='section-card'><h3>01 · Station and rain status</h3><div class='sub'>Select the observation site and confirm whether rain is already occurring today.</div></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            location_value = st.selectbox("Location", location_options, index=location_options.index("Sydney") if "Sydney" in location_options else 0)
        with col2:
            rain_today = st.selectbox("RainToday", ["No", "Yes"], index=0)

        st.markdown("<div class='section-card'><h3>02 · Temperature and rainfall</h3><div class='sub'>Enter a value manually or adjust the matching slider.</div></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            min_temp = numeric_control("MinTemp (°C)", -15.0, 50.0, 15.0, 0.1, "min_temp")
            max_temp = numeric_control("MaxTemp (°C)", -10.0, 60.0, 25.0, 0.1, "max_temp")
            rainfall = numeric_control("Rainfall (mm)", 0.0, 400.0, 0.0, 0.1, "rainfall")
        with col2:
            evaporation = numeric_control("Evaporation (mm)", 0.0, 40.0, 4.0, 0.1, "evaporation")
            sunshine = numeric_control("Sunshine (hours)", 0.0, 15.0, 7.0, 0.1, "sunshine")

        st.markdown("<div class='section-card'><h3>03 · Humidity and pressure</h3><div class='sub'>These conditions help the model judge how saturated the atmosphere is before tomorrow's rain event.</div></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            humidity_9am = numeric_control("Humidity9am (%)", 0.0, 100.0, 55.0, 0.1, "humidity_9am")
            pressure_3pm = numeric_control("Pressure3pm (hPa)", 970.0, 1050.0, 1015.0, 0.1, "pressure_3pm")
        with col2:
            humidity_3pm = numeric_control("Humidity3pm (%)", 0.0, 100.0, 50.0, 0.1, "humidity_3pm")

        st.markdown("<div class='section-card'><h3>04 · Cloud cover</h3><div class='sub'>Cloud amount is one of the clearest indicators of how much solar energy is being blocked.</div></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            cloud_9am = numeric_control("Cloud9am (oktas)", 0.0, 9.0, 3.0, 0.1, "cloud_9am")
        with col2:
            cloud_3pm = numeric_control("Cloud3pm (oktas)", 0.0, 9.0, 4.0, 0.1, "cloud_3pm")

        st.markdown("<div class='section-card'><h3>05 · Wind conditions</h3><div class='sub'>Standard compass directions are used so the categorical encoding matches the trained model.</div></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            wind_gust_speed = numeric_control("WindGustSpeed (km/h)", 0.0, 150.0, 35.0, 0.1, "wind_gust_speed")
            wind_speed_9am = numeric_control("WindSpeed9am (km/h)", 0.0, 100.0, 15.0, 0.1, "wind_speed_9am")
            wind_gust_dir = st.selectbox("WindGustDir", direction_options, index=direction_options.index("N") if "N" in direction_options else 0)
            wind_dir_9am = st.selectbox("WindDir9am", direction_options, index=direction_options.index("N") if "N" in direction_options else 0)
        with col2:
            wind_speed_3pm = numeric_control("WindSpeed3pm (km/h)", 0.0, 100.0, 18.0, 0.1, "wind_speed_3pm")
            wind_dir_3pm = st.selectbox("WindDir3pm", direction_options, index=direction_options.index("N") if "N" in direction_options else 0)

        values = {
            "Location": location_value,
            "RainToday": "Yes" if rainfall > 1.0 else rain_today,
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "Evaporation": evaporation,
            "Sunshine": sunshine,
            "WindGustSpeed": wind_gust_speed,
            "WindSpeed9am": wind_speed_9am,
            "WindSpeed3pm": wind_speed_3pm,
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure3pm": pressure_3pm,
            "Cloud9am": cloud_9am,
            "Cloud3pm": cloud_3pm,
            "WindGustDir": wind_gust_dir,
            "WindDir9am": wind_dir_9am,
            "WindDir3pm": wind_dir_3pm,
        }

        submitted = st.button("Predict tomorrow's forecast", use_container_width=True, type="primary")

    if submitted:
        validation_errors = validate_input_ranges(values)
        if validation_errors:
            st.error("Please correct the following range errors before running the forecast:")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            try:
                normalized = normalize_observation(values)
                label, pred_value, probability, _ = predict_rain_tomorrow(normalized, model, scaler=scaler)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result = {"label": label, "value": pred_value, "probability": probability if probability is not None else 0.0, "timestamp": timestamp}
                save_prediction(normalized, result)
                render_weather_overview(normalized, label, probability if probability is not None else 0.0)
                col_a, col_b = st.columns(2)
                with col_a:
                    render_result_card("Tomorrow forecast", label, probability)
                with col_b:
                    st.markdown(
                        """
                        <div style="background: rgba(20,34,51,.9); border:1px solid #35506d; border-radius:14px; padding:1rem; min-height:190px;">
                            <div style="font-size:.7rem; letter-spacing:.15em; text-transform:uppercase; color:#9abbd6; font-weight:700;">Key drivers behind this forecast</div>
                            <div style="margin-top:1rem; color:#dfeaf7; line-height:1.8; font-size:.9rem;">
                                <div>{reason}</div>
                            </div>
                        </div>
                        """.format(
                            reason=build_reason_summary(normalized, label, probability if probability is not None else 0.0),
                        ),
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"Prediction could not be produced: {exc}")


if __name__ == "__main__":
    main()
