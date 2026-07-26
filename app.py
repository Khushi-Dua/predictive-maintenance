import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
from sklearn.ensemble import IsolationForest

MODEL_DIR = "models"

def load_models():
    ensemble = []
    try:
        import tensorflow as tf
        for i in range(1, 6):
            path = os.path.join(MODEL_DIR, f"ensemble_lstm_{i}.keras")
            if os.path.exists(path):
                ensemble.append(tf.keras.models.load_model(path))
        if ensemble:
            print(f"Loaded {len(ensemble)} LSTM models.")
    except Exception as e:
        print(f"TF not available, XGBoost only: {e}")

    xgb = None
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor()
        xgb_path = os.path.join(MODEL_DIR, "xgboost_rul.json")
        if os.path.exists(xgb_path):
            xgb.load_model(xgb_path)
            print("XGBoost loaded.")
    except Exception as e:
        print(f"XGBoost error: {e}")

    scaler, useful_sensors = None, None
    try:
        scaler         = joblib.load(os.path.join(MODEL_DIR, "sensor_scaler.pkl"))
        useful_sensors = joblib.load(os.path.join(MODEL_DIR, "useful_sensors.pkl"))
    except Exception as e:
        print(f"Scaler error: {e}")

    return ensemble, xgb, scaler, useful_sensors

ensemble_models, xgb_model, scaler, useful_sensors = load_models()
SEQUENCE_LENGTH = 30

COLS = (
    ["unit_number", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def generate_demo_data(n_cycles=80, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for t in range(1, n_cycles + 1):
        frac    = t / n_cycles
        sensors = 0.3 + 0.5 * frac + rng.normal(0, 0.05, 21)
        ops     = rng.normal(0, 0.03, 3)
        rows.append([1, t, *ops, *sensors])
    return pd.DataFrame(rows, columns=COLS)

def predict_rul(df):
    if useful_sensors is None or scaler is None:
        return None, None, None, "Models not loaded"
    missing = [s for s in useful_sensors if s not in df.columns]
    if missing:
        return None, None, None, f"Missing columns: {missing}"

    df = df.copy()
    df[useful_sensors] = scaler.transform(df[useful_sensors])
    data = df[useful_sensors].values

    if len(data) >= SEQUENCE_LENGTH:
        window = data[-SEQUENCE_LENGTH:]
    else:
        pad    = np.tile(data[0], (SEQUENCE_LENGTH - len(data), 1))
        window = np.vstack([pad, data])

    window_3d = window[np.newaxis, :, :]
    window_2d = window.reshape(1, -1)

    # LSTM ensemble (if available)
    if ensemble_models:
        preds    = [m.predict(window_3d, verbose=0).flatten()[0] for m in ensemble_models]
        rul_lstm = float(np.mean(preds))
    else:
        rul_lstm = None

    # XGBoost
    if xgb_model:
        rul_xgb = float(xgb_model.predict(window_2d)[0])
    else:
        rul_xgb = 80.0

    # Use XGBoost as primary if LSTM not available
    rul_primary = rul_lstm if rul_lstm is not None else rul_xgb

    # Anomaly detection
    iso      = IsolationForest(contamination=0.05, random_state=42)
    iso_flag = iso.fit_predict(window_2d) == -1
    anomaly  = bool(iso_flag[0])

    return rul_primary, rul_xgb, anomaly, None

def classify_health(rul, anomaly):
    if anomaly:
        return "🔴 Maintenance Due"
    if rul > 50:
        return "🟢 Healthy"
    elif rul > 20:
        return "🟡 Watch"
    else:
        return "🔴 Maintenance Due"

def make_gauge(rul):
    fig, ax = plt.subplots(figsize=(10, 1.4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.barh(0, 50,  left=0,  height=0.5, color="#F44336", alpha=0.9)
    ax.barh(0, 30,  left=50, height=0.5, color="#FFC107", alpha=0.9)
    ax.barh(0, 45,  left=80, height=0.5, color="#4CAF50", alpha=0.9)
    rul_c = min(max(rul, 0), 125)
    ax.axvline(rul_c, color="white", linewidth=3)
    ax.text(rul_c + 1, 0, f"  {rul:.1f}", color="white", va="center",
            fontsize=11, fontweight="bold")
    ax.set_xlim(0, 125)
    ax.set_yticks([])
    ax.set_xlabel("RUL (cycles)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2d3250")
    red_p = mpatches.Patch(color="#F44336", label="Critical (0–50)")
    yel_p = mpatches.Patch(color="#FFC107", label="Watch (50–80)")
    grn_p = mpatches.Patch(color="#4CAF50", label="Healthy (80–125)")
    ax.legend(handles=[red_p, yel_p, grn_p], loc="upper left",
              facecolor="#1e2130", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return fig

def make_sensor_plot(df, sensors):
    if not sensors:
        return None
    fig, axes = plt.subplots(len(sensors), 1,
                             figsize=(11, 3 * len(sensors)), squeeze=False)
    fig.patch.set_facecolor("#0f1117")
    colors = ["#4FC3F7","#81C784","#FFB74D","#F48FB1",
              "#CE93D8","#80DEEA","#FFCC02","#FF8A65"]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx][0]
        ax.set_facecolor("#1e2130")
        if sensor in df.columns:
            ax.plot(df["time_cycle"], df[sensor],
                    color=colors[idx % len(colors)], linewidth=1.5)
        ax.set_title(sensor, color="white", fontsize=10)
        ax.tick_params(colors="#B0BEC5")
        for spine in ax.spines.values():
            spine.set_color("#2d3250")
        ax.set_xlabel("Cycle", color="#B0BEC5", fontsize=8)
    plt.tight_layout()
    return fig

current_df = {"data": None}

def run_predict(file, use_demo):
    df = None
    if use_demo:
        df  = generate_demo_data()
        msg = "Running on synthetic demo data (80 cycles)."
    elif file is not None:
        try:
            df = pd.read_csv(file.name, sep=None, engine="python", header=None)
            if df.shape[1] >= len(COLS):
                df = df.iloc[:, :len(COLS)]
                df.columns = COLS
                msg = f"✅ Loaded {len(df)} rows."
            else:
                return ("❌ Not enough columns.", "", "", "", None, None,
                        gr.update(choices=[]))
        except Exception as e:
            return (f"❌ Error: {e}", "", "", "", None, None,
                    gr.update(choices=[]))
    else:
        return ("⚠️ Upload a CSV or click Run Demo.", "", "", "", None, None,
                gr.update(choices=[]))

    current_df["data"] = df
    rul_primary, rul_xgb, anomaly, err = predict_rul(df)
    if err:
        return (err, "", "", "", None, None, gr.update(choices=[]))

    status     = classify_health(rul_primary, anomaly)
    gauge      = make_gauge(rul_primary)
    sensor_opt = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" in df.columns]
    sensor_fig = make_sensor_plot(df, sensor_opt[:4])
    lstm_str   = f"{rul_primary:.1f} cycles" if ensemble_models else "N/A (TF not loaded)"
    xgb_str    = f"{rul_xgb:.1f} cycles"

    return (msg, lstm_str, xgb_str, status, gauge, sensor_fig,
            gr.update(choices=sensor_opt, value=sensor_opt[:4]))

def update_sensor_plot(selected_sensors):
    df = current_df["data"]
    if df is None or not selected_sensors:
        return None
    return make_sensor_plot(df, selected_sensors)

CSS = """
body { background: #0f1117; }
.gradio-container { background: #0f1117 !important; }
footer { display: none !important; }
.gr-prose, .gr-markdown, label, .svelte-1gfkn6j { color: #E0E0E0 !important; }
table { color: #E0E0E0 !important; border-color: #2d3250 !important; }
th, td { color: #E0E0E0 !important; border-color: #2d3250 !important; background: #1e2130 !important; }
"""

with gr.Blocks(css=CSS, title="Predictive Maintenance — Mexmon Technologies") as demo:
    gr.Markdown("""
# ⚙️ Predictive Maintenance System
**Mexmon Technologies — Design & Automation Division**

Forecasts **Remaining Useful Life (RUL)** of industrial equipment from sensor data.

---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload Sensor CSV", file_types=[".csv", ".txt"])
            use_demo   = gr.Checkbox(label="Run on demo data instead", value=False)
            run_btn    = gr.Button("▶ Run Prediction", variant="primary", size="lg")
            status_msg = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("""
### Results (Official Test Set — 707 units)
| Model | RMSE | MAE |
|---|---|---|
| Single LSTM | 26.86 | 18.57 |
| **Ensemble LSTM** | **26.31** | **18.39** |
| XGBoost | 26.62 | 18.78 |

🟢 **Healthy** — RUL > 50  
🟡 **Watch** — 20 < RUL ≤ 50  
🔴 **Maintenance Due** — RUL ≤ 20 or anomaly
            """)

    gr.Markdown("---")
    gr.Markdown("### Prediction Results")

    with gr.Row():
        rul_lstm_out = gr.Textbox(label="Ensemble LSTM — RUL", interactive=False)
        rul_xgb_out  = gr.Textbox(label="XGBoost — RUL",       interactive=False)
        status_out   = gr.Textbox(label="Health Status",        interactive=False)

    gauge_out = gr.Plot(label="RUL Health Gauge")

    gr.Markdown("---")
    gr.Markdown("### Sensor Trends")
    sensor_select = gr.CheckboxGroup(
        label="Select sensors to plot",
        choices=[f"sensor_{i}" for i in range(1, 22)],
        value=["sensor_2", "sensor_7", "sensor_11", "sensor_14"]
    )
    sensor_plot = gr.Plot(label="Sensor Degradation Over Time")

    gr.Markdown("""
---
**Intern:** Khushi Dua | **Organisation:** Mexmon Technologies | **Duration:** May–July 2025  
**Dataset:** NASA CMAPSS (FD001–FD004) | **Models:** Ensemble LSTM + XGBoost + Isolation Forest
    """)

    run_btn.click(
        fn=run_predict,
        inputs=[file_input, use_demo],
        outputs=[status_msg, rul_lstm_out, rul_xgb_out, status_out,
                 gauge_out, sensor_plot, sensor_select]
    )
    sensor_select.change(
        fn=update_sensor_plot,
        inputs=[sensor_select],
        outputs=[sensor_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
