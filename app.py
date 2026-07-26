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
    except Exception as e:
        print(f"TF not available: {e}")

    xgb = None
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor()
        xgb_path = os.path.join(MODEL_DIR, "xgboost_rul.json")
        if os.path.exists(xgb_path):
            xgb.load_model(xgb_path)
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
    ["unit_number","time_cycle","op_setting_1","op_setting_2","op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def generate_demo_data(n_cycles=80, seed=42):
    rng  = np.random.RandomState(seed)
    rows = []
    for t in range(1, n_cycles + 1):
        frac    = t / n_cycles
        sensors = 0.3 + 0.5 * frac + rng.normal(0, 0.05, 21)
        ops     = rng.normal(0, 0.03, 3)
        rows.append([1, t, *ops, *sensors])
    return pd.DataFrame(rows, columns=COLS)

def predict_rul(df):
    if useful_sensors is None or scaler is None:
        return None, None, None, "Model files not found in /models folder"
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

    rul_lstm = None
    if ensemble_models:
        preds    = [m.predict(window_3d, verbose=0).flatten()[0] for m in ensemble_models]
        rul_lstm = float(np.mean(preds))

    rul_xgb = float(xgb_model.predict(window_2d)[0]) if xgb_model else 80.0
    rul_primary = rul_lstm if rul_lstm is not None else rul_xgb

    iso      = IsolationForest(contamination=0.05, random_state=42)
    anomaly  = bool(iso.fit_predict(window_2d)[0] == -1)

    return rul_primary, rul_xgb, anomaly, None

def classify_health(rul, anomaly):
    if anomaly or rul <= 20:
        return "🔴 Maintenance Due"
    elif rul <= 50:
        return "🟡 Watch"
    else:
        return "🟢 Healthy"

def make_gauge(rul):
    fig, ax = plt.subplots(figsize=(9, 1.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.barh(0, 50,  left=0,  height=0.5, color="#ef5350", alpha=0.9, label="Critical (0–50)")
    ax.barh(0, 30,  left=50, height=0.5, color="#FFA726", alpha=0.9, label="Watch (50–80)")
    ax.barh(0, 45,  left=80, height=0.5, color="#66BB6A", alpha=0.9, label="Healthy (80–125)")
    rul_c = min(max(rul, 0), 125)
    ax.axvline(rul_c, color="#1a1a2e", linewidth=4)
    ax.text(rul_c + 1.5, 0.01, f"{rul:.1f}", color="#1a1a2e",
            va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 125)
    ax.set_yticks([])
    ax.set_xlabel("Remaining Useful Life (cycles)", fontsize=10, color="#333")
    ax.tick_params(colors="#333")
    for spine in ax.spines.values():
        spine.set_color("#ccc")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    return fig

def make_sensor_plot(df, sensors):
    if not sensors:
        return None
    colors = ["#1565C0","#2E7D32","#E65100","#6A1B9A",
              "#00838F","#558B2F","#AD1457","#4527A0"]
    fig, axes = plt.subplots(len(sensors), 1,
                             figsize=(11, 2.8 * len(sensors)), squeeze=False)
    fig.patch.set_facecolor("white")
    for idx, sensor in enumerate(sensors):
        ax = axes[idx][0]
        ax.set_facecolor("#f8f9fa")
        if sensor in df.columns:
            ax.plot(df["time_cycle"], df[sensor],
                    color=colors[idx % len(colors)], linewidth=1.8)
        ax.set_title(sensor, fontsize=11, fontweight="bold", color="#1a1a2e")
        ax.tick_params(colors="#555")
        ax.set_xlabel("Cycle", fontsize=9, color="#555")
        for spine in ax.spines.values():
            spine.set_color("#ddd")
    plt.tight_layout()
    return fig

current_df = {"data": None}

def run_predict(file, use_demo):
    df = None
    if use_demo:
        df  = generate_demo_data()
        msg = "✅ Running on synthetic demo data (80 cycles)."
    elif file is not None:
        try:
            df = pd.read_csv(file.name, sep=None, engine="python", header=None)
            if df.shape[1] >= len(COLS):
                df = df.iloc[:, :len(COLS)]
                df.columns = COLS
                msg = f"✅ Loaded {len(df)} rows successfully."
            else:
                return ("❌ Not enough columns in CSV.", "", "", "", None, None,
                        gr.update(choices=[]))
        except Exception as e:
            return (f"❌ Error reading file: {e}", "", "", "", None, None,
                    gr.update(choices=[]))
    else:
        return ("⚠️ Upload a CSV or tick 'Run on demo data'.", "", "", "", None, None,
                gr.update(choices=[]))

    current_df["data"] = df
    rul_primary, rul_xgb, anomaly, err = predict_rul(df)
    if err:
        return (f"❌ {err}", "", "", "", None, None, gr.update(choices=[]))

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

with gr.Blocks(theme=gr.themes.Default(), title="Predictive Maintenance — Mexmon Technologies") as demo:

    gr.Markdown("# ⚙️ Predictive Maintenance System")
    gr.Markdown("**Mexmon Technologies — Design & Automation Division**")
    gr.Markdown("Forecasts **Remaining Useful Life (RUL)** of industrial equipment from sensor data.")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload Sensor CSV", file_types=[".csv", ".txt"])
            use_demo   = gr.Checkbox(label="Run on demo data instead (no upload needed)", value=False)
            run_btn    = gr.Button("▶  Run Prediction", variant="primary", size="lg")
            status_msg = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Model Results (Official Test Set — 707 units)")
            gr.Dataframe(
                value=pd.DataFrame({
                    "Model":  ["Single LSTM", "Ensemble LSTM ★", "XGBoost"],
                    "RMSE":   [26.86, 26.31, 26.62],
                    "MAE":    [18.57, 18.39, 18.78]
                }),
                interactive=False
            )
            gr.Markdown("""
🟢 **Healthy** — RUL > 50, no anomaly  
🟡 **Watch** — 20 < RUL ≤ 50  
🔴 **Maintenance Due** — RUL ≤ 20 or anomaly flagged
            """)

    gr.Markdown("---")
    gr.Markdown("### Prediction Results")

    with gr.Row():
        rul_lstm_out = gr.Textbox(label="Ensemble LSTM — Predicted RUL", interactive=False)
        rul_xgb_out  = gr.Textbox(label="XGBoost — Predicted RUL",       interactive=False)
        status_out   = gr.Textbox(label="Health Status",                  interactive=False)

    gauge_out = gr.Plot(label="RUL Health Gauge")

    gr.Markdown("---")
    gr.Markdown("### Sensor Trends")
    sensor_select = gr.CheckboxGroup(
        label="Select sensors to visualise",
        choices=[f"sensor_{i}" for i in range(1, 22)],
        value=["sensor_2", "sensor_7", "sensor_11", "sensor_14"]
    )
    sensor_plot = gr.Plot(label="Sensor Degradation Over Time")

    gr.Markdown("---")
    gr.Markdown("""
**Intern:** Khushi Dua &nbsp;|&nbsp; **Organisation:** Mexmon Technologies &nbsp;|&nbsp; **Duration:** May–July 2025  
**Dataset:** NASA CMAPSS (FD001–FD004) — 707 test units &nbsp;|&nbsp; **Models:** Ensemble LSTM + XGBoost + Isolation Forest
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
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
