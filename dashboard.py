"""Start with: python -m streamlit run dashboard.py"""
import json
import joblib
import pandas as pd
import streamlit as st
from settings import (DATA_PATH, DEPARTMENTS, FEATURES, LABELS, MODEL_PATH,
                      REPORTS, ROOT)
from inference import classify

st.set_page_config(page_title="AirportNet-AI", page_icon="✈️", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
[data-testid="stMetric"] { border: 1px solid rgba(110,130,150,.25); padding: 12px; border-radius: 9px; }
</style>
""", unsafe_allow_html=True)
st.title("✈️ AirportNet-AI")
st.subheader("Intelligent Airport Network Monitoring")
st.info("Educational prototype. Model trained on **synthetic** traffic; results are **not live Cisco Packet Tracer telemetry** and must not be used for operational or safety decisions.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python generate_data.py` then `python train.py`.")
    st.stop()

@st.cache_resource
def get_model():
    # Only load the trusted model trained by this project, not user-uploaded model files.
    return joblib.load(MODEL_PATH)

model = get_model()
tab_monitor, tab_batch, tab_results, tab_guide = st.tabs([
    "📡 Traffic prediction", "📂 CSV batch analysis", "📈 Model evaluation", "📘 Integration guide"
])

with tab_monitor:
    st.markdown("### Enter one five-minute network observation")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        department = st.selectbox("Department", DEPARTMENTS, index=1)
        packet_count = st.number_input("Packets observed in five minutes", min_value=0, value=12000, step=500)
        connected_devices = st.number_input("Connected devices", min_value=0, value=18)
        utilization_pct = st.slider("Link utilization (%)", 0.0, 100.0, 70.0, step=0.5)
        packet_loss_pct = st.slider("Packet loss (%)", 0.0, 100.0, 2.5, step=0.5)
        delay_ms = st.number_input("Average delay (ms)", min_value=0.0, value=125.0, step=5.0)
        jitter_ms = st.number_input("Jitter (ms)", min_value=0.0, value=25.0, step=5.0)
        smoke = st.checkbox("Smoke detector event")
        motion = st.checkbox("Motion detector event")
    record = pd.DataFrame([{
        "department": department, "packet_count": packet_count,
        "connected_devices": connected_devices, "utilization_pct": utilization_pct,
        "packet_loss_pct": packet_loss_pct, "delay_ms": delay_ms,
        "jitter_ms": jitter_ms, "smoke_alert": int(smoke), "motion_alert": int(motion)
    }])
    outcome = classify(model, record).iloc[0]
    with right:
        st.markdown("### AI classification")
        cols = st.columns(3)
        cols[0].metric("Network status", outcome["predicted_status"])
        cols[1].metric("Packet loss", f"{packet_loss_pct:.1f}%")
        cols[2].metric("Utilization", f"{utilization_pct:.1f}%")
        probability = pd.DataFrame({
            "Status": LABELS,
            "Model score": [outcome["prob_" + label.lower()] for label in LABELS]
        }).set_index("Status")
        st.bar_chart(probability, y="Model score", color="#2870b2")
        st.caption("Random Forest class vote fractions; not calibrated real-world probabilities.")
        if smoke:
            st.error("🚨 Smoke alert: contact airport security/operations, following local emergency procedures. This dashboard does not activate a physical siren.")
        if motion:
            st.warning("⚠️ Motion alert: request security review. This is a rule-based IoT event.")
        if outcome["predicted_status"] == "Critical":
            st.error("Network classified Critical: inspect connectivity and security event logs.")
        elif outcome["predicted_status"] == "Congested":
            st.warning("Network classified Congested: inspect queueing and bandwidth usage.")
        else:
            st.success("Network classified Normal in the synthetic-data model.")

with tab_batch:
    st.markdown("### Classify a CSV exported or transcribed from your simulation")
    st.write("Use `data/packet_tracer_measurements_example.csv` as a filled illustrative example, or `data/packet_tracer_blank_template.csv` to enter your own measurements.")
    st.caption("The upload is processed locally in this Streamlit application. Packet Tracer does not automatically send measurements here.")
    example_path = ROOT / "data" / "packet_tracer_measurements_example.csv"
    st.download_button("Download example CSV", example_path.read_bytes(), file_name="packet_tracer_measurements_example.csv", mime="text/csv")
    upload = st.file_uploader("Upload measurements (.csv)", type="csv", key="measurements")
    if upload is not None:
        try:
            source = pd.read_csv(upload)
            predicted = classify(model, source)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"CSV validation failed: {exc}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", len(predicted))
            c2.metric("Normal", int((predicted.predicted_status == "Normal").sum()))
            c3.metric("Congested", int((predicted.predicted_status == "Congested").sum()))
            c4.metric("Critical", int((predicted.predicted_status == "Critical").sum()))
            st.dataframe(predicted, hide_index=True, use_container_width=True)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### Predicted traffic class")
                totals = predicted.predicted_status.value_counts().reindex(LABELS, fill_value=0)
                st.bar_chart(totals.rename_axis("Status").to_frame("Records"))
            with col_r:
                st.markdown("#### By airport department")
                grouped = pd.crosstab(predicted.department, predicted.predicted_status).reindex(columns=LABELS, fill_value=0)
                st.bar_chart(grouped)
            alerts = predicted[predicted.alert != "None"]
            if len(alerts):
                st.error(f"{len(alerts)} event(s) need human attention (separate rule-based IoT alerts).")
                st.dataframe(alerts[["department", "alert"]], hide_index=True, use_container_width=True)
            st.download_button("Download predictions CSV", predicted.to_csv(index=False).encode("utf-8"),
                file_name="airportnet_predictions.csv", mime="text/csv")

with tab_results:
    st.markdown("### Synthetic held-out session evaluation")
    metrics_path = REPORTS / "metrics.json"
    if not metrics_path.exists():
        st.warning("Run `python train.py` to generate the evaluation reports.")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{metrics['test_accuracy']:.1%}")
        k2.metric("Macro F1", f"{metrics['test_macro_f1']:.1%}")
        k3.metric("Balanced accuracy", f"{metrics['test_balanced_accuracy']:.1%}")
        k4.metric("Majority baseline", f"{metrics['majority_baseline_accuracy']:.1%}")
        st.caption(f"{metrics['training_rows']:,} training rows / {metrics['test_rows']:,} held-out rows. No shared session IDs across split.")
        p1, p2 = st.columns(2)
        with p1:
            st.image(str(REPORTS / "confusion_matrix.png"), caption="Held-out confusion matrix")
        with p2:
            st.image(str(REPORTS / "feature_importance.png"), caption="Permutation feature importance")
        report = pd.DataFrame(metrics["report"]).T
        st.dataframe(report.round(3), use_container_width=True)
        st.warning("Synthetic test performance is not independently validated operational accuracy. No live airport network data were used.")

with tab_guide:
    st.markdown("""
### Architecture
**Packet Tracer departments → five-minute measurement table (CSV) → Python model → Streamlit dashboard.**

### Required CSV columns
- `department`: Check-In, Security, Operations, Baggage, Servers, Wireless, IoT
- `packet_count`: observed or estimated packets during a stated five-minute window
- `utilization_pct`: link utilization percent (0–100)
- `packet_loss_pct`: packet loss percent (0–100)
- `delay_ms`: mean delay in milliseconds
- `jitter_ms`: estimated delay variation in milliseconds
- `connected_devices`: number of connected hosts

Optional: `smoke_alert`, `motion_alert` (0 or 1). IoT alerts follow deterministic rules; they are NOT model features.

### Important limitations
- Packet Tracer **Simulation Mode** helps inspect packets, but does not automatically generate all required numeric metrics.
- When values cannot be measured directly, explicitly label them as **illustrative/estimated**, never captured telemetry.
- This demonstration does not configure Cisco devices or control airport alarms automatically.
- The classifier was trained from simulated latent traffic scenarios; it does not prove security intrusion detection.
""")
