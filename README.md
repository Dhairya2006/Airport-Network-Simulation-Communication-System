# AirportNet-AI

**Python AI model + evaluation + Streamlit dashboard**, designed to accompany a Cisco Packet Tracer airport communication topology containing Check-In, Security, Operations, Baggage, servers, wireless and IoT devices.

> **Data integrity:** The included observations are **synthetic**, not measurements from an airport or a Cisco Packet Tracer file. The classifier predicts a *traffic-state scenario* (Normal / Congested / Critical); it is **not a validated intrusion detector**. Smoke and motion notifications are rules, not ML predictions. This educational dashboard does not actually configure Cisco switches or operate real sirens.

## Files

```text
AirportNet_AI_Complete/
  generate_data.py        create reproducible synthetic 5-minute observations
  train.py                train, session-split evaluation, plots and saved pipeline
  inference.py            reusable CSV validation and inference
  predict_csv.py          command-line batch classification
  dashboard.py            interactive Streamlit dashboard
  settings.py             paths and schema definitions
  requirements.txt        Python dependencies
  data/
    synthetic_airport_traffic.csv          generated example training data
    packet_tracer_blank_template.csv      measurement input template
    packet_tracer_measurements_example.csv clearly illustrative scenarios
  models/airportnet_model.joblib          trained scikit-learn pipeline
  reports/
    metrics.json                         evaluation metrics
    classification_report.txt            class precision/recall/F1
    confusion_matrix.png                 held-out confusion matrix
    feature_importance.png               permutation feature importance
    feature_importance.csv               sortable feature importances
    test_predictions.csv                 held-out predictions
  tests/test_project.py   unit tests
```

## Setup (Windows or macOS/Linux, Python 3.10+)

In a terminal inside the extracted `AirportNet_AI_Complete` directory:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python train.py
python -m streamlit run dashboard.py
```

On some Windows computers, run `py` instead of `python`.
After launching Streamlit, open the local URL printed to the terminal (often http://localhost:8501).

**Pre-generated dataset, model, and reports are bundled** so the dashboard can be demonstrated after installing requirements. The bundled model was saved using scikit-learn 1.8.0; re-run `python train.py` after installing dependencies to regenerate a compatible model for your installed scikit-learn version.

## Command-line predictions

```bash
python predict_csv.py data/packet_tracer_measurements_example.csv --output data/predictions.csv
python -m unittest discover -s tests -v
```

## Model details

- Dataset: 600 synthetic monitoring sessions x 12 observations = 7,200 samples.
- Session labels drawn from a mixture: Normal ~57%, Congested ~29%, Critical ~14%.
- Features: department, packets observed, link utilization, packet loss, average delay, jitter, connected devices.
- Classes are underlying **synthetic traffic scenario categories**, not operational truth from real devices.
- For each traffic category, measurements follow *overlapping* noisy distributions, with department and session-specific shifts; no direct threshold conversion is used to create targets.
- 80/20 **group split by session_id** prevents one session's temporally correlated records appearing in both training and test partitions.
- Baseline: most-frequent-class classifier; main model: RandomForestClassifier with one-hot encoding of department.
- Outputs: accuracy, balanced accuracy, macro F1, per-class precision/recall, confusion matrix, and permutation importance.
- **Limitation:** An independent real-world test, with actual measured metrics and verified labels, is necessary before any operational performance claim.

## Packet Tracer CSV workflow (manual, no live bridge)

1. Keep your existing airport topology: router, core switch, Check-In/Security/Operations/Baggage switches, servers, APs and IoT gateway.
2. Generate and trace traffic using PC `ping` and HTTP requests; inspect **Simulation Mode** packet events.
3. Define one consistent observation window (suggested **5 minutes**) for each department. Record packets and hosts; if Packet Tracer doesn't expose utilization/latency/jitter/loss as measured values, measure via another validated method or mark entries explicitly as *estimated for demonstration*.
4. Open `data/packet_tracer_blank_template.csv` and fill the required fields: `department,packet_count,utilization_pct,packet_loss_pct,delay_ms,jitter_ms,connected_devices,smoke_alert,motion_alert`. Supported department names are in `settings.py`. Use 0/1 for optional event flags.
5. Upload the populated file in the dashboard's **CSV batch analysis** tab, or use `predict_csv.py`.
6. Download the resulting CSV with `predicted_status`, `prob_normal`, `prob_congested`, `prob_critical`, and rule-based `alert`.

The example measurement CSV uses **invented values solely for the demo**. You cannot infer actual link utilization, delay, jitter or loss from Packet Tracer screenshots alone.

## Suggested classroom demonstration

1. Show connectivity in Packet Tracer between departments and to the website server.
2. Open the dashboard; demonstrate Normal, Congested and Critical responses to modified metrics.
3. Enable the smoke checkbox; show the independent rule-based notification.
4. Upload the included **illustrative** example CSV and export predictions.
5. Open Model evaluation and explain the grouped test split, confusion matrix and the majority baseline.

## Practical considerations

- Do not claim IoT alerts are connected to physical detectors or that the classifier can actuate alarms.
- A 100% result on synthetic, threshold-derived data would be misleading; this dataset deliberately overlaps class distributions to create a less trivial learning exercise.
- Device metrics should be obtained from actual measurements before claiming use of real Packet Tracer telemetry. Estimated/demo values should be disclosed in the report.
- Never load untrusted `.joblib` model files; `joblib.load` can run arbitrary code.
