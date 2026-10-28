"""Input validation and safe CSV inference, shared by CLI and dashboard."""
import numpy as np
import pandas as pd
from settings import DEPARTMENTS, FEATURES, LABELS, NUMERIC_FEATURES


def validate_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("CSV has no data rows")
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    result = frame.copy()
    result["department"] = result["department"].astype(str).str.strip()
    invalid_dept = sorted(set(result["department"]) - set(DEPARTMENTS))
    if invalid_dept:
        raise ValueError("Unknown department(s): " + ", ".join(invalid_dept) + ". Use " + ", ".join(DEPARTMENTS))
    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if not np.isfinite(result[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {col} contains missing or non-numeric values")
        if (result[col] < 0).any():
            raise ValueError(f"Column {col} cannot be negative")
    for col in ("utilization_pct", "packet_loss_pct"):
        if (result[col] > 100).any():
            raise ValueError(f"Column {col} must be between 0 and 100")
    for col in ("packet_count", "connected_devices"):
        if ((result[col] % 1) != 0).any():
            raise ValueError(f"Column {col} must contain whole numbers")
    for col in ("smoke_alert", "motion_alert"):
        if col not in result.columns:
            result[col] = 0
        mapping = {"1": 1, "true": 1, "yes": 1, "on": 1, "0": 0, "false": 0, "no": 0, "off": 0}
        value = result[col].astype(str).str.strip().str.lower().map(mapping)
        if value.isna().any():
            raise ValueError(f"Column {col} must contain 0/1 or True/False")
        result[col] = value.astype(int)
    return result


def classify(model, frame):
    data = validate_frame(frame)
    predictions = model.predict(data[FEATURES])
    probs = model.predict_proba(data[FEATURES])
    result = data.copy()
    result["predicted_status"] = predictions
    for label in LABELS:
        idx = list(model.classes_).index(label)
        result["prob_" + label.lower()] = np.round(probs[:, idx], 4)
    result["alert"] = np.select(
        [result["smoke_alert"].eq(1), result["motion_alert"].eq(1)],
        ["SMOKE: notify security/operations", "MOTION: security review"],
        default="None",
    )
    return result
