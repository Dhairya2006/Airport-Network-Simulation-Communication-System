"""Shared paths and schemas for AirportNet-AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DEPARTMENTS = ["Check-In", "Security", "Operations", "Baggage", "Servers", "Wireless", "IoT"]
LABELS = ["Normal", "Congested", "Critical"]
NUMERIC_FEATURES = [
    "packet_count", "utilization_pct", "packet_loss_pct",
    "delay_ms", "jitter_ms", "connected_devices",
]
FEATURES = ["department", *NUMERIC_FEATURES]
OPTIONAL_EVENTS = ["smoke_alert", "motion_alert"]
MODEL_PATH = MODELS / "airportnet_model.joblib"
DATA_PATH = DATA / "synthetic_airport_traffic.csv"

for folder in (DATA, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)
