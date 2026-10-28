"""Batch CSV prediction, independent of the Streamlit interface."""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
from inference import classify
from settings import MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Classify a measurement CSV")
    parser.add_argument("input", type=Path, help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/airportnet_predictions.csv"))
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run python generate_data.py and python train.py")
    try:
        data = pd.read_csv(args.input)
        model = joblib.load(MODEL_PATH)
        predictions = classify(model, data)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        sys.exit(f"Prediction failed: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print("Saved", len(predictions), "predictions to", args.output)
    print(predictions.predicted_status.value_counts().to_string())
    print("Rule-based event alerts:", int((predictions.alert != "None").sum()))


if __name__ == "__main__":
    main()
