"""Train and evaluate a supervised Random Forest on synthetic sessions."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from settings import DATA_PATH, FEATURES, LABELS, MODEL_PATH, MODELS, REPORTS


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Dataset absent. Run: python generate_data.py")
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + ["session_id", "status"]
    if data[required].isna().any().any():
        raise ValueError("Dataset has missing training values")
    # No session's observations appear in both training and held-out test data.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(data, data["status"], groups=data["session_id"]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    X_train, y_train = train[FEATURES], train["status"]
    X_test, y_test = test[FEATURES], test["status"]
    encoder = ColumnTransformer(
        [("department", OneHotEncoder(handle_unknown="ignore"), ["department"])],
        remainder="passthrough",
    )
    model = Pipeline([
        ("prepare", encoder),
        ("classifier", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    report = classification_report(y_test, predicted, labels=LABELS, output_dict=True, zero_division=0)
    info = {
        "dataset": "SYNTHETIC, simulated latent classes with overlapping distributions",
        "split": "80% training sessions / 20% held-out sessions (GroupShuffleSplit)",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "training_sessions": int(train.session_id.nunique()),
        "test_sessions": int(test.session_id.nunique()),
        "test_accuracy": float(accuracy_score(y_test, predicted)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "test_macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "report": report,
    }
    joblib.dump(model, MODEL_PATH)
    (REPORTS / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (REPORTS / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(classification_report(y_test, predicted, labels=LABELS, zero_division=0))
    results = test[["session_id", "timestamp", "department", "status"]].copy()
    results["predicted_status"] = predicted
    results.to_csv(REPORTS / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=LABELS,
        display_labels=LABELS, cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Held-out synthetic sessions: confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Permutation importance operates on original input columns, including category.
    subset = X_test.sample(n=min(900, len(X_test)), random_state=42)
    subset_y = y_test.loc[subset.index]
    importance = permutation_importance(model, subset, subset_y, n_repeats=3,
        random_state=42, scoring="f1_macro", n_jobs=1)
    importance_df = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean})
    importance_df = importance_df.sort_values("importance", ascending=True)
    importance_df.to_csv(REPORTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#2870b2")
    ax.set_xlabel("Permutation decrease in macro F1 (held-out synthetic subset)")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=160)
    plt.close(fig)

    print("AirportNet-AI training complete (SYNTHETIC evaluation)")
    print(f"Train: {len(train)} rows / {train.session_id.nunique()} sessions")
    print(f"Test:  {len(test)} rows / {test.session_id.nunique()} sessions")
    print(f"Accuracy: {info['test_accuracy']:.3f}; Macro F1: {info['test_macro_f1']:.3f}; Majority baseline: {info['majority_baseline_accuracy']:.3f}")
    print(classification_report(y_test, predicted, labels=LABELS, zero_division=0))
    print("Saved model to", MODEL_PATH)
    print("Saved plots, classification report, and test predictions to", REPORTS)
    print("WARNING: All measured accuracy is on synthetic scenarios, not real airport traffic.")


if __name__ == "__main__":
    main()
