"""Health risk model: trains once, saves to disk, and exposes predict_risk()."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "risk_model.joblib"

FEATURES = ["age", "bmi", "glucose", "bp"]
HIGH_RISK_THRESHOLD = 60  # percent; at or above this is "High Risk"
MODERATE_RISK_THRESHOLD = 30  # percent; at or above this is "Moderate Risk"

# Plausible ranges used to reject impossible inputs
VALID_RANGES = {
    "age": (1, 120),
    "bmi": (10, 70),
    "glucose": (40, 500),
    "bp": (40, 250),
}


def train_model(save=True):
    """Train on dataset.csv, print evaluation, and (optionally) save the model."""
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate on data the model has not seen
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))

    # Refit on all data for the final model
    model.fit(X, y)
    if save:
        joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    """Load the saved model, or train and save it on first run."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model(save=True)


_model = load_model()


def _validate(values):
    for name, value in values.items():
        low, high = VALID_RANGES[name]
        if not (low <= value <= high):
            raise ValueError(f"{name} must be between {low} and {high} (got {value}).")


def predict_risk(age, bmi, glucose, bp):
    """Return (risk percentage, risk level). Same interface as before."""
    _validate({"age": age, "bmi": bmi, "glucose": glucose, "bp": bp})

    row = pd.DataFrame([[age, bmi, glucose, bp]], columns=FEATURES)
    prob = _model.predict_proba(row)[0][1]
    percentage = round(float(prob) * 100, 2)
    if percentage >= HIGH_RISK_THRESHOLD:
        level = "High Risk"
    elif percentage >= MODERATE_RISK_THRESHOLD:
        level = "Moderate Risk"
    else:
        level = "Low Risk"
    return percentage, level


if __name__ == "__main__":
    # Run `python model.py` to retrain and see evaluation metrics
    train_model(save=True)