"""
AI-Based School Safety & Real-Time Emergency Alert System
Educational prototype using Multiclass Logistic Regression.

This project uses a public, simulated dataset. It is not a certified
life-safety system and must not be used for real emergency decisions.
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA_URL = (
    "https://github.com/YBIFoundation/Internship/raw/main/"
    "SchoolSafetyEmergencyDataset.csv"
)

CLASS_NAMES = {
    0: "Safe",
    1: "Normal Alert",
    2: "High-Priority Emergency",
}

FEATURE_COLS = [
    "temperature_c",
    "smoke_level",
    "co_ppm",
    "gas_level",
    "sound_db",
    "crowd_count",
    "motion_index",
    "flame_detected",
    "event_duration_min",
]


def load_data():
    """Load the simulated school-safety dataset."""
    return pd.read_csv(DATA_URL)


def train_model(df):
    """Split the data, train the model, and return evaluation results."""
    X = df[FEATURE_COLS]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")
    print(f"Accuracy     : {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[CLASS_NAMES[i] for i in [0, 1, 2]],
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1, 2],
        display_labels=[CLASS_NAMES[i] for i in [0, 1, 2]],
        xticks_rotation=20,
    )
    plt.title("School Safety Model - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model


def create_alert_message(predicted_class):
    """Convert the predicted class into a demonstration alert."""
    if predicted_class == 0:
        return "CLEAR / SAFE: No active emergency."
    if predicted_class == 1:
        return "NORMAL ALERT: Please inspect the location."
    return "HIGH-PRIORITY EMERGENCY: Immediate response required!"


def predict_reading(model, reading):
    """Predict one sensor reading and display class probabilities."""
    sample = pd.DataFrame([reading], columns=FEATURE_COLS)
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    print("Predicted status:", CLASS_NAMES[prediction])
    print(create_alert_message(prediction))
    print("\nClass probabilities:")
    for class_id, probability in zip(model.classes_, probabilities):
        print(f"{CLASS_NAMES[int(class_id)]:25s}: {probability:.3f}")

    return prediction


def simulate_monitoring(model, df, count=20):
    """
    Treat selected dataset rows as an incoming sensor stream.
    This is a simulation, not a connection to physical sensors.
    """
    risky = df[df["severity"].isin([1, 2])].sample(
        n=min(7, len(df[df["severity"].isin([1, 2])])),
        random_state=10,
    )
    safe = df[df["severity"] == 0].sample(
        n=min(4, len(df[df["severity"] == 0])),
        random_state=30,
    )
    remaining = df.sample(n=min(count, len(df)), random_state=40)
    stream = pd.concat([risky, safe, remaining]).head(count).reset_index(drop=True)

    history = []
    active_alert = False

    for step, row in stream.iterrows():
        sample = row[FEATURE_COLS].to_frame().T
        predicted = int(model.predict(sample)[0])

        if predicted == 0:
            if active_alert:
                action = "CLEAR: Previous alert closed automatically"
                active_alert = False
            else:
                action = "SAFE: No active alert"
        elif predicted == 1:
            action = "NORMAL ALERT: Inspection required"
            active_alert = True
        else:
            action = "HIGH-PRIORITY EMERGENCY: Immediate response required"
            active_alert = True

        history.append(
            {
                "stream_step": step + 1,
                "incident_type": row.get("incident_type", "Not specified"),
                "temperature_c": row["temperature_c"],
                "smoke_level": row["smoke_level"],
                "flame_detected": row["flame_detected"],
                "predicted_status": CLASS_NAMES[predicted],
                "system_action": action,
            }
        )

    history_df = pd.DataFrame(history)
    print("\nLATEST MONITORING STATUS")
    print(history_df.tail(1).to_string(index=False))
    print("\nRECENT ALERT HISTORY")
    print(history_df.tail(10).to_string(index=False))
    return history_df


def main():
    print("Loading school safety dataset...")
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["severity_label"].value_counts())

    df["severity_label"].value_counts().plot(
        kind="bar",
        title="Safety Status Distribution",
        xlabel="Status",
        ylabel="Number of Records",
    )
    plt.tight_layout()
    plt.show()

    model = train_model(df)

    sample_reading = {
        "temperature_c": 65,
        "smoke_level": 88,
        "co_ppm": 40,
        "gas_level": 92,
        "sound_db": 101,
        "crowd_count": 48,
        "motion_index": 90,
        "flame_detected": 1,
        "event_duration_min": 8,
    }
    print("\nSAMPLE SENSOR READING")
    predict_reading(model, sample_reading)

    history = simulate_monitoring(model, df)
    history.to_csv("alert_history.csv", index=False)
    print("\nSaved alert history to alert_history.csv")


if __name__ == "__main__":
    main()
