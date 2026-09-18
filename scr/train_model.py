from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDER = PROJECT_ROOT / "models"
FEATURE_FOLDER = PROJECT_ROOT / "data" / "features"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_FOLDER / "oa_risk_model.pkl"


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURES = [
    # Movement features
    "left_rom",
    "right_rom",
    "rom_asymmetry",
    "left_average_angle",
    "right_average_angle",
    "left_consistency",
    "right_consistency",

    # Questionnaire features
    "age",
    "sex",
    "pain",
    "pain_duration",
    "morning_stiffness",
    "stairs_difficulty",
    "walking_difficulty",
    "previous_injury",
    "previous_surgery",
    "family_history"
]


# ============================================================
# GENERATE SYNTHETIC PROTOTYPE DATA
# ============================================================

def generate_data():

    samples_per_class = 150

    data = []

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        left_rom = np.random.normal(45, 5)
        right_rom = np.random.normal(45, 5)

        rom_asymmetry = np.random.normal(5, 3)

        left_average_angle = np.random.normal(170, 4)
        right_average_angle = np.random.normal(170, 4)

        left_consistency = np.random.normal(88, 6)
        right_consistency = np.random.normal(88, 6)

        age = np.random.normal(30, 8)

        sex = np.random.choice([0, 1, 2])

        pain = np.random.normal(1.5, 1)

        pain_duration = np.random.choice([0, 1])

        morning_stiffness = np.random.choice([0, 1], p=[0.8, 0.2])

        stairs_difficulty = np.random.choice(
            [0, 1, 2],
            p=[0.75, 0.2, 0.05]
        )

        walking_difficulty = np.random.choice(
            [0, 1, 2],
            p=[0.85, 0.12, 0.03]
        )

        previous_injury = np.random.choice(
            [0, 1],
            p=[0.8, 0.2]
        )

        previous_surgery = np.random.choice(
            [0, 1],
            p=[0.95, 0.05]
        )

        family_history = np.random.choice(
            [0, 1],
            p=[0.75, 0.25]
        )

        data.append([
            left_rom,
            right_rom,
            rom_asymmetry,
            left_average_angle,
            right_average_angle,
            left_consistency,
            right_consistency,
            age,
            sex,
            pain,
            pain_duration,
            morning_stiffness,
            stairs_difficulty,
            walking_difficulty,
            previous_injury,
            previous_surgery,
            family_history,
            "LOW"
        ])

    # --------------------------------------------------------
    # MODERATE RISK
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        left_rom = np.random.normal(32, 5)
        right_rom = np.random.normal(34, 5)

        rom_asymmetry = np.random.normal(15, 5)

        left_average_angle = np.random.normal(160, 6)
        right_average_angle = np.random.normal(163, 6)

        left_consistency = np.random.normal(70, 8)
        right_consistency = np.random.normal(72, 8)

        age = np.random.normal(48, 10)

        sex = np.random.choice([0, 1, 2])

        pain = np.random.normal(4.5, 1.5)

        pain_duration = np.random.choice(
            [1, 2, 3],
            p=[0.15, 0.55, 0.30]
        )

        morning_stiffness = np.random.choice(
            [0, 1],
            p=[0.35, 0.65]
        )

        stairs_difficulty = np.random.choice(
            [0, 1, 2],
            p=[0.2, 0.6, 0.2]
        )

        walking_difficulty = np.random.choice(
            [0, 1, 2],
            p=[0.25, 0.6, 0.15]
        )

        previous_injury = np.random.choice(
            [0, 1],
            p=[0.6, 0.4]
        )

        previous_surgery = np.random.choice(
            [0, 1],
            p=[0.85, 0.15]
        )

        family_history = np.random.choice(
            [0, 1],
            p=[0.5, 0.5]
        )

        data.append([
            left_rom,
            right_rom,
            rom_asymmetry,
            left_average_angle,
            right_average_angle,
            left_consistency,
            right_consistency,
            age,
            sex,
            pain,
            pain_duration,
            morning_stiffness,
            stairs_difficulty,
            walking_difficulty,
            previous_injury,
            previous_surgery,
            family_history,
            "MODERATE"
        ])

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        left_rom = np.random.normal(20, 5)
        right_rom = np.random.normal(22, 5)

        rom_asymmetry = np.random.normal(28, 7)

        left_average_angle = np.random.normal(148, 8)
        right_average_angle = np.random.normal(152, 8)

        left_consistency = np.random.normal(52, 10)
        right_consistency = np.random.normal(55, 10)

        age = np.random.normal(62, 10)

        sex = np.random.choice([0, 1, 2])

        pain = np.random.normal(7, 1.5)

        pain_duration = np.random.choice(
            [2, 3],
            p=[0.35, 0.65]
        )

        morning_stiffness = np.random.choice(
            [0, 1],
            p=[0.1, 0.9]
        )

        stairs_difficulty = np.random.choice(
            [1, 2],
            p=[0.35, 0.65]
        )

        walking_difficulty = np.random.choice(
            [1, 2],
            p=[0.35, 0.65]
        )

        previous_injury = np.random.choice(
            [0, 1],
            p=[0.4, 0.6]
        )

        previous_surgery = np.random.choice(
            [0, 1],
            p=[0.75, 0.25]
        )

        family_history = np.random.choice(
            [0, 1],
            p=[0.3, 0.7]
        )

        data.append([
            left_rom,
            right_rom,
            rom_asymmetry,
            left_average_angle,
            right_average_angle,
            left_consistency,
            right_consistency,
            age,
            sex,
            pain,
            pain_duration,
            morning_stiffness,
            stairs_difficulty,
            walking_difficulty,
            previous_injury,
            previous_surgery,
            family_history,
            "HIGH"
        ])

    return pd.DataFrame(
        data,
        columns=FEATURES + ["risk"]
    )


# ============================================================
# CREATE DATASET
# ============================================================

print("\nGenerating synthetic prototype dataset...")

df = generate_data()

print(f"Dataset size: {len(df)} samples")
print("\nClass distribution:")
print(df["risk"].value_counts())


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURES]
y = df["risk"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


print("\nTraining model...")

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Prototype accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL USING JOBLIB
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(f"Model location:")
print(MODEL_PATH)

print("\nModel features:")
for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")

print("\nIMPORTANT:")
print("This model is trained on SYNTHETIC prototype data.")
print("It is NOT clinically validated.")
print("It must NOT be used as a medical diagnosis.")