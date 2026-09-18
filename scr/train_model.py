from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDER = PROJECT_ROOT / "models"
FEATURE_FOLDER = PROJECT_ROOT / "data" / "features"
DATA_PATH = FEATURE_FOLDER / "training_dataset.csv"
MODEL_PATH = MODEL_FOLDER / "oa_risk_model.pkl"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD CLINICAL DATASET
# ============================================================
if not DATA_PATH.exists():
  raise FileNotFoundError(
      f"Dataset not found at {DATA_PATH}. Run preprocess_oai.py first."
  )

print(f"\nLoading real clinical dataset: {DATA_PATH.name}")
df = pd.read_csv(DATA_PATH)
print(f"Total patient records: {len(df)}")

print("\nClass distribution:")
print(df["risk"].value_counts())

# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================
FEATURES = [
    # Kinematic Movement features
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
    "family_history",
]

X = df[FEATURES]
y = df["risk"]

# ============================================================
# STRATIFIED TRAIN / TEST SPLIT (80% Train, 20% Test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================
print("\nTraining Random Forest model on OAI clinical records...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

# ============================================================
# CLINICAL EVALUATION
# ============================================================
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print("\n========================================")
print("CLINICAL MODEL EVALUATION")
print("========================================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ============================================================
# SAVE TRAINED ESTIMATOR
# ============================================================
joblib.dump(model, MODEL_PATH)

print("========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")
print(f"Model saved to: {MODEL_PATH}")
print(f"Trained on {len(X_train)} samples across {len(FEATURES)} features.")