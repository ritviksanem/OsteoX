import pandas as pd
import joblib
from pathlib import Path

# ======================================
# PATHS
# ======================================

project_root = Path(__file__).resolve().parent.parent

features_folder = project_root / "data" / "features"
models_folder = project_root / "models"

walking_file = features_folder / "walking_features.csv"
questionnaire_file = features_folder / "questionnaire_features.csv"
model_file = models_folder / "oa_risk_model.pkl"


# ======================================
# LOAD MODEL
# ======================================

model = joblib.load(model_file)

walking = pd.read_csv(walking_file)
questionnaire = pd.read_csv(questionnaire_file)

print("======================================")
print("       KNEESENSE NER AI SCREENING")
print("======================================")


# ======================================
# DISPLAY PATIENT INFORMATION
# ======================================

print("\nPATIENT INFORMATION")
print("--------------------------------------")

print("Age              :", questionnaire["age"].iloc[0])
print("Pain Score       :", questionnaire["pain"].iloc[0], "/ 10")

print(
    "Morning Stiffness:",
    "Yes" if questionnaire["morning_stiffness"].iloc[0] == 1 else "No"
)

print(
    "Previous Injury  :",
    "Yes" if questionnaire["previous_injury"].iloc[0] == 1 else "No"
)

print(
    "Family History   :",
    "Yes" if questionnaire["family_history"].iloc[0] == 1 else "No"
)


# ======================================
# FEATURES
# ======================================

features = [
    "left_rom",
    "right_rom",
    "rom_asymmetry",
    "left_average_angle",
    "right_average_angle",
    "left_consistency",
    "right_consistency",

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


# ======================================
# COMBINE MOVEMENT + QUESTIONNAIRE
# ======================================

combined = pd.DataFrame()

for feature in features:

    if feature in walking.columns:
        combined[feature] = [walking[feature].iloc[0]]

    elif feature in questionnaire.columns:
        combined[feature] = [questionnaire[feature].iloc[0]]

    else:
        print("Missing feature:", feature)
        raise ValueError(
            f"Required feature '{feature}' not found."
        )


# ======================================
# DISPLAY MOVEMENT FEATURES
# ======================================

print("\nMOVEMENT FEATURES")
print("--------------------------------------")

print(
    "Left ROM          :",
    round(combined["left_rom"].iloc[0], 2)
)

print(
    "Right ROM         :",
    round(combined["right_rom"].iloc[0], 2)
)

print(
    "ROM Asymmetry     :",
    round(combined["rom_asymmetry"].iloc[0], 2),
    "%"
)

print(
    "Left Consistency  :",
    round(combined["left_consistency"].iloc[0], 2),
    "%"
)

print(
    "Right Consistency :",
    round(combined["right_consistency"].iloc[0], 2),
    "%"
)


# ======================================
# AI PREDICTION
# ======================================

X = combined[features]

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

classes = model.classes_


# ======================================
# RESULT
# ======================================

print("\n======================================")
print("             AI RESULT")
print("======================================")

print("\nPreliminary Risk Category:", prediction)

print("\nModel probability:")

for class_name, probability in zip(classes, probabilities):
    print(
        f"{class_name:10s}: {probability * 100:.2f}%"
    )


# ======================================
# RECOMMENDATION
# ======================================

print("\n======================================")
print("          RECOMMENDATION")
print("======================================")

if prediction == "LOW":

    print("""
Low preliminary risk based on the
movement and questionnaire features.

Continue healthy physical activity
and monitor knee symptoms.
""")

elif prediction == "MODERATE":

    print("""
Moderate preliminary risk detected.

Consider further assessment by a
qualified healthcare professional,
especially if pain or stiffness persists.
""")

else:

    print("""
High preliminary risk detected.

Further clinical evaluation by a
qualified healthcare professional
is recommended.
""")


# ======================================
# DISCLAIMER
# ======================================

print("======================================")
print("NOTE:")
print("This is a preliminary screening")
print("system and NOT a medical diagnosis.")
print("The current AI model uses synthetic")
print("prototype training data.")
print("Clinical validation is required.")
print("======================================")