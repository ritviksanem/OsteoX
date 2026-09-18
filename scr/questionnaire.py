import pandas as pd
from pathlib import Path

print("======================================")
print("       KNEESENSE NER QUESTIONNAIRE")
print("======================================")

print("\nPlease enter the following information.\n")

age = int(input("Age: "))

sex = input("Sex (Male/Female/Other): ").strip()

pain = int(input("Current knee pain (0-10): "))

pain_duration = input(
    "Pain duration (None/<3 months/3-12 months/>1 year): "
).strip()

morning_stiffness = input(
    "Morning knee stiffness? (Yes/No): "
).strip().lower()

stairs_difficulty = input(
    "Difficulty climbing stairs? (None/Mild/Severe): "
).strip()

walking_difficulty = input(
    "Difficulty walking? (None/Mild/Severe): "
).strip()

previous_injury = input(
    "Previous knee injury? (Yes/No): "
).strip().lower()

previous_surgery = input(
    "Previous knee surgery? (Yes/No): "
).strip().lower()

family_history = input(
    "Family history of osteoarthritis? (Yes/No): "
).strip().lower()


# Convert categorical answers to numerical values

sex_value = {
    "male": 0,
    "female": 1,
    "other": 2
}.get(sex.lower(), 2)

pain_duration_value = {
    "none": 0,
    "<3 months": 1,
    "3-12 months": 2,
    ">1 year": 3
}.get(pain_duration, 0)

morning_stiffness_value = 1 if morning_stiffness == "yes" else 0

stairs_value = {
    "none": 0,
    "mild": 1,
    "severe": 2
}.get(stairs_difficulty.lower(), 0)

walking_value = {
    "none": 0,
    "mild": 1,
    "severe": 2
}.get(walking_difficulty.lower(), 0)

previous_injury_value = 1 if previous_injury == "yes" else 0
previous_surgery_value = 1 if previous_surgery == "yes" else 0
family_history_value = 1 if family_history == "yes" else 0


data = {
    "age": [age],
    "sex": [sex_value],
    "pain": [pain],
    "pain_duration": [pain_duration_value],
    "morning_stiffness": [morning_stiffness_value],
    "stairs_difficulty": [stairs_value],
    "walking_difficulty": [walking_value],
    "previous_injury": [previous_injury_value],
    "previous_surgery": [previous_surgery_value],
    "family_history": [family_history_value]
}

df = pd.DataFrame(data)

project_root = Path(__file__).resolve().parent.parent
features_folder = project_root / "data" / "features"

features_folder.mkdir(parents=True, exist_ok=True)

file_path = features_folder / "questionnaire_features.csv"

df.to_csv(file_path, index=False)

print("\n======================================")
print("     QUESTIONNAIRE SAVED")
print("======================================")

print("\nSaved to:")
print(file_path)

print("\nThank you.")