import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "walking_raw.csv"

print("======================================")
print("       WALKING DATA VISUALIZATION")
print("======================================")

df = pd.read_csv(file_path)

print("\nFile loaded successfully.")
print("Samples:", len(df))

print("\nColumns found:")
print(df.columns.tolist())

# Convert columns to numbers
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["left_knee_angle_raw"] = pd.to_numeric(
    df["left_knee_angle_raw"], errors="coerce"
)
df["right_knee_angle_raw"] = pd.to_numeric(
    df["right_knee_angle_raw"], errors="coerce"
)
df["left_knee_angle_smoothed"] = pd.to_numeric(
    df["left_knee_angle_smoothed"], errors="coerce"
)
df["right_knee_angle_smoothed"] = pd.to_numeric(
    df["right_knee_angle_smoothed"], errors="coerce"
)

df = df.dropna()

print("\nValid samples:", len(df))

# Create graph
plt.figure(figsize=(14, 7))

# Raw signals
plt.plot(
    df["time"],
    df["left_knee_angle_raw"],
    alpha=0.25,
    label="Left Knee - Raw"
)

plt.plot(
    df["time"],
    df["right_knee_angle_raw"],
    alpha=0.25,
    label="Right Knee - Raw"
)

# Smoothed signals
plt.plot(
    df["time"],
    df["left_knee_angle_smoothed"],
    linewidth=2,
    label="Left Knee - Smoothed"
)

plt.plot(
    df["time"],
    df["right_knee_angle_smoothed"],
    linewidth=2,
    label="Right Knee - Smoothed"
)

plt.title("Knee Angle During Walking")
plt.xlabel("Time (seconds)")
plt.ylabel("Knee Angle (degrees)")

plt.ylim(30, 190)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

print("\nShowing graph...")
plt.show(block=True)