from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "osteox_patients.db"
)


def get_connection():
  """Creates a database connection with foreign keys enabled."""
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.execute("PRAGMA foreign_keys = ON;")
  return conn


def init_db():
  """Initializes the database schema if tables do not exist."""
  with get_connection() as conn:
    cursor = conn.cursor()

    # Patients demographic table
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex INTEGER NOT NULL,
                contact TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # Screening visits longitudinal table
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS screening_visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Kinematic Features (7)
                left_rom REAL,
                right_rom REAL,
                rom_asymmetry REAL,
                left_avg_angle REAL,
                right_avg_angle REAL,
                left_consistency REAL,
                right_consistency REAL,
                
                -- Clinical Features (10)
                pain INTEGER,
                pain_duration INTEGER,
                morning_stiffness INTEGER,
                stairs_difficulty INTEGER,
                walking_difficulty INTEGER,
                previous_injury INTEGER,
                previous_surgery INTEGER,
                family_history INTEGER,
                
                -- Model Output
                predicted_risk TEXT NOT NULL,
                prob_low REAL,
                prob_moderate REAL,
                prob_high REAL,
                clinical_notes TEXT,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            );
        """)
    conn.commit()


def generate_patient_id():
  """Generates an incremental clinical ID like OX-2026-0001."""
  year = datetime.now().year
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM patients WHERE patient_id LIKE ?",
        (f"OX-{year}-%",),
    )
    count = cursor.fetchone()[0] + 1
    return f"OX-{year}-{count:04d}"


def register_patient(name: str, age: int, sex: int, contact: str = "") -> str:
  """Registers a new patient and returns the assigned patient_id."""
  pid = generate_patient_id()
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO patients (patient_id, full_name, age, sex, contact)
            VALUES (?, ?, ?, ?, ?)
        """,
        (pid, name.strip(), age, sex, contact.strip()),
    )
    conn.commit()
  return pid


def get_all_patients():
  """Returns a list of tuples: (patient_id, full_name, age, sex) for dropdowns."""
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT patient_id, full_name, age, sex FROM patients ORDER BY"
        " created_at DESC"
    )
    return cursor.fetchall()


def log_screening_visit(
    patient_id: str,
    kinematics: dict,
    questionnaire: dict,
    risk: str,
    probabilities: dict,
    notes: str = "",
):
  """Atomically records a screening session into screening_visits."""
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO screening_visits (
                patient_id,
                left_rom, right_rom, rom_asymmetry,
                left_avg_angle, right_avg_angle,
                left_consistency, right_consistency,
                pain, pain_duration, morning_stiffness,
                stairs_difficulty, walking_difficulty,
                previous_injury, previous_surgery, family_history,
                predicted_risk, prob_low, prob_moderate, prob_high, clinical_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            kinematics.get("left_rom"),
            kinematics.get("right_rom"),
            kinematics.get("rom_asymmetry"),
            kinematics.get("left_average_angle"),
            kinematics.get("right_average_angle"),
            kinematics.get("left_consistency"),
            kinematics.get("right_consistency"),
            questionnaire.get("pain"),
            questionnaire.get("pain_duration"),
            questionnaire.get("morning_stiffness"),
            questionnaire.get("stairs_difficulty"),
            questionnaire.get("walking_difficulty"),
            questionnaire.get("previous_injury"),
            questionnaire.get("previous_surgery"),
            questionnaire.get("family_history"),
            risk,
            probabilities.get("LOW", 0.0),
            probabilities.get("MODERATE", 0.0),
            probabilities.get("HIGH", 0.0),
            notes,
        ),
    )
    conn.commit()


def get_patient_history(patient_id: str) -> pd.DataFrame:
  """Retrieves all past visits for a specific patient as a DataFrame."""
  with get_connection() as conn:
    query = """
            SELECT visit_id, timestamp, predicted_risk, 
                   left_rom, right_rom, rom_asymmetry, pain, morning_stiffness
            FROM screening_visits
            WHERE patient_id = ?
            ORDER BY timestamp DESC
        """
    return pd.read_sql_query(query, conn, params=(patient_id,))


if __name__ == "__main__":
  init_db()
  print(f"Database initialized at {DB_PATH}")