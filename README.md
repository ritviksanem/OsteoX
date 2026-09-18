```markdown
# OsteoX (KneeSense)

An AI-assisted multimodal musculoskeletal screening and longitudinal tracking platform designed to evaluate preliminary markers of Knee Osteoarthritis (OA). 

OsteoX combines subjective clinical symptoms, computer-vision gait kinematics, and deep learning radiographic analysis into a unified triage interface built with an Apple Health / Samsung Health Bento design system.

---

## Key Capabilities

* **Subjective Assessment (Clinical Questionnaire):** Evaluates patient symptoms, morning stiffness, functional pain, and joint injury history.
* **Functional Kinematics (30s Computer Vision Gait Analysis):** Uses Google MediaPipe Pose to track knee flexion/extension angles, Range of Motion (ROM), joint symmetry, and cycle consistency in real time via webcam.
* **Structural Radiography (Deep Learning X-Ray Classifier):** Evaluates standard anterior-posterior (AP) knee radiographs using a fine-tuned ResNet-18 vision model (trained on NIH Osteoarthritis Initiative cohort data).
* **Multimodal Clinical Synthesis:** Dynamically cross-references functional biomechanical deficits against radiographic joint findings to classify clinical concordance.
* **Longitudinal Patient Records:** Persistent SQLite backend (`osteox_patients.db`) providing visit histories and ROM recovery/degeneration trajectories across multiple appointments.

---

## Project Architecture

```text
Kneesense/
├── app/
│   ├── app.py                 # Streamlit clinical dashboard & workflow
│   ├── styles.py              # Apple/Samsung Health design system & CSS
│   └── ui_components.py       # Bento metric tiles and diagnostic cards
├── data/
│   ├── raw/                   # OAI tabular clinical datasets
│   ├── features/              # Extracted questionnaire & movement features
│   └── xray/                  # AP knee radiograph dataset (train/val/test)
├── models/
│   ├── oa_risk_model.pkl      # Random Forest kinematic + clinical model
│   ├── knee_xray_model.pth    # Fine-tuned ResNet-18 vision checkpoint
│   └── pose_landmarker_lite.task # MediaPipe Tasks pose model
├── scr/
│   ├── db.py                  # SQLite schema, migrations, and visit logging
│   ├── train_model.py         # Tabular ML training pipeline
│   ├── train_xray.py          # PyTorch transfer learning training script
│   └── xray_inference.py      # ResNet-18 inference and scoring wrapper
└── requirements.txt

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/KneeSense.git](https://github.com/your-username/KneeSense.git)
cd KneeSense

```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

```

### 3. Install Dependencies

Install core packages:

```bash
pip install streamlit opencv-python mediapipe numpy pandas scipy av streamlit-webrtc joblib pillow

```

Install PyTorch with NVIDIA CUDA acceleration (recommended for GPU inference):

```bash
# NVIDIA CUDA 12.4 build:
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# Or CPU-only build:
# pip install torch torchvision

```

---

## Running the Application

Launch the local Streamlit dashboard:

```bash
streamlit run app/app.py

```

1. **Step 1:** Register a new patient or select an existing patient ID.
2. **Step 2:** Fill in symptom scoring and pain history.
3. **Step 3:** Perform the 30-second webcam walking test (stand 2–3m back with full body in view).
4. **Step 4 & 5:** Review the AI-predicted risk tier, kinematic metrics, and longitudinal ROM graph.
5. **Step 6 (Optional):** Upload an AP knee radiograph (`.jpg`/`.png`) to inspect structural joint space findings.

---

## Disclaimer

OsteoX is a preliminary screening and research tool. It does not provide definitive medical diagnoses. All risk predictions and radiographic assessments must be corroborated by a qualified orthopedic specialist or registered medical practitioner.

```

Save the file and commit it to git:

```powershell
git add README.md
git commit -m "docs: add comprehensive project README and setup guide"
git push origin main

```
