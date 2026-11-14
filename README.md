# AI-Based School Safety & Real-Time Emergency Alert System

**Machine Learning model:** Multiclass Logistic Regression  
**Environment:** Python / Google Colab  
**Project type:** Educational prototype using simulated sensor data

## 1. Problem statement
Schools and campuses need a way to identify abnormal conditions such as smoke,
fire, gas leakage, unusual heat, excessive sound, or crowd disturbance. This
prototype classifies sensor readings into three severity levels and generates
a demonstration alert.

## 2. Objectives
- Classify a situation as Safe, Normal Alert, or High-Priority Emergency.
- Train and evaluate a multiclass Logistic Regression model.
- Convert predictions into readable alert messages.
- Simulate a sequence of incoming readings.
- Automatically mark an active alert as cleared when a later prediction is Safe.
- Store the simulated alert history in a CSV file.

## 3. Dataset and input features
The notebook loads the simulated dataset from:

`https://github.com/YBIFoundation/Internship/raw/main/SchoolSafetyEmergencyDataset.csv`

The model uses these nine input features:
- Temperature (°C)
- Smoke level
- Carbon monoxide (CO) concentration
- Gas level
- Sound level (dB)
- Crowd count
- Motion index
- Flame detection
- Event duration (minutes)

`incident_type` and `severity_label` are descriptive fields. The target column is
`severity`: 0 = Safe, 1 = Normal Alert, and 2 = High-Priority Emergency.

## 4. Methodology
1. Load the dataset using pandas.
2. Separate the nine model features from the target.
3. Split the data into 80% training and 20% testing using stratification.
4. Standardize numerical inputs with `StandardScaler`.
5. Train `LogisticRegression` with balanced class weights.
6. Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Predict a sample sensor reading and generate an alert message.
8. Simulate a sensor stream and record alert/clear events.
9. Export the history to `alert_history.csv`.

## 5. Expected output
The original project run reported approximately **94% test accuracy** on its
80/20 split (240 test rows). Results can vary if the dataset or code changes.
Run the notebook to generate the current accuracy and classification report.

## 6. Run in Google Colab
1. Open https://colab.research.google.com/
2. Upload `School_Safety_Emergency_Alert.ipynb`.
3. Select **Runtime → Run all**.
4. Allow the notebook to download the dataset from the source URL.
5. Review the evaluation output, sample prediction, and alert history.

## 7. Run locally
```bash
pip install -r requirements.txt
python school_safety.py
```

## 8. Repository contents
- `School_Safety_Emergency_Alert.ipynb` — complete Colab/Jupyter notebook.
- `school_safety.py` — runnable Python implementation.
- `requirements.txt` — required Python packages.
- `README.md` — project overview and instructions.
- `.gitignore` — excludes generated files and local environments.

## 9. Limitations and safety note
This is an academic demonstration using a simulated dataset and simulated
monitoring stream. It does not connect to real cameras or physical sensors and
does not send SMS, email, or WhatsApp messages. The dataset and model are not
validated for real-world emergency detection. A real deployment would require
validated sensors, reliable communications, extensive testing, human oversight,
and approval by qualified safety professionals. Do not rely on this prototype
to make real emergency decisions.
