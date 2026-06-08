import pandas as pd

files = [
    "data/raw/patients.csv",
    "data/raw/admissions.csv",
    "data/raw/icustays.csv",
    "data/raw/diagnoses_icd.csv",
    "data/raw/chart_events.csv",
    "data/raw/d_labitems.csv",
    "data/processed/sepsis_dataset.csv"
]

for file in files:
    df = pd.read_csv(file)
    print(file, "->", df.shape)