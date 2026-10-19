
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "data" / "raw" / "metadata"
OUTPUT = ROOT / "data" / "processed"

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

df = pd.read_csv(METADATA / "Data_Entry_2017_v2020.csv")

# Convert each image's findings into 14 independent binary labels.
findings = df["Finding Labels"].str.split("|")

known_labels = set(LABELS) | {"No Finding"}
unknown = set(label for row in findings for label in row) - known_labels
if unknown:
    raise ValueError(f"Unexpected labels found: {unknown}")

for label in LABELS:
    df[label] = findings.apply(lambda row: int(label in row))

# Assign entire patients to splits, never individual images.
patients = df["Patient ID"].drop_duplicates()

train_patients, temp_patients = train_test_split(
    patients, test_size=0.30, random_state=42
)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.50, random_state=42
)

split_map = {
    **{p: "train" for p in train_patients},
    **{p: "val" for p in val_patients},
    **{p: "test" for p in test_patients},
}

df["split"] = df["Patient ID"].map(split_map)

OUTPUT.mkdir(parents=True, exist_ok=True)

columns = ["Image Index", "Patient ID", "Finding Labels", "split"] + LABELS
df[columns].to_csv(OUTPUT / "splits.csv", index=False)

# Verify that patient leakage is impossible.
patient_sets = {
    name: set(df.loc[df["split"] == name, "Patient ID"])
    for name in ["train", "val", "test"]
}

assert patient_sets["train"].isdisjoint(patient_sets["val"])
assert patient_sets["train"].isdisjoint(patient_sets["test"])
assert patient_sets["val"].isdisjoint(patient_sets["test"])

print("Images per split:")
print(df["split"].value_counts())

print("\nUnique patients per split:")
print(df.groupby("split")["Patient ID"].nunique())

print("\nPositive image labels per split:")
print(df.groupby("split")[LABELS].sum().T.to_string())

print("\nSaved:", OUTPUT / "splits.csv")
print("Patient leakage check passed.")
