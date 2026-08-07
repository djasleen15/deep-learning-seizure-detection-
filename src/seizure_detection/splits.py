"""Patient-level splits for CHB-MIT experiments.

The final project's main constraint is strict patient-level separation:
no patient may appear in more than one split.
"""

from __future__ import annotations


TRAIN_PATIENTS = [f"chb{i:02d}" for i in range(1, 19)]
VAL_PATIENTS = [f"chb{i:02d}" for i in range(19, 22)]
TEST_PATIENTS = [f"chb{i:02d}" for i in range(22, 25)]
ALL_PATIENTS = TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS


PATIENT_SPLITS = {
    "train": TRAIN_PATIENTS,
    "val": VAL_PATIENTS,
    "test": TEST_PATIENTS,
}


def split_for_patient(patient: str) -> str:
    """Return the split name for one CHB-MIT patient id."""
    for split, patients in PATIENT_SPLITS.items():
        if patient in patients:
            return split

    raise ValueError(f"Unknown patient or patient outside configured splits: {patient}")

