import typing as t
from pathlib import Path

import joblib
import pandas as pd

from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    return dataframe

def save_model(*, model_file: str):

        # Prepare versioned save file name
    save_file_name = f"{config.app_config.save_model_file}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    joblib.dump(model_file, save_path)    


def load_model(*, file_name: str) -> None:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model    