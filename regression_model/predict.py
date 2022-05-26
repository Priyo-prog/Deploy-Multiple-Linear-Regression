import typing as t

import numpy as np
import pandas as pd

from regression_model.config.core import config
from regression_model.processing.data_manager import load_model
from processing.data_manager import load_dataset, save_model

model_file_name = f"{config.app_config.save_model_file}.pkl"
salary_model = load_model(file_name=model_file_name)

data = load_dataset(file_name=config.app_config.training_data_file)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=config.model_config.test_size, 
    random_state=config.model_config.random_state
)

input_data = X_test

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)

    predictions = salary_model.predict(
            X=data
        )
    return predictions    
