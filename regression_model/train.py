import numpy as np
from config.core import config
from processing.data_manager import load_dataset, save_model
from sklearn.model_selection import train_test_split
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=config.model_config.test_size, 
        random_state=config.model_config.random_state
    )
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Save the model
    save_model(model_file=regressor)

if __name__ == '__main__':
    run_training()    




