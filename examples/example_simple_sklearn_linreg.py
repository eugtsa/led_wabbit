from sklearn.metrics import mean_squared_error
from led_wabbit.models import LinearRegression

import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

if __name__ == '__main__':
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    header_dict = {0: ('n', 'X', 'x0')}

    # Create linear regression model
    lr = LinearRegression(learning_rate=0.8, passes=5, header_dict=header_dict)

    lr.fit(diabetes_X_train, diabetes_y_train)

    preds_probas = lr.predict(diabetes_X_test)

    print(mean_squared_error(diabetes_y_test,preds_probas))

