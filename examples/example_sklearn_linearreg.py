from sklearn.model_selection import GridSearchCV
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

    # Create linear regression object
    regr = LinearRegression(passes=5, header_dict=header_dict)

    params = {'passes': [1, 2, 3, 10, 30, 50, 100, 200, 500], 'header_dict': [header_dict], \
              'learning_rate': [0.2, 0.5, 0.8, 1.0], 'log_stderr_to_file': [True]}  # 'loss':['logistic'],

    gs = GridSearchCV(regr, params, scoring='neg_mean_squared_error', n_jobs=4)

    gs.fit(diabetes_X_train, diabetes_y_train)
    print(gs.best_score_)
    print(gs.best_params_)
