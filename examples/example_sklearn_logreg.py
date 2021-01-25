from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from itertools import chain
import numpy as np

from led_wabbit.models import LogisticRegressionBinary

if __name__ == '__main__':
    X1 = [[0, 1, 1] for i in range(40)]
    X2 = [[0, 2, 0] for i in range(40)]
    X3 = [[1, 0, 1] for i in range(40)]
    X4 = [[0, 2, 2] for i in range(3)]

    X = np.array([x for x in chain(X1, X2, X3, X4)])

    Y1 = [0 for i in range(40)]
    Y2 = [1 for i in range(40)]
    Y3 = [0 for i in range(40)]
    Y4 = [1 for i in range(3)]

    Y = np.array([y for y in chain(Y1, Y2, Y3, Y4)])

    X, Y = shuffle(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0')}

    clf = LogisticRegressionBinary(learning_rate=5, header_dict=header_dict)  # loss='logistic',

    params = {'passes': [50, 100], 'header_dict': [header_dict], \
              'learning_rate': [0.5, 0.2, 0.8], 'log_stderr_to_file': [True]}  # 'loss':['logistic'],

    gs = GridSearchCV(clf, params, scoring='roc_auc', n_jobs=4)

    gs.fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)