from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
import random
from math import sqrt

from led_wabbit.models import MulticlassOAA

if __name__ == '__main__':
    X = list()
    Y = list()
    W = list()
    for class_index, class_name in enumerate(['cat', 'dog', 'mouse']):
        cur_w = sqrt(class_index+1)
        for i in range(3000):
            Y.append(class_name)
            X.append([class_index+1.5*random.random(),random.random(), random.random()*10])
            W.append(cur_w)

    X, Y = shuffle(X, Y)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, Y, W, test_size=0.1)

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0')}

    clf = MulticlassOAA(passes=5, header_dict=header_dict, oaa=3)  # loss='logistic',

    params = {'passes': [50, 100],
              'header_dict': [header_dict],
              'quadratic': ['XY', 'YZ', 'XY YZ'],
              'learning_rate': [0.5, 0.2, 0.8]}

    gs = GridSearchCV(clf, params, scoring='log_loss', fit_params={'sample_weight': w_train}, n_jobs=4)

    gs.fit(X_train, y_train)

    print('best logloss is {}'.format(gs.best_score_))
    print('best params are {}'.format(gs.best_params_))