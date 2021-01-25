from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from itertools import chain
import numpy as np
import random

from led_wabbit.models import LogisticRegressionBinary

if __name__ == '__main__':
    X1 = [[0, 1, random.random()*3] for i in range(40)]
    X2 = [[0, 2, random.random()*3-1] for i in range(40)]
    X3 = [[1, 0, random.random()*3+1] for i in range(40)]
    X4 = [[0, 2, random.random()*3-2] for i in range(3)]

    X = np.array([x for x in chain(X1, X2, X3, X4)])

    Y1 = [0 for i in range(40)]
    Y2 = [1 for i in range(40)]
    Y3 = [0 for i in range(40)]
    Y4 = [1 for i in range(3)]

    Y = np.array([y for y in chain(Y1, Y2, Y3, Y4)])

    X, Y = shuffle(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0')}

    clf = LogisticRegressionBinary(passes=50,
                                   learning_rate=0.5,
                                   header_dict=header_dict,
                                   quadratic='XY YZ')

    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)

    print(log_loss(y_test,preds))