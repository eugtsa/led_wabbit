from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from itertools import chain
import numpy as np
import random
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import random
import numpy as np
from itertools import chain

from led_wabbit.models import MulticlassOAA

if __name__ == '__main__':
    X = list()
    Y = list()

    X_feats=[['aa','bbbbbb'],['bb','cc'],['cc',np.nan]]
    for class_index, class_name in enumerate(['cat', 'dog', 'mouse']):
        for i in range(3000):
            Y.append(class_name)
            X.append([class_index+1.5*random.random(),random.random(), random.random()*10,
                      random.choice(X_feats[class_index])])

    X, Y = shuffle(X, Y)

    X_tr, X_test, y_tr, y_test = train_test_split(X, Y, test_size=0.1)
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0'), 3: ('c','A','some_category')}

    all_preds = []
    all_true = []

    skf = StratifiedKFold(y_tr, n_folds=8)

    for train_index, test_index in skf:
        X_train, X_test = X_tr[train_index], X_tr[test_index]
        y_train, y_test = y_tr[train_index], y_tr[test_index]

        X_train, x_calibrate, y_train, y_calibrate = train_test_split(X_train, y_train, test_size=0.05)

        clf = MulticlassOAA(passes=1, header_dict=header_dict, oaa=3, learning_rate=0.025, quadratic='nf ni nn',
                            log_stderr_to_file=True)  # loss='logistic',
        clf.fit(X_train, y_train)

        clf = CalibratedClassifierCV(clf, cv='prefit')

        clf.fit(x_calibrate, y_calibrate)

        all_true.append(y_test)
        all_preds.append(clf.predict_proba(X_test))

    print('best log_loss is {}'.format(log_loss([l for l in chain(*y_test)],[l for l in chain(*all_preds)])))
    #print('best params are {}'.format(gs.best_params_))
