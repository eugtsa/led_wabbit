from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import random
import numpy as np

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

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0'), 3: ('c','A','some_category')}

    clf = MulticlassOAA(passes=5, header_dict=header_dict, oaa=3, quadratic='XY', log_stderr_to_file=True)

    clf = CalibratedClassifierCV(clf, cv=8)

    preds = cross_val_predict(clf, X, Y, n_jobs=4)

    #gs.fit(X_train, y_train)
    #best_clf = gs.best_estimator_

    clf.fit(X,Y)
    probs = clf.predict_proba(X)

    print('best log_loss is {}'.format(log_loss(y_test,probs)))