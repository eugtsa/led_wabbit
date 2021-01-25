from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
import random
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import log_loss

from led_wabbit.models import MulticlassOAA

if __name__ == '__main__':
    X = list()
    Y = list()
    for class_index, class_name in enumerate(['cat', 'dog', 'mouse']):
        for i in range(3000):
            Y.append(class_name)
            X.append([class_index+1.5*random.random(),random.random(), random.random()*10])

    X, Y = shuffle(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0')}

    clf = MulticlassOAA(passes=5, header_dict=header_dict, oaa=3)  # loss='logistic',

    params = {'passes': [50, 100], 'header_dict': [header_dict], \
              'learning_rate': [0.5, 0.2, 0.8],'quadratic':['XY'],'log_stderr_to_file': [True]}  # 'loss':['logistic'],

    gs = GridSearchCV(clf, params, scoring='log_loss')#, n_jobs=4)

    preds = cross_val_predict(MulticlassOAA(passes=5, header_dict=header_dict, learning_rate=0.2, quadratic='XY',
                                            oaa=3, log_stderr_to_file=True), X_train, y_train)

    gs.fit(X_train, y_train)
    best_clf = gs.best_estimator_

    probs = best_clf.predict_proba(X_test)

    print('best log_loss is {}'.format(log_loss(y_test,probs)))
  #  print('best params are {}'.format(gs.best_params_))

    #print(probs)