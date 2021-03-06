led_wabbit - wrapper with sklearn interface for vowpal wabbit
=======================

### Description
led_wabbit is wrapper for vowpal wabbit, fully implemented in python. It's my implementation of vowpal wabbit wrapper 
based on vowpal_porpoise (https://github.com/josephreisinger/vowpal_porpoise) developed by Austin Waters, Joseph Reisinger 
and Daniel Duckworth.

Why does it called led_wabbit and not vowpal_porpoise? Because it is:
* slow like turtle
* fully python
* has different purposes (for prototyping)

### Features

Currently implements:
* linear regression(LinearRegression)
* logistic regression(LogisticRegressionBinary)
* multiclass one-versus-all regression(MulticlassOAA)

These classes support sklearn interface with `fit()`, `predict()`, `predict_proba()` methods.  

### How to install

python2 install(run command without sudo):
`pip install -i https://testpypi.python.org/pypi led_wabbit --user --upgrade`


python3 install(run command without sudo):
`pip3 install -i https://testpypi.python.org/pypi led_wabbit --user --upgrade`


### Requirements

* `vw` command should be available

### Important remarks about usage

#### header_dict

One necessary parameter “header_dict” should be passed to classifier(or regressor) constructor when instance is created. 
This parameter contains dictionary with input features data mapping.

Header dictionary example:

`header_dict = {0: ('n', 'X', 'x0'), 1: ('n', 'Y', 'y0'), 2: ('n', 'Z', 'z0')}`

Header dictionary format:

* keys are indexes of columns in input numpy array X (which would be passed to fit method). Attention! amount of keys should be same as number of columns in input feature-array X. Also they should be in range 0 - t-1 if there are t columns in input data X.
values of dictionary are triples:
* feature type - 'n' for numerical feature, 'с' for categorical.
* feature namespace.
* feature name. If feature type is numerical in header_dict, this name would be used for generating vw line (for example, ('n', 'Y', 'y0') would be converted to : … |Y y0:{float_val} … )

### Examples


**Example 1 - simple linear regression**

```python
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
```

**Example 2 - logistic regression with predict_proba and quadratic features**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
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

    clf = LogisticRegressionBinary(passes=50, learning_rate=0.5, header_dict=header_dict,  quadratic='XY YZ')

    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)

    print(log_loss(y_test,preds))
```

**Example 3 - Multiclass logistic regression with quadratic features and GridSearch**

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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

    gs = GridSearchCV(clf, params, scoring='neg_log_loss', fit_params={'sample_weight': w_train}, n_jobs=4)

    gs.fit(X_train, y_train)

    print('best logloss is {}'.format(gs.best_score_))
    print('best params are {}'.format(gs.best_params_))
```