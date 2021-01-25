from __future__ import absolute_import
import sklearn.base
import numpy as np
from led_wabbit.led_vw_clf_exceptions import HeaderIsInvalid, NotEnoughYLabelsInYForOAA
import os

from led_wabbit.led_vw import VW


class _VW(sklearn.base.BaseEstimator):
    """scikit-learn interface for Vowpal Wabbit
    """

    def __init__(self,
                 logger=None,
                 vw='vw',
                 model_prefix='model',
                 name=None,
                 bits=None,
                 loss=None,
                 passes=10,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 decay_learning_rate=None,
                 initial_t=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 nn=None,
                 header_dict=None,
                 quiet=True,
                 type=None
                 ):
        self.header_dict=header_dict
        self.logger = logger
        self.vw = vw

        self.model_prefix = model_prefix

        self.name = name
        self.bits = bits
        self.loss = loss
        self.passes = passes
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.quadratic = quadratic
        self.audit = audit
        self.power_t = power_t
        self.adaptive = adaptive
        self.decay_learning_rate = decay_learning_rate
        self.initial_t = initial_t
        self.total = total
        self.node = node
        self.unique_id = unique_id
        self.span_server = span_server
        self.bfgs = bfgs
        self.oaa = oaa
        self.old_model = old_model
        self.incremental = incremental
        self.mem = mem
        self.nn = nn
        self.type = type
        self.quiet = quiet

        if self.header_dict:
            if not self._header_is_valid(self.header_dict):
                raise HeaderIsInvalid
            self.header = self._create_header(self.header_dict)
        else:
            self.header = None

        self.working_directory = os.getcwd()

    def _create_header(self, header_dict):
        header_str = ''
        old_capital = ''

        all_items = list(header_dict.items())
        all_items.sort(key=lambda x:x[1][1])

        for i in range(len(all_items)):
            cur_item = all_items[i]
            if old_capital != cur_item[1][1]:
                old_capital = cur_item[1][1]
                header_str += ' |'+old_capital
            if cur_item[1][0] == 'n':
                header_str += ' '+cur_item[1][2]+':{'+str(cur_item[0])+'}'
            else:
                header_str += ' '+'{'+str(cur_item[0])+'}'
        return header_str

    def _header_is_valid(self, header_dict):
        letters_dict = dict()
        for i in range(len(header_dict.keys())):
            if i not in header_dict:
                return False
            if len(header_dict[i]) != 3:
                return False
            if header_dict[i][1] not in letters_dict:
                letters_dict[header_dict[i][1]] = [header_dict[i][2]]
            else:
                letters_dict[header_dict[i][1]].append(header_dict[i][2])
        for letter in letters_dict.keys():
            if len(letters_dict[letter]) != len(set(letters_dict[letter])):
                return False
        return True

    def fit(self, X, y, sample_weight=None):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: numpy array of features (or tuple or iterable)
            input features
        y: [int or float]
            output labels
        """
        examples = self._as_vw_strings(X, y, sample_weight)

        # initialize model
        self.vw_ = VW(
            logger=self.logger,
            vw=self.vw,
            model=self.model_prefix,
            name=self.name,
            bits=self.bits,
            loss=self.loss,
            passes=self.passes,
            log_stderr_to_file=self.log_stderr_to_file,
            silent=self.silent,
            l1=self.l1,
            l2=self.l2,
            learning_rate=self.learning_rate,
            quadratic=self.quadratic,
            audit=self.audit,
            power_t=self.power_t,
            adaptive=self.adaptive,
            decay_learning_rate=self.decay_learning_rate,
            initial_t=self.initial_t,
            total=self.total,
            node=self.node,
            unique_id=self.unique_id,
            span_server=self.span_server,
            bfgs=self.bfgs,
            oaa=self.oaa,
            old_model=self.old_model,
            incremental=self.incremental,
            mem=self.mem,
            quiet=self.quiet,
            nn=self.nn
        )

        if self.oaa:
            if len(self.label_dict.keys())<1:
                all_labels = []
                for y_val in y:
                    if y_val not in all_labels:
                        all_labels.append(y_val)
                        if len(all_labels) == self.oaa:
                            break

                if len(all_labels) < self.oaa:
                    raise NotEnoughYLabelsInYForOAA()

                lexicographic_list = [(l,str(l)) for l in all_labels]

                lexicographic_list.sort(key=lambda x:x[1])

                self.label_dict = {l[0]: str(i+1) for i, l in enumerate(lexicographic_list)}
                self.invert_label_dict = {i+1:l[0] for i, l in enumerate(lexicographic_list)}

        # add examples to model
        with self.vw_.training():
            for instance in examples:
                self.vw_.push_instance(instance)

        # learning done after "with" statement
        return self

    def iterate_over_vw_strings(self, X, y=None, sample_weight=None):
        for e in self._as_vw_strings(X, y, sample_weight):
            yield e

    def predict(self, X, raw=False):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: numpy array of features (or tuple or iterable)
            input features
        """
        examples = self._as_vw_strings(X)

        # add test examples to model
        with self.vw_.predicting(raw):
            for instance in examples:
                self.vw_.push_instance(instance)

        # read out predictions
        predictions = np.asarray(list(self.vw_.read_predictions_(raw)))

        return predictions

    def _as_vw_string(self, x, y=None, weight=None):
        """Convert {feature: value} to something _VW understands

        Parameters
        ----------
        x : {<feature>: <value>}
        y : int or float
        """
        result = str(y)
        x = " ".join(["%s:%f" % (key, value) for (key, value) in x.items()])
        if weight is not None:
            return result + " " + str(weight) + " | " + x
        return result + " | " + x

    def _as_vw_strings(self, X, y=None, sample_weight=None,labels=None):
        n_samples = len(X)  # np.shape(X)[0]
        if y is None:
            if labels is not None:
                for i in range(n_samples):
                    yield self._as_vw_string(X[i], y, label=labels[i])
            else:
                for i in range(n_samples):
                    yield self._as_vw_string(X[i], y)
        else:
            for i in range(n_samples):
                if sample_weight:
                    yield self._as_vw_string(X[i], y[i], sample_weight[i])
                else:
                    yield self._as_vw_string(X[i], y[i])


class LinearReg(sklearn.base.RegressorMixin, _VW):
    def __init__(self,
                 logger=None,
                 vw='vw',
                 model_prefix=None,
                 name=None,
                 bits=None,
                 passes=None,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 decay_learning_rate=None,
                 initial_t=None,
                 old_model=None,
                 incremental=False,
                 header_dict=None,
                 quiet=True,
                 **kwargs):

        super(LinearReg, self).__init__(logger=logger,
                                        vw=vw,
                                        model_prefix=model_prefix,
                                        name=name,
                                        bits=bits,
                                        passes=passes,
                                        log_stderr_to_file=log_stderr_to_file,
                                        silent=silent,
                                        l1=l1,
                                        l2=l2,
                                        learning_rate=learning_rate,
                                        quadratic=quadratic,
                                        audit=audit,
                                        power_t=power_t,
                                        adaptive=adaptive,
                                        decay_learning_rate=decay_learning_rate,
                                        initial_t=initial_t,
                                        old_model=old_model,
                                        incremental=incremental,
                                        loss='squared',
                                        header_dict=header_dict,
                                        quiet=quiet,
                                        **kwargs)
        self.working_directory = os.getcwd()

    def predict(self, X):
        result = super(LinearReg, self).predict(X)
        return result

    def _as_vw_string(self, x, y=None,label=None):
        vw_string = self.header.format(*x)
        if label is not None:
            vw_string = str(label)+vw_string

        if y is not None:
            vw_string = str(y) + vw_string

        return vw_string


class LogRegBinary(sklearn.base.ClassifierMixin, _VW):
    def __init__(self,
                 logger=None,
                 vw='vw',
                 model_prefix=None,
                 name=None,
                 bits=None,
                 passes=None,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 decay_learning_rate=None,
                 initial_t=None,
                 old_model=None,
                 incremental=False,
                 header_dict=None,
                 quiet=True,
                 **kwargs):

        super(LogRegBinary, self).__init__(logger=logger,
                                           vw=vw,
                                           model_prefix=model_prefix,
                                           name=name,
                                           bits=bits,
                                           passes=passes,
                                           log_stderr_to_file=log_stderr_to_file,
                                           silent=silent,
                                           l1=l1,
                                           l2=l2,
                                           learning_rate=learning_rate,
                                           quadratic=quadratic,
                                           audit=audit,
                                           power_t=power_t,
                                           adaptive=adaptive,
                                           decay_learning_rate=decay_learning_rate,
                                           initial_t=initial_t,
                                           old_model=old_model,
                                           incremental=incremental,
                                           loss='logistic',
                                           header_dict=header_dict,
                                           quiet=quiet,
                                           **kwargs)
        self.working_directory = os.getcwd()

    def predict_raw(self, X):
        return super(LogRegBinary, self).predict(X)

    def predict_proba(self, X):
        """Predict probability of binary class for Vowpal Wabbit

        Parameters
        ----------
        X: numpy array of features
            input features
        """

        predictions_positive = 1 / (1 + np.exp(-self.predict_raw(X)))
        predictions_negative = 1 - predictions_positive
        return np.vstack((predictions_negative, predictions_positive)).transpose()

    def predict(self, X):
        return 1 / (1 + np.exp(-self.predict_raw(X)))
        
        result = self.predict_raw(X)
        for i in range(len(result)):
            result[i] = 1 if result[i] > 0 else 0

        return result

    def _as_vw_string(self, x, y=None):
        vw_string = self.header.format(*x)
        if not y is None:
            if y>0.99:
                vw_string = '1 '+vw_string
            else:
                vw_string = '-1 '+vw_string
        return vw_string


class MulticlassOAA(sklearn.base.ClassifierMixin, _VW):
    def __init__(self,
                 logger=None,
                 vw='vw',
                 model_prefix=None,
                 name=None,
                 bits=None,
                 passes=None,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 decay_learning_rate=None,
                 initial_t=None,
                 old_model=None,
                 incremental=False,
                 header_dict=None,
                 oaa=None,
                 **kwargs):

        super(MulticlassOAA, self).__init__(logger=logger,
                                            vw=vw,
                                            model_prefix=model_prefix,
                                            name=name,
                                            bits=bits,
                                            passes=passes,
                                            log_stderr_to_file=log_stderr_to_file,
                                            silent=silent,
                                            l1=l1,
                                            l2=l2,
                                            learning_rate=learning_rate,
                                            quadratic=quadratic,
                                            audit=audit,
                                            power_t=power_t,
                                            adaptive=adaptive,
                                            decay_learning_rate=decay_learning_rate,
                                            initial_t=initial_t,
                                            old_model=old_model,
                                            incremental=incremental,
                                            header_dict=header_dict,
                                            oaa=oaa,
                                            **kwargs)

        self.working_directory = os.getcwd()

        self.label_dict=dict()
        self.invert_label_dict = dict()
        self._cur_label_index = 1

    def predict_proba(self, X):
        result = super(MulticlassOAA, self).predict(X, raw=True)
        returned_result = np.zeros((len(result),len(self.label_dict.keys())))
        # ugly here, max 99 classes in vw
        if len(self.label_dict.keys()) < 10:
            # this must work little faster
            for i,res in enumerate(result):
                for j, res_str in enumerate(res.split(' ')):
                    returned_result[i, j] = float(res_str[2:])
        else:
            for i,res in enumerate(result):
                for j, res_str in enumerate(res.split(' ')):
                    if j < 9:
                        returned_result[i, j] = float(res_str[2:])
                    else:
                        returned_result[i, j] = float(res_str[3:])
        return returned_result

    def predict(self, X):
        result = super(MulticlassOAA, self).predict(X)
        return [self.invert_label_dict[res] for res in result]

    def _as_vw_string(self, x, y=None, weight=None):
        vw_string = self.header.format(*x)
        if y is not None:
            if y:
                if weight is not None:
                    return self.label_dict[y] + ' ' + str(weight) + ' ' + vw_string
                return self.label_dict[y]+' '+vw_string
            else:
                return ' '+vw_string
        return vw_string
