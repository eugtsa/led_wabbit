from __future__ import absolute_import
import os
import sys
import subprocess
import shlex
import tempfile
import random
import sklearn.base
import numpy as np

from contextlib import contextmanager
from led_wabbit.models_exceptions import HeaderIsInvalid, NotEnoughYLabelsInYForOAA


def _safe_remove(f):
    try:
        os.remove(f)
    except OSError:
        pass


class VW:
    def __init__(self,
                 logger=None,
                 vw='vw',
                 model=None,
                 name=None,
                 bits=None,
                 loss=None,
                 passes=None,
                 log_stderr_to_file=True,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 #  working_dir=None,
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
                 quiet=True,
                 **kwargs):

        if logger is None:
            self.log = lambda x: None

        self.node = node
        self.total = total
        self.unique_id = unique_id
        self.span_server = span_server
        if self.node is not None:
            assert self.total is not None
            assert self.unique_id is not None
            assert self.span_server is not None

        if name is None:
            hash = random.getrandbits(128)
            self.handle = '%s_%016x' % (model, hash)
        else:
            self.handle = '%s.%s' % (model, name)

        if self.node is not None:
            self.handle = "%s.%d" % (self.handle, self.node)

        if old_model is None:
            self.filename = '%s.model' % self.handle
            self.incremental = False
        else:
            self.filename = old_model
            self.incremental = True

        self.incremental = incremental
        self.filename = '%s.model' % self.handle

        self.name = name
        self.bits = bits
        self.loss = loss
        self.vw = vw
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.passes = passes
        self.quadratic = quadratic
        self.power_t = power_t
        self.adaptive = adaptive
        self.decay_learning_rate = decay_learning_rate
        self.audit = audit
        self.initial_t = initial_t
        self.oaa = oaa
        self.bfgs = bfgs
        self.mem = mem
        self.nn = nn
        self.quiet = quiet

        self.working_directory = os.getcwd()

    def vw_base_command(self, base, test=False, raw=False):
        l = base
        if self.bits                is not None: l.append('-b %d' % self.bits)
        if self.learning_rate       is not None: l.append('--learning_rate=%f' % self.learning_rate)
        if self.l1                  is not None: l.append('--l1=%f' % self.l1)
        if self.l2                  is not None: l.append('--l2=%f' % self.l2)
        if self.initial_t           is not None: l.append('--initial_t=%f' % self.initial_t)
        if self.quadratic           is not None:
            for quad in self.quadratic.split(' '):
                l.append('-q %s' % quad)
        if self.power_t             is not None: l.append('--power_t=%f' % self.power_t)
        if self.loss                is not None: l.append('--loss_function=%s' % self.loss)
        if self.decay_learning_rate is not None: l.append('--decay_learning_rate=%f' % self.decay_learning_rate)
        if not test and self.oaa    is not None: l.append('--oaa=%d' % self.oaa)
        if raw and self.oaa         is not None: l.append('--link logistic')
        if self.unique_id           is not None: l.append('--unique_id=%d' % self.unique_id)
        if self.total               is not None: l.append('--total=%d' % self.total)
        if self.node                is not None: l.append('--node=%d' % self.node)
        if self.span_server         is not None: l.append('--span_server=%s' % self.span_server)
        if self.mem                 is not None: l.append('--mem=%d' % self.mem)
        if self.audit:                           l.append('--audit')
        if self.bfgs:                            l.append('--bfgs')
        if self.adaptive:                        l.append('--adaptive')
        if self.nn                  is not None: l.append('--nn=%d' % self.nn)
        if self.quiet:                           l.append('--quiet')
        return ' '.join(l)

    def vw_train_command(self, cache_file, model_file):
        if os.path.exists(model_file) and self.incremental:
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -i %s -f %s' \
                    % (self.passes, cache_file, model_file, model_file)
        else:
            self.log('No existing model file or not options.incremental')
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -f %s' \
                    % (self.passes, cache_file, model_file)

    def vw_test_command(self, model_file, prediction_file):
        return self.vw_base_command([self.vw], test=True) + ' -t -i %s -p %s' % (model_file, prediction_file)

    def vw_test_raw_command(self, model_file, prediction_file):
        return self.vw_base_command([self.vw], test=True, raw=True) + ' -t -i %s -r %s' % (model_file, prediction_file)

    def vw_test_command_library(self, model_file):
        return self.vw_base_command([]) + ' -t -i %s' % (model_file)

    @contextmanager
    def training(self):
        self.start_training()
        yield
        self.close_process()
        os.remove(self.cache_file)

    @contextmanager
    def predicting(self, raw=False):
        self.start_predicting(raw)
        yield
        self.close_process()

    def start_training(self):
        cache_file = self.get_cache_file()
        self.cache_file = cache_file
        model_file = self.get_model_file()

        # Remove the old cache and model files
        if not self.incremental:
            _safe_remove(cache_file)
            _safe_remove(model_file)

        # Run the actual training
        self.vw_process = self.make_subprocess(self.vw_train_command(cache_file, model_file))

        # set the instance pusher
        self.push_instance = self.push_instance_stdin

    def close_process(self):
        # Close the process
        assert self.vw_process
        self.vw_process.stdin.flush()
        self.vw_process.stdin.close()
        if self.vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" % \
                (self.vw_process.pid, self.vw_process.command, self.vw_process.returncode))

        if self.quiet:
            if self.current_stdout:
                with open(self.current_stdout, 'r') as f:
                    all_lines_stdout = f.readlines()
                    len_file_stdout = len(all_lines_stdout)
                if len_file_stdout == 1:
                    os.remove(self.current_stdout)

            if self.current_stderr:
                with open(self.current_stderr, 'r') as f:
                    all_lines_stderr = f.readlines()
                    len_file_stderr = len(all_lines_stderr)
                if len_file_stderr == 1:
                    os.remove(self.current_stderr)

    def push_instance_stdin(self, instance):
        self.vw_process.stdin.write(('%s\n' % instance))#.encode('utf8'))

    def start_predicting(self, raw=False):
        model_file = self.get_model_file()
        # Be sure that the prediction file has a unique filename, since many processes may try to
        # make predictions using the same model at the same time
        _, prediction_file = tempfile.mkstemp(dir='.', prefix=self.get_prediction_file())
        os.close(_)

        if raw:
            self.vw_process = self.make_subprocess(self.vw_test_raw_command(model_file, prediction_file))
        else:
            self.vw_process = self.make_subprocess(self.vw_test_command(model_file, prediction_file))
        self.prediction_file = prediction_file
        self.push_instance = self.push_instance_stdin

    def parse_prediction(self, p):
        if self.oaa:
            return int(p.split()[0])
        else:
            return float(p.split()[0])

    def read_predictions_(self, raw=False):
        if raw:
            for x in open(self.prediction_file):
                yield x
        else:
            for x in open(self.prediction_file):
                yield self.parse_prediction(x)
        # clean up the prediction file
        os.remove(self.prediction_file)

    def predict_push_instance(self, instance):
        return self.parse_prediction(self.vw_process.learn(('%s\n' % instance).encode('utf8')))

    def make_subprocess(self, command):
        if not self.log_stderr_to_file:
            stdout = open('/dev/null', 'w')
            stderr = open('/dev/null', 'w') if self.silent else sys.stderr
            self.current_stdout = None
            self.current_stderr = None
        else:
            # Save the output of vw to file for debugging purposes
            try:
                os.mkdir('./temp-vw/')
            except FileExistsError as e:
                pass
            log_file_base = tempfile.mktemp(dir=self.working_directory, prefix="./temp-vw/temp-vw-")

            self.current_stdout = log_file_base + '.out'
            self.current_stderr = log_file_base + '.err'

            stdout = open(self.current_stdout, 'w')
            stderr = open(self.current_stderr, 'w')
            stdout.write(command + '\n')
            stderr.write(command + '\n')

        self.log('Running command: "%s"' % str(command))
        result = subprocess.Popen(shlex.split(str(command)), stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, close_fds=True, universal_newlines=True)
        result.command = command
        return result

    def get_current_stdout(self):
        return open(self.current_stdout)

    def get_current_stderr(self):
        return open(self.current_stderr)

    def get_model_file(self):
        temp_dir_models = 'temp-vw-models-cache'
        try:
            os.mkdir(temp_dir_models)
        except FileExistsError as e:
            pass

        return os.path.join(self.working_directory, temp_dir_models, self.filename)

    def get_cache_file(self):
        temp_dir_models = 'temp-vw-models-cache'
        try:
            os.mkdir(temp_dir_models)
        except FileExistsError as e:
            pass

        _,t_file = tempfile.mkstemp(dir=os.path.join(self.working_directory,temp_dir_models),
                                    prefix='%s.cache.' % (self.handle))
        os.close(_)

        return t_file

    def get_prediction_file(self):
        return os.path.join(self.working_directory, '%s.prediction' % (self.handle))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_process()
        _safe_remove(self.get_model_file())
        _safe_remove(self.get_cache_file())


class _VW(sklearn.base.BaseEstimator):
    def __init__(self,
                 logger=None,
                 vw='vw',
                 model_prefix='model',
                 name=None,
                 bits=None,
                 loss=None,
                 passes=1,
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

    def iterate_over_vw_strings(self, X, y, sample_weight=None):
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

    def _as_vw_strings(self, X, y=None, sample_weight=None):
        n_samples = len(X)  # np.shape(X)[0]
        if y is None:
            for i in range(n_samples):
                yield self._as_vw_string(X[i], y)
        else:
            for i in range(n_samples):
                if sample_weight:
                    yield self._as_vw_string(X[i], y[i], sample_weight[i])
                else:
                    yield self._as_vw_string(X[i], y[i])


class LinearRegression(sklearn.base.RegressorMixin, _VW):
    def __init__(self,
                 header_dict=None,
                 bits=None,
                 passes=None,
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
                 quiet=True,
                 logger=None,
                 name=None,
                 log_stderr_to_file=False,
                 vw='vw',
                 model_prefix=None,
                 **kwargs):
        '''
        Linear Regression provided by vw
        :param header_dict: dictionary with input features data mapping, read more in documentation
        :param bits: 2**bits - number of buckets to use in hashing trick (corresponds to -b  [--bit_precision] vw argument)
        :param passes: number of training passes through data (corresponds to --passes vw argument)
        :param silent: used only if log_stderr_to_file set to False. True value - don't write stderr to sys.stderr
        :param l1: l_1 lambda (L1 regularization) (corresponds to --l1 vw argument)
        :param l2: l_2 lambda (L2 regularization) (corresponds to --l2 vw argument)
        :param learning_rate: set (initial) learning rate (corresponds to -l [ --learning_rate ] vw argument)
        :param quadratic: create and use quadratic features (corresponds to -q [ --quadratic ] vw argument)
        :param audit:
        :param power_t: t power value (corresponds to --power_t vw argument)
        :param adaptive: use adaptive, individual learning rates (corresponds to --adaptive vw argument)
        :param decay_learning_rate: set decay factor for learning_rate between passes(corresponds to --decay_learning_rate vw argument)
        :param initial_t: initial t value (corresponds to --initial_t vw argument)
        :param old_model:
        :param incremental:
        :param quiet:
        :param logger:
        :param name:
        :param log_stderr_to_file:  must be True if using led_wabbit in ipython notebook. Save vw process output's to file (by default, './temp-vw/temp-vw-*' with extentions .err and .out)
        :param vw:
        :param model_prefix:
        :param kwargs:
        '''
        super(LinearRegression, self).__init__(logger=logger,
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
        result = super(LinearRegression, self).predict(X)
        return result

    def _as_vw_string(self, x, y=None):
        vw_string = self.header.format(*x)
        if y is not None:
            vw_string = str(y) + vw_string

        return vw_string


class LogisticRegressionBinary(sklearn.base.ClassifierMixin, _VW):
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
        '''
        
        :param logger: 
        :param vw: 
        :param model_prefix: 
        :param name: 
        :param bits: 
        :param passes: 
        :param log_stderr_to_file: 
        :param silent: 
        :param l1: 
        :param l2: 
        :param learning_rate: 
        :param quadratic: 
        :param audit: 
        :param power_t: 
        :param adaptive: 
        :param decay_learning_rate: 
        :param initial_t: 
        :param old_model: 
        :param incremental: 
        :param header_dict: 
        :param quiet: 
        :param kwargs: 
        '''

        super(LogisticRegressionBinary, self).__init__(logger=logger,
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
        return super(LogisticRegressionBinary, self).predict(X)

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
