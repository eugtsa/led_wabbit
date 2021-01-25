from contextlib import contextmanager
import os
import sys
import subprocess
import shlex
import tempfile
import random


def safe_remove(f):
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
            safe_remove(cache_file)
            safe_remove(model_file)

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
        safe_remove(self.get_model_file())
        safe_remove(self.get_cache_file())
