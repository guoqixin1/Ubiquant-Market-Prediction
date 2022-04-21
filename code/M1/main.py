import argparse
import inspect
import sys
import csv
import time

import config_utils as c
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell


def get_data(file_dir):
    train, valid, test = [], [], []
    with open(file_dir+'/train.csv')as f:
        csv_reader = csv.reader(f)
        for items in csv_reader:
            train.append(items)
    with open(file_dir+'/dev.csv')as f:
        csv_reader = csv.reader(f)
        for items in csv_reader:
            valid.append(items)
    with open(file_dir+'test.csv')as f:
        csv_reader = csv.reader(f)
        for items in csv_reader:
            test.append(items)
    return train, valid, test


def seq_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.float32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.float32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


class StockLSTM(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)]
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])

        self._output = output = tf.nn.xw_plus_b(rnn_output,
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost = cost = tf.reduce_mean(tf.square(output - tf.reshape(self._targets, [-1])))
        self._final_state = states[-1]

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def output(self):
        return self._output

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def main(config_size='small', num_epochs=10):
    def get_config(config_size):
        config_size = config_size.lower()
        if config_size == 'small':
            return c.SmallConfig()
        elif config_size == 'medium':
            return c.MediumConfig()
        elif config_size == 'large':
            return c.LargeConfig()
        else:
            raise ValueError('Unknown config size {} (small, medium, large)'.format(config_size))

    def run_epoch(session, m, data, eval_op, verbose=False):
        """Runs the model on the given data."""
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        print(epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = m.initial_state.eval()
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                         {m.input_data: x, m.targets: y, m.initial_state: state})
            costs += cost
            iters += m.num_steps

            print_interval = 20
            if verbose and epoch_size > print_interval \
                    and step % (epoch_size // print_interval) == print_interval:
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters,
                                                          iters * m.batch_size / (time.time() - start_time)))
        return costs / (iters if iters > 0 else 1)

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config(config_size)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = StockLSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = StockLSTM(is_training=False, config=config)

        tf.initialize_all_variables().run()

        train_data, valid_data, test_data = get_data()

        for epoch in range(num_epochs):
            lr_decay = config.lr_decay ** max(epoch - num_epochs, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            cur_lr = session.run(m.lr)

            mse = run_epoch(session, m, train_data, m.train_op, verbose=True)
            vmse = run_epoch(session, mtest, valid_data, tf.no_op())
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - test mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        tmse = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test mse: %.3f" % tmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name, arg_type in zip(ma.args[-len(ma.defaults):], [type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
