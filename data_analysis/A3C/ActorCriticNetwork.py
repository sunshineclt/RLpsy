import tensorflow as tf
import numpy as np


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype, partition_info):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ActorCriticNetwork:
    def __init__(self, scope, trainer, action_size):
        with tf.variable_scope(scope):
            # the first None dimension is time, not batch
            self.state = tf.placeholder(shape=[None, 6], dtype=tf.float32, name="state_input")
            self.previous_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="previous_reward")
            self.previous_action = tf.placeholder(shape=[None], dtype=tf.int32, name="previous_action")
            self.previous_action_onehot = tf.one_hot(self.previous_action, action_size, name="previous_action_onehot")
            total_input = tf.concat([self.state, self.previous_reward, self.previous_action_onehot], axis=1)
            # rnn_input is [batch, time, content], batch=1
            rnn_input = tf.expand_dims(total_input, axis=0)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_input = tf.placeholder(shape=[1, lstm_cell.state_size.c], dtype=tf.float32)
            h_input = tf.placeholder(shape=[1, lstm_cell.state_size.h], dtype=tf.float32)
            self.state_input = tf.contrib.rnn.LSTMStateTuple(c_input, h_input)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_input, initial_state=self.state_input,
                                                         sequence_length=tf.shape(self.previous_reward)[:1])
            lstm_c, lstm_h = lstm_state
            self.state_output = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_output = tf.reshape(lstm_outputs, [-1, 48])

            # output
            self.policy = tf.contrib.slim.fully_connected(rnn_output, action_size,
                                                          activation_fn=tf.nn.softmax,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          biases_initializer=None)
            self.value = tf.contrib.slim.fully_connected(rnn_output, 1,
                                                         activation_fn=None,
                                                         weights_initializer=normalized_columns_initializer(1.0),
                                                         biases_initializer=None)

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
            self.actions_onehot = tf.one_hot(self.actions, action_size)

            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name="target_v")
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global")
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
