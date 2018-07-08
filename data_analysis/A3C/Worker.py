import numpy as np
import scipy.signal
import tensorflow as tf

from data_analysis.A3C.ActorCriticNetwork import ActorCriticNetwork


class Worker:
    def __init__(self, env, number, trainer, model_path, global_episodes):
        self.name = "worker_" + str(number)
        self.number = number
        self.model_path = model_path
        self.action_size = 3
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        self.local_AC = ActorCriticNetwork(self.name, trainer, action_size=3)

        # Copies one set of variables to another.
        # Used to set worker network parameters to those of global network.
        def update_target_graph(from_scope, to_scope):
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
            to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

            op_holder = []
            for from_var, to_var in zip(from_vars, to_vars):
                op_holder.append(to_var.assign(from_var))
            return op_holder

        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env

        self.previous_actions = 0
        self.previous_rewards = 0
        self.rewards_plus = 0
        self.value_plus = 0

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        previous_rewards = [0] + rewards[:-1].tolist()
        previous_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        self.previous_rewards = previous_rewards
        self.previous_actions = previous_actions

        # Discounting function used to calculate discounted returns.
        def discount(x, gamma):
            return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state: np.stack(states, axis=0),
                     self.local_AC.previous_reward: np.vstack(previous_rewards),
                     self.local_AC.previous_action: previous_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_input[0]: rnn_state[0],
                     self.local_AC.state_input[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                reward = 0
                action = 0
                t = 0
                state, is_reset = self.env.reset()
                if is_reset:
                    break
                rnn_state = self.local_AC.state_init

                while not done:
                    t += 1
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_output],
                        feed_dict={
                            self.local_AC.state: [state],
                            self.local_AC.previous_reward: [[reward]],
                            self.local_AC.previous_action: [action],
                            self.local_AC.state_input[0]: rnn_state[0],
                            self.local_AC.state_input[1]: rnn_state[1]})
                    action = np.random.choice([i for i in range(0, self.action_size)], p=a_dist[0])

                    rnn_state = rnn_state_new
                    s1, reward, done = self.env.step(action, t)
                    episode_buffer.append([state, action, reward, t, done, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1
                    state = s1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 10 == 0 and episode_count != 0:
                    if episode_count % 360 == 0 and self.name == 'worker_0' and train:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Model Saved at ", episode_count)

                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    mean_value = np.mean(self.episode_mean_values[-10:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if train:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
