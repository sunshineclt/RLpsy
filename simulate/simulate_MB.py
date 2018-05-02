import numpy as np
from psychopy import data

from Transition import Transition
from utils import utils
import random
import os


def simulate_MB(randomized=True,
                repeat=36,
                gamma=0.9,
                eta=0.1,
                tau=1,
                forward_planning=6):
    for time in range(0, repeat):

        np.random.seed(time)
        random.seed(time)
        np.random.shuffle([1, 2, 3, 4, 5, 6])

        task_order = []
        number_of_trials = 144
        if randomized:
            # randomized
            for i in range(0, number_of_trials):
                task_order.append({"from": i % 3, "to": (i + 1) % 3})
            np.random.shuffle(task_order)
        else:
            # block
            for i in range(0, 36):
                task_order.append({"from": 0, "to": 1})
            for i in range(0, 36):
                task_order.append({"from": 1, "to": 2})
            for i in range(0, 36):
                task_order.append({"from": 2, "to": 0})
            temp_task = []
            for i in range(0, 36):
                temp_task.append({"from": i % 3, "to": (i + 1) % 3})
            np.random.shuffle(temp_task)
            task_order.extend(temp_task)

        trials = data.TrialHandler(task_order, nReps=1, method="sequential", originPath=".")

        transition = Transition()
        timesteps_record = []
        episode = 0
        total_reward = 0

        # model and parameters
        trans_prob = np.zeros(shape=[6, 3, 6]) + 1 / 3

        for trial in trials:
            episode += 1

            trial_start_state = trial["from"]
            trial_end_state = trial["to"]

            # Start Free Choice
            step = 0
            now_state = trial_start_state
            trial_record = []
            r = np.zeros(shape=[6]) - 1
            r[trial_end_state] = 20
            while now_state != trial_end_state:
                # print("now: ", now_state)
                step += 1

                q_value = np.zeros(shape=[6, 3])
                for iter_times in range(forward_planning):
                    new_q_value = np.zeros(shape=[6, 3])
                    for state in range(6):
                        for action in range(3):
                            new_q_value[state, action] = np.sum(trans_prob[state, action] * (r + gamma * np.max(q_value, axis=1)))
                            # for state_1 in range(6):
                            #     new_q_value[state, action] += trans_prob[state, action, state_1] * (
                            #                 r[state_1] + gamma * np.max(q_value[state_1]))
                    q_value = new_q_value
                # print(np.max(q_value))

                action = utils.random_pick([0, 1, 2],
                                           utils.softmax([q_value[now_state, 0],
                                                          q_value[now_state, 1],
                                                          q_value[now_state, 2]], tau))

                new_state = transition.step(now_state, action)
                trial_record.append([now_state, action, new_state])

                trans_prob_to_new_state = trans_prob[now_state, action, new_state]
                trans_prob[now_state, action] *= (1 - eta)
                trans_prob[now_state, action, new_state] = trans_prob_to_new_state + eta * (1 - trans_prob_to_new_state)

                now_state = new_state

            # raw_data storing
            total_reward += max(21 - step, 1)
            trials.addData("trial_data", trial_record)
            timesteps_record.append(step)

        dir_path = "data/MB/eta%.1f_tau%.1f_forward%d/" % (eta, tau, forward_planning) + ("randomized" if randomized else "block")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        trials.saveAsWideText(dir_path + "/" + str(time) + "_simulate.csv", delim="#")
        print("eta: %.1f, tau: %.1f, forward: %d, times: %d, Total Reward: %d" % (eta, tau, forward_planning, time, total_reward))


if __name__ == "__main__":
    for eta_value in [i / 10 for i in range(1, 11)]:
        for randomized_value in [True, False]:
            for tau_value in [0.1, 0.5, 1, 5, 10, 100]:
                for forward_planning_value in [2, 3, 4, 5, 6, 7]:
                    simulate_MB(randomized=randomized_value,
                                eta=eta_value,
                                tau=tau_value,
                                forward_planning=forward_planning_value)


