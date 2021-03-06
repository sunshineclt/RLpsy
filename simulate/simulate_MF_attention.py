import datetime
import multiprocessing as mp
import os
import random
import shutil

import numpy as np
from psychopy import data

from Transition import Transition
from utils import utils


def simulate_MF_attention(randomized=True,
                          alpha=0.1,
                          tau=1.0,
                          repeat=50,
                          gamma=1,
                          forget=0.001,
                          alpha_attention=0.1,
                          path=None,
                          seed=None):
    if path:
        dir_path = path
    else:
        dir_path = "data/MF_forget/alpha%.1f_tau%.1f/" % (alpha, tau) + ("randomized" if randomized else "block") + "/"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    for time in range(0, repeat):

        if seed:
            np.random.seed(seed)
            random.seed(seed)
        else:
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
        random.seed(datetime.datetime.now())
        np.random.seed()

        def state_transform(end_state, state):
            if end_state == state:
                return 0
            elif end_state == state - 3:
                return 1
            else:
                return 2

        # model and parameters
        q_value = np.zeros(shape=[3, 6, 3])
        attention = np.zeros(shape=[3, 3])

        for trial in trials:
            episode += 1

            trial_start_state = trial["from"]
            trial_end_state = trial["to"]

            # Start Free Choice
            step = 0
            now_state = trial_start_state
            trial_record = []
            while now_state != trial_end_state:
                step += 1

                action = utils.random_pick([0, 1, 2],
                                           utils.softmax(q_value[trial_end_state][now_state],
                                                         tau))

                new_state = transition.step(now_state, action)
                trial_record.append([now_state, action, new_state])

                if trial_end_state == new_state:
                    target = max(21 - step, 1)
                else:
                    target = gamma * attention[trial_end_state][state_transform(trial_end_state, new_state)]
                delta = target - attention[trial_end_state][state_transform(trial_end_state, now_state)]
                attention[trial_end_state][state_transform(trial_end_state, now_state)] += alpha_attention * delta
                lr = attention[trial_end_state][state_transform(trial_end_state, now_state)] * alpha
                lr = min(lr, 1)

                if trial_end_state == new_state:
                    target = max(21 - step, 1)
                else:
                    target = gamma * np.max(q_value[trial_end_state][new_state])
                delta = target - q_value[trial_end_state][now_state, action]
                q_value[trial_end_state][now_state, action] += lr * delta

                now_state = new_state
                q_value *= (1 - forget)

            # raw_data storing
            total_reward += max(21 - step, 1)
            trials.addData("trial_data", trial_record)
            timesteps_record.append(step)

        trials.saveAsWideText(dir_path + "/" + str(time) + "_simulate.csv", delim="#")
        # print("alpha: %.1f, tau: %.1f, %s, times: %d, Total Reward: %d" %
        #       (alpha, tau,
        #        ("randomized" if randomized else "block"),
        #        time, total_reward))


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print("start time: ", time_stamp.strftime('%H:%M:%S'))
    condition = []
    for alpha_value in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for randomized_value in [True, False]:
            for tau_value in [0.1, 0.3, 0.5, 1, 10, 100]:
                condition.append((randomized_value, alpha_value, tau_value))
    time_stamp = datetime.datetime.now()
    print("start execution time: ", time_stamp.strftime('%H:%M:%S'))
    pool = mp.Pool()
    pool.starmap(simulate_MF, condition)
    # simulate_MF(False, 0.3, 0.3)
    pool.close()
    print("main process finished")
    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
