import os
import random
import shutil

import numpy as np
from psychopy import data

from Transition import Transition
from data_analysis.analysis_optimal import calculate_optimal


def simulate_optimal(randomized=True,
                     repeat=50):

    dir_path = "data/optimal/" + ("randomized" if randomized else "block") + "/"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    optimal_reward = 0
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
        optimal = calculate_optimal(transition)
        timesteps_record = []
        episode = 0
        total_reward = 0

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

                action = optimal[now_state, trial_end_state]

                new_state = transition.step(now_state, action)
                trial_record.append([now_state, action, new_state])
                now_state = new_state

            # raw_data storing
            total_reward += max(21 - step, 1)
            trials.addData("trial_data", trial_record)
            timesteps_record.append(step)

        trials.saveAsWideText(dir_path + "/" + str(time) + "_simulate.csv", delim="#")
        print("%s, times: %d, Total Reward: %d" %
              (("randomized" if randomized else "block"), time, total_reward))
        optimal_reward += total_reward

    print("Optimal Reward: ", optimal_reward / repeat)


if __name__ == "__main__":
    for randomized_value in [True, False]:
        simulate_optimal(randomized=randomized_value)
