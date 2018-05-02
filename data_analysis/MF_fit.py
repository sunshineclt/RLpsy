import csv
import datetime
import json
import os
import random
import shutil

import numpy as np
from psychopy import data
from scipy import optimize

from Transition import Transition
from utils import utils


def MF_lld(alpha=0.1,
           tau=1,
           gamma=0.9,
           randomized=True):

    # model and parameters
    q_value = np.zeros(shape=[3, 6, 3])

    lld = 0

    for episode in range(144):
        trial_end_state = trials_data[episode][-1][0]

        step = 0
        last_state = 0
        last_action = 0
        for transit in trials_data[episode]:
            now_state = transit[0]
            action = transit[1]
            step += 1

            lld -= np.log(utils.softmax(np.array([q_value[trial_end_state][now_state, 0],
                                                  q_value[trial_end_state][now_state, 1],
                                                  q_value[trial_end_state][now_state, 2]]),
                                        tau)
                          [action])

            if step != 1:
                target = 0 + gamma * q_value[trial_end_state][now_state, action]
                delta = target - q_value[trial_end_state][last_state, last_action]
                q_value[trial_end_state][last_state, last_action] += alpha * delta

            last_state = now_state
            last_action = action
            q_value *= 0.99

        target = max(21 - step, 1)
        delta = target - q_value[trial_end_state][last_state, last_state]
        q_value[trial_end_state][last_state, last_state] += alpha * delta

    return lld


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print("start time: ", time_stamp.strftime('%H:%M:%S'))

    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144

    for lists in os.listdir("data/"):
        path = os.path.join("data/", lists)
        print("Loading ", path)
        split = lists.split("_")
        participant_id = int(split[0])

        rawFile = open(path, "r")
        reader = csv.DictReader(rawFile, delimiter="#")
        trials_data = []
        length = []
        for row in reader:
            trial = row["trial_data"]
            if trial != "--":
                transformed_trial = json.loads(trial)
                trials_data.append(transformed_trial)
                length.append(len(transformed_trial))
        trials_data = trials_data[:TRIAL_LENGTH]

        result = optimize.minimize(MF_lld, np.array([0.1, 0.5, 0.9]), (participant_id % 2 == 0,))
        print("For participant %d, best fit lld is %.3f, alpha=%.2f, tau=%.2f, gamma=%.2f" %
              (participant_id, result.fun, result.x[0], result.x[1], result.x[2]))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
