import csv
import datetime
import json
import os

import numpy as np
from scipy import optimize

from utils import utils


def MB_lld(params):
    eta = params[0]
    tau = params[1]
    gamma = params[2]
    forget = params[3]
    forward_planning = 2

    # model and parameters
    trans_prob = np.zeros(shape=[6, 3, 6]) + 1 / 6

    lld = 0

    for episode in range(144):
        trial_end_state = trials_data[episode][-1][2]

        step = 0
        r = np.zeros(shape=[6]) - 1
        r[trial_end_state] = 20
        for transit in trials_data[episode]:
            now_state = transit[0]
            action_chosen = transit[1]
            step += 1

            q_value = np.zeros(shape=[6, 3])
            for iter_times in range(forward_planning):
                new_q_value = np.zeros(shape=[6, 3])
                for state in range(6):
                    for action in range(3):
                        new_q_value[state, action] = np.sum(
                            trans_prob[state, action] * (r + gamma * np.max(q_value, axis=1)))
                        # for state_1 in range(6):
                        #     new_new_q_value[state, action] += trans_prob[state, action, state_1] * \
                        #                                   (r[state_1] + gamma * np.max(q_value[state_1]))
                q_value = new_q_value

            likelihood = utils.softmax(np.array([q_value[now_state, 0],
                                                 q_value[now_state, 1],
                                                 q_value[now_state, 2]]),
                                       tau)[action_chosen]
            if likelihood < 1e-200:
                lld += 460.517
            else:
                lld -= np.log(likelihood)

            new_state = transit[2]
            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - eta)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + eta * (1 - trans_prob_to_new_state)
            # trans_prob = (1/6 - trans_prob) * forget + trans_prob

    # print(lld)
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

        result = optimize.minimize(MB_lld, np.array([0.7, 1, 0.9, 0.001]), bounds=[(0, 1), (1e-5, 100), (0, 1), (1e-100, 1)])
        print("For participant %d, best fit lld is %.3f, eta=%.2f, tau=%.2f, gamma=%.2f, forget=%.4f" %
              (participant_id, result.fun, result.x[0], result.x[1], result.x[2], result.x[3]))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
