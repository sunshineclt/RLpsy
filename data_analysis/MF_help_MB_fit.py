import csv
import datetime
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import optimize

from utils import utils


def MF_help_MB_lld(params):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    eta = params[3]
    forget_MF = params[4]
    forget_MB = params[5]
    I = params[6]
    k = params[7]
    forward_planning = 2

    # model and parameters
    q_value_MF = np.zeros(shape=[3, 6, 3])
    trans_prob = np.zeros(shape=[6, 3, 6]) + 1 / 6

    lld = 0
    global_step = 0
    for episode in range(144):
        trial_end_state = trials_data[episode][-1][2]

        step = 0
        r = np.zeros(shape=[6]) - 1
        r[trial_end_state] = 20
        for transit in trials_data[episode]:
            now_state = transit[0]
            action_chosen = transit[1]
            step += 1
            global_step += 1

            successor = trans_prob[now_state].copy()  # shape = 3, 6
            it = trans_prob[now_state].copy()  # shape = 3, 6
            for iter_times in range(forward_planning - 1):
                new_it = np.zeros(shape=[3, 6])
                for action in range(3):
                    for old_state in range(6):
                        old_action = np.argmax(q_value_MF[trial_end_state][old_state])
                        new_it[action] += it[action, old_state] * trans_prob[old_state, old_action]  # vector calculation
                assert np.all(np.sum(new_it, axis=1) - 1 < 1e-1)
                it = new_it
                successor += it * (gamma ** iter_times)
            q_value_MB = successor.dot(r)

            weight_MB = I * (-np.exp(-k * global_step) + 1)
            hybrid_q_value = q_value_MB * weight_MB + (1 - weight_MB) * q_value_MF[trial_end_state][now_state]

            likelihood = utils.softmax(hybrid_q_value,
                                       tau)[action_chosen]

            if likelihood < 1e-200:
                lld += 1000000
            else:
                lld -= np.log(likelihood)

            new_state = transit[2]
            if trial_end_state == new_state:
                target = max(21 - step, 1)
            else:
                target = gamma * np.max(q_value_MF[trial_end_state][new_state])
            delta = target - q_value_MF[trial_end_state][now_state, action_chosen]
            q_value_MF[trial_end_state][now_state, action_chosen] += alpha * delta
            q_value_MF *= (1 - forget_MF)

            # if utils.softmax(q_value_MF[trial_end_state][last_state], tau)[last_action] > 0.7:
                # lr = eta * utils.softmax(q_value_MF[trial_end_state][last_state], tau)[last_action]
            lr = eta
            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - lr)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + lr * (
                        1 - trans_prob_to_new_state)
            trans_prob = (1 / 6 - trans_prob) * forget_MB + trans_prob

    return lld


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print("start time: ", time_stamp.strftime('%H:%M:%S'))

    hybrid_fit_result = open("hybrid_result.csv", "w")
    fieldnames = ["participant",
                  "nlld",
                  "alpha",
                  "tau",
                  "gamma",
                  "eta",
                  "forget_MF",
                  "forget_MB"]
    writer = csv.DictWriter(hybrid_fit_result, fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))

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


        def minimize(args):
            f, x, bound = args
            res = optimize.minimize(f, x, bounds=bound)
            return res


        pool = mp.Pool(12)
        initial_value = [np.array([0.3, 1, 0.9, 0.3, 0.001, 0.001, 1, 0.01]),
                         np.array([0.3, 1, 0.9, 0.5, 0.001, 0.001, 1, 0.01]),
                         np.array([0.3, 1, 0.9, 0.7, 0.001, 0.001, 1, 0.01]),
                         np.array([0.5, 1, 0.9, 0.3, 0.001, 0.001, 1, 0.01]),
                         np.array([0.5, 1, 0.9, 0.5, 0.001, 0.001, 1, 0.01]),
                         np.array([0.5, 1, 0.9, 0.7, 0.001, 0.001, 1, 0.01]),
                         np.array([0.7, 1, 0.9, 0.3, 0.001, 0.001, 1, 0.01]),
                         np.array([0.7, 1, 0.9, 0.5, 0.001, 0.001, 1, 0.01]),
                         np.array([0.7, 1, 0.9, 0.7, 0.001, 0.001, 1, 0.01])]

        bounds = [(0, 1), (1e-5, 100), (0, 1), (0, 1), (0, 0.01), (0, 0.01), (0, 1), (0, 10)]
        condition = [(MF_help_MB_lld, initial, bounds) for initial in initial_value]

        # MFMB_lld([0.99, 0.11, 0.03, 1.00, 0.001, 0.001])
        # break

        result = pool.map(minimize, condition)
        min_fun = 1e50
        min_fun_x = 0
        for i in range(len(initial_value)):
            if result[i].fun < min_fun:
                min_fun = result[i].fun
                min_fun_x = result[i].x
        print(
            "For participant %d, best fit lld is %.3f, alpha=%.2f, tau=%.2f, gamma=%.2f, eta=%.2f, forget_MF=%.5f, forget_MB=%.5f, I=%.2f, k=%.2f" %
            (participant_id, min_fun, *min_fun_x))
        writer.writerow(dict(zip(fieldnames, [participant_id, min_fun, *min_fun_x])))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
