import csv
import datetime
import json
import multiprocessing as mp
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

            q_value_MB = np.zeros(shape=[6, 3])
            for iter_times in range(forward_planning):
                new_q_value = np.zeros(shape=[6, 3])
                for state in range(6):
                    for action in range(3):
                        new_q_value[state, action] = np.sum(
                            trans_prob[state, action] * (r + gamma * np.max(q_value_MB, axis=1)))
                        # for state_1 in range(6):
                        #     new_new_q_value[state, action] += trans_prob[state, action, state_1] * \
                        #                                   (r[state_1] + gamma * np.max(q_value_MB[state_1]))
                q_value_MB = new_q_value

            # successor = trans_prob[now_state].copy()  # shape = 3, 6
            # it = trans_prob[now_state].copy()  # shape = 3, 6
            # for iter_times in range(forward_planning - 1):
            #     new_it = np.zeros(shape=[3, 6])
            #     for action in range(3):
            #         for old_state in range(6):
            #             old_action = np.argmax(q_value_MF[trial_end_state][old_state])
            #             new_it[action] += it[action, old_state] * trans_prob[old_state, old_action]  # vector calculation
            #     assert np.all(np.sum(new_it, axis=1) - 1 < 1e-1)
            #     it = new_it
            #     successor += it * (gamma ** iter_times)
            # q_value_MB = successor.dot(r)

            likelihood = utils.softmax(q_value_MB[now_state],
                                       tau)[action_chosen]
            if likelihood < 1e-200:
                lld += 460.517
            else:
                lld -= np.log(likelihood)

            new_state = transit[2]
            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - eta)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + eta * (1 - trans_prob_to_new_state)
            trans_prob = (1/6 - trans_prob) * forget + trans_prob

    # print(lld)
    return lld


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print("start time: ", time_stamp.strftime('%H:%M:%S'))

    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144

    MB_fit_result = open("MB_fit_result.csv", "w")
    fieldnames = ["participant",
                  "nlld",
                  "eta",
                  "tau",
                  "gamma",
                  "forget_MB"]
    writer = csv.DictWriter(MB_fit_result, fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))

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
        bounds = [(0, 1), (1e-5, 100), (0, 1), (0, 0.05), (0, 1)]
        initial_value = []
        for i in range(12):
            initial_value.append(np.array([np.random.random(),
                                           np.random.random() * 100 + 1e-5,
                                           np.random.random(),
                                           np.random.random() * 0.05,
                                           np.random.random(),
                                           np.random.random()]))

        condition = [(MB_lld, initial[:4], bounds[:4]) for initial in initial_value]

        result = pool.map(minimize, condition)
        min_fun = 1e50
        min_fun_x = 0
        for i in range(len(initial_value)):
            if result[i].fun < min_fun:
                min_fun = result[i].fun
                min_fun_x = result[i].x

        # result = optimize.minimize(MF_lld, np.array([0.1, 1, 0.9, 0.001, 0.1][:4]), bounds=[(0, 1), (1e-5, 100), (0, 1), (0, 0.05), (0, 1)][:4])
        print("For participant %d, best fit lld is %.3f, eta=%.2f, tau=%.2f, gamma=%.2f, forget=%.5f" %
              (participant_id, min_fun, *min_fun_x))
        writer.writerow(dict(zip(fieldnames, [participant_id, min_fun, *min_fun_x])))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
