import csv
import datetime
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import optimize

from utils import utils


def hybrid_lld(params):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    eta = params[3]
    I = params[4]
    k = params[5]
    forget_MF = params[6]
    forget_MB = params[7]
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
            new_state = transit[2]
            step += 1
            global_step += 1

            q_value_MB = np.zeros(shape=[6, 3])
            for iter_times in range(forward_planning):
                new_q_value = np.zeros(shape=[6, 3])
                for state in range(6):
                    for action in range(3):
                        new_q_value[state, action] = np.sum(
                            trans_prob[state, action] * (r + gamma * np.max(q_value_MB, axis=1)))
                        # for state_1 in range(6):
                        #     new_new_q_value[state, action] += trans_prob[state, action, state_1] * \
                        #                                   (r[state_1] + gamma * np.max(q_value[state_1]))
                q_value_MB = new_q_value

            weight_MB = I * np.exp(-k * global_step)
            hybrid_q_value = q_value_MB[now_state] * weight_MB + (1 - weight_MB) * q_value_MF[trial_end_state][
                now_state]
            likelihood = utils.softmax(np.array([hybrid_q_value[0],
                                                 hybrid_q_value[1],
                                                 hybrid_q_value[2]]),
                                       tau)[action_chosen]

            if likelihood < 1e-200:
                lld += 1000000
            else:
                lld -= np.log(likelihood)

            if trial_end_state == new_state:
                target = max(21 - step, 1)
            else:
                target = gamma * np.max(q_value_MF[trial_end_state][new_state])
            delta = target - q_value_MF[trial_end_state][now_state, action_chosen]
            q_value_MF[trial_end_state][now_state, action_chosen] += alpha * delta
            q_value_MF *= (1 - forget_MF)

            new_state = transit[2]
            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - eta)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + eta * (
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
                  "I",
                  "k",
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
        initial_value = []
        for i in range(12):
            initial_value.append(np.array([np.random.random(),
                                           np.random.random() * 100 + 1e-5,
                                           np.random.random(),
                                           np.random.random(),
                                           np.random.random(),
                                           np.random.random() * 0.1,
                                           np.random.random() * 0.05,
                                           np.random.random() * 0.05]))

        bounds = [(0, 1), (1e-5, 100), (0, 1), (0, 1), (0, 1), (0, 0.1), (0, 0.05), (0, 0.05)]
        condition = [(hybrid_lld, initial, bounds) for initial in initial_value]

        result = pool.map(minimize, condition)
        min_fun = 1e50
        min_fun_x = 0
        for i in range(len(initial_value)):
            if result[i].fun < min_fun:
                min_fun = result[i].fun
                min_fun_x = result[i].x
        print(
            "For participant %d, best fit lld is %.3f, alpha=%.2f, tau=%.2f, gamma=%.2f, eta=%.2f, I=%.2f, k=%.2f, forget_MF=%.2f, forget_MB=%.2f" %
            (participant_id, min_fun, *min_fun_x))
        writer.writerow(dict(zip(fieldnames, [participant_id, min_fun, *min_fun_x])))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
