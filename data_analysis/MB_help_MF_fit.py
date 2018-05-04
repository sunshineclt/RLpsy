import csv
import datetime
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import optimize

from utils import utils


def MB_help_MF_lld(params):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    eta = params[3]
    forget_MF = params[4]
    forget_MB = params[5]
    # I = params[6]
    # k = params[7]
    forward_planning = 2

    # model and parameters
    q_value_MF = np.zeros(shape=[3, 6, 3])
    trans_prob = np.zeros(shape=[6, 3, 6]) + 1 / 6

    lld = 0
    global_step = 0
    for episode in range(144):
        trial_end_state = trials_data[episode][-1][2]

        step = 0
        r = np.zeros(shape=[6])
        for transit in trials_data[episode]:
            now_state = transit[0]
            action_chosen = transit[1]
            new_state = transit[2]
            step += 1
            global_step += 1

            action_chosen_prob = utils.softmax(q_value_MF[trial_end_state][now_state], tau)
            likelihood = action_chosen_prob[action_chosen]

            if likelihood < 1e-200:
                lld += 1000000
            else:
                lld -= np.log(likelihood)

            # if utils.softmax(q_value_MF[trial_end_state][last_state], tau)[last_action] > 0.7:
            # lr = eta * utils.softmax(q_value_MF[trial_end_state][last_state], tau)[last_action]
            lr = eta
            #  now_state == 0 and action_chosen == 0:
            #     print(new_state)
            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - lr)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + lr * (
                    1 - trans_prob_to_new_state)
            trans_prob = (1 / 6 - trans_prob) * forget_MB + trans_prob

            # r[trial_end_state] = 20
            # q_value_MB = trans_prob[now_state].dot(r)  # shape = 3, 6
            # it = trans_prob[now_state].copy()  # shape = 3, 6
            # for iter_times in range(forward_planning - 1):
            #     new_it = np.zeros(shape=[3, 6])
            #     for action in range(3):
            #         for old_state in range(6):
            #             old_action = np.argmax(q_value_MF[trial_end_state][old_state])
            #             new_it[action] += it[action, old_state] * trans_prob[old_state, old_action]  # vector calculation
            #     r[trial_end_state] = (19 - iter_times) * (gamma ** iter_times)
            #     assert np.all(np.sum(new_it, axis=1) - 1 < 1e-3)
            #     it = new_it
            #     q_value_MB += it.dot(r)
            # if now_state == 0:
            #     print(successor[0])
            # q_value_MB = successor.dot(r)
            # q_value_MB = it.dot(np.max(q_value_MF[trial_end_state], axis=1))
            # delta = q_value_MB - q_value_MF[trial_end_state][now_state]
            # q_value_MF[trial_end_state][now_state] += alpha * delta


            if trial_end_state == new_state:
                target = max(21 - step, 1)
            else:
                target = gamma * np.max(q_value_MF[trial_end_state][new_state])

            if np.max(trans_prob[now_state, :, new_state]) <= .7:
                delta = target - q_value_MF[trial_end_state][now_state, action_chosen]
                q_value_MF[trial_end_state][now_state, action_chosen] += alpha * delta
                q_value_MF *= (1 - forget_MF)
            else:
                action_should_chosen = np.argmax(trans_prob[now_state, :, new_state])
                if action_should_chosen != action_chosen:
                    print(trans_prob[now_state, action_chosen, new_state], trans_prob[now_state, action_should_chosen, new_state])

                delta = target - q_value_MF[trial_end_state][now_state, action_should_chosen]
                q_value_MF[trial_end_state][now_state, action_should_chosen] += alpha * delta
                delta = target - q_value_MF[trial_end_state][now_state, action_chosen]
                q_value_MF[trial_end_state][now_state, action_chosen] += alpha * delta
                q_value_MF *= (1 - forget_MF)

    return lld


if __name__ == "__main__":
    start_time_stamp = datetime.datetime.now()
    print("start time: ", start_time_stamp.strftime('%H:%M:%S'))

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
        condition = [(MB_help_MF_lld, initial[:6], bounds[:6]) for initial in initial_value]

        MB_help_MF_lld([0.3, 1, 1, 0.3, 0.001, 0.001])
        break

        result = pool.map(minimize, condition)
        min_fun = 1e50
        min_fun_x = 0
        for i in range(len(initial_value)):
            if result[i].fun < min_fun:
                min_fun = result[i].fun
                min_fun_x = result[i].x
        print(
            "For participant %d, best fit lld is %.3f, alpha=%.2f, tau=%.2f, gamma=%.2f, eta=%.2f, forget_MF=%.5f, forget_MB=%.5f" %
            (participant_id, min_fun, *min_fun_x))
        writer.writerow(dict(zip(fieldnames, [participant_id, min_fun, *min_fun_x])))

    end_time_stamp = datetime.datetime.now()
    print("end time: ", end_time_stamp.strftime('%H:%M:%S'))
    print("takes time: ", (end_time_stamp - start_time_stamp).seconds)
