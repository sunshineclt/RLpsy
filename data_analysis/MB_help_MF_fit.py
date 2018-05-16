import csv
import datetime
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import optimize

from utils import utils
import utils.Params as Params


def MB_help_MF_lld(params):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    eta = params[3]
    forget_MF = params[4]
    forget_MB = params[5]

    q_value_MF = np.zeros(shape=[3, 6, 3])
    trans_prob = np.zeros(shape=[6, 3, 6]) + 1 / 6

    lld = 0
    for episode in range(144):
        trial_end_state = trials_data[episode][-1][2]

        step = 0
        for transit in trials_data[episode]:
            now_state = transit[0]
            action_chosen = transit[1]
            new_state = transit[2]
            step += 1

            action_chosen_prob = utils.softmax(q_value_MF[trial_end_state][now_state], tau)
            likelihood = action_chosen_prob[action_chosen]

            if likelihood < 1e-200:
                lld += 460.517
            else:
                lld -= np.log(likelihood)

            trans_prob_to_new_state = trans_prob[now_state, action_chosen, new_state]
            trans_prob[now_state, action_chosen] *= (1 - eta)
            trans_prob[now_state, action_chosen, new_state] = trans_prob_to_new_state + eta * (
                    1 - trans_prob_to_new_state)
            # transition trace
            trans_prob = (1 / 6 - trans_prob) * forget_MB + trans_prob

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
                delta = target - q_value_MF[trial_end_state][now_state, action_should_chosen]
                q_value_MF[trial_end_state][now_state, action_should_chosen] += alpha * delta
                q_value_MF *= (1 - forget_MF)

    return lld


if __name__ == "__main__":
    start_time_stamp = datetime.datetime.now()
    print("start time: ", start_time_stamp.strftime('%H:%M:%S'))

    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144

    MBHMF_fit_result = open("MBHMF_result.csv", "w")
    fieldnames = ["participant",
                  "nlld",
                  "alpha",
                  "tau",
                  "gamma",
                  "eta",
                  "forget_MF",
                  "forget_MB"]
    writer = csv.DictWriter(MBHMF_fit_result, fieldnames)
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


        pool = mp.Pool(32)
        bounds = [Params.PARAM_BOUNDS["alpha"],
                  Params.PARAM_BOUNDS["tau"],
                  Params.PARAM_BOUNDS["gamma"],
                  Params.PARAM_BOUNDS["eta"],
                  Params.PARAM_BOUNDS["forget_MF"],
                  Params.PARAM_BOUNDS["forget_MB"]]
        NUMBER_OF_INITIAL_VALUE = 1000
        initial_value = np.zeros(shape=[NUMBER_OF_INITIAL_VALUE, len(bounds)])
        for i in range(NUMBER_OF_INITIAL_VALUE):
            for bound_index, bound in enumerate(bounds):
                initial_value[i, bound_index] = np.random.random() * (bound[1] - bound[0]) + bound[0]

        condition = [(MB_help_MF_lld, initial, bounds) for initial in initial_value]

        result = pool.map(minimize, condition)
        pool.close()

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
