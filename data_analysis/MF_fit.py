import csv
import datetime
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import optimize

from utils import utils
import utils.Params as Params
from data_analysis.analysis_optimal import optimal_probability


def MF_lld(params):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    forget = params[3]

    q_value = np.zeros(shape=[3, 6, 3])

    lld = 0
    for episode in range(TRIAL_LENGTH):
        trial_end_state = trials_data[episode][-1][2]

        step = 0
        if not Q_LEARNING:
            last_state = 0
            last_action = 0
        for transit in trials_data[episode]:
            now_state = transit[0]
            action_chosen = transit[1]
            step += 1

            likelihood = utils.softmax(np.array([q_value[trial_end_state][now_state, 0],
                                                 q_value[trial_end_state][now_state, 1],
                                                 q_value[trial_end_state][now_state, 2]]),
                                       tau)[action_chosen]
            if likelihood < 1e-200:
                lld += 460.517
            else:
                lld -= np.log(likelihood)

            if Q_LEARNING:
                new_state = transit[2]
                if trial_end_state == new_state:
                    target = max(21 - step, 1)
                else:
                    target = gamma * np.max(q_value[trial_end_state][new_state])
                delta = target - q_value[trial_end_state][now_state, action_chosen]
                q_value[trial_end_state][now_state, action_chosen] += alpha * delta
            elif step != 1:
                target = 0 + gamma * q_value[trial_end_state][now_state, action_chosen]
                delta = target - q_value[trial_end_state][last_state, last_action]
                q_value[trial_end_state][last_state, last_action] += alpha * delta

            if not Q_LEARNING:
                last_state = now_state
                last_action = action_chosen

            # forget
            q_value *= (1 - forget)

        if not Q_LEARNING:
            target = max(21 - step, 1)
            delta = target - q_value[trial_end_state][last_state, last_action]
            q_value[trial_end_state][last_state, last_action] += alpha * delta

    return lld


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print("start time: ", time_stamp.strftime('%H:%M:%S'))

    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144
    Q_LEARNING = True

    MF_fit_result = open("MF_fit_result.csv", "w")
    fieldnames = ["participant",
                  "nlld",
                  "alpha",
                  "tau",
                  "gamma",
                  "forget_MF"]
    writer = csv.DictWriter(MF_fit_result, fieldnames)
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

        optimal_analysis_result = optimal_probability(participant_id, trials_data, is_simulate=False)

        # model fitting
        def minimize(args):
            f, x, bound = args
            res = optimize.minimize(f, x, bounds=bound)
            return res

        pool = mp.Pool(32)
        bounds = [Params.PARAM_BOUNDS["alpha"],
                  Params.PARAM_BOUNDS["tau"],
                  Params.PARAM_BOUNDS["gamma"],
                  Params.PARAM_BOUNDS["forget_MF"]]
        NUMBER_OF_INITIAL_VALUE = 1000
        initial_value = np.zeros(shape=[NUMBER_OF_INITIAL_VALUE, len(bounds)])
        for i in range(NUMBER_OF_INITIAL_VALUE):
            for bound_index, bound in enumerate(bounds):
                initial_value[i, bound_index] = np.random.random() * (bound[1] - bound[0]) + bound[0]

        condition = [(MF_lld, initial, bounds) for initial in initial_value]

        result = pool.map(minimize, condition)
        pool.close()

        min_fun = 1e50
        min_fun_x = -1
        for i in range(len(initial_value)):
            if result[i].fun < min_fun:
                min_fun = result[i].fun
                min_fun_x = result[i].x

        print("For participant %d, best fit lld is %.3f, alpha=%.2f, tau=%.2f, gamma=%.2f, forget=%.5f" %
              (participant_id, min_fun, *min_fun_x))
        writer.writerow(dict(zip(fieldnames, [participant_id, min_fun, *min_fun_x])))

    time_stamp = datetime.datetime.now()
    print("end time: ", time_stamp.strftime('%H:%M:%S'))
