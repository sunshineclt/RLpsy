import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.savitzky_golay import savitzky_golay


def plot_and_calculate(data, data_name):
    data = np.array(data)

    mean = np.mean(data, axis=0)
    mean = savitzky_golay(mean, 25, 1)
    std = np.std(data, axis=0)
    np.save(os.path.join(BASE_PATH, "mean.npy"), mean)
    plt.plot(mean, linewidth=3.0, label="mean")
    plt.fill_between(range(0, TRIAL_LENGTH), mean - 2 * std, mean + 2 * std, alpha=0.2)
    for participant in range(0, NUMBER_OF_PARTICIPANT, 2):
        plt.plot(savitzky_golay(data[participant], 25, 1), linewidth=0.5, label=str(participant))
    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    else:
        plt.ylim([0, 4])
    plt.title(data_name + " under " + ("randomized" if randomized else "block") + " condition")
    plt.show()

    print("All ", data_name, ":")
    print(data.mean(axis=0)[:36].mean())
    print(data.mean(axis=0)[36:72].mean())
    print(data.mean(axis=0)[72:108].mean())
    print(data.mean(axis=0)[108:].mean())


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 50
    TRIAL_LENGTH = 144
    BASE_PATH = "data/MF/"
    randomized = True
    BASE_PATH += ("randomized" if randomized else "block")

    all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_steps = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    for lists in os.listdir(BASE_PATH):
        path = os.path.join(BASE_PATH, lists)
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
        all_data[participant_id] = trials_data

        steps = []
        for trial in trials_data:
            steps.append(len(trial))
        all_steps[participant_id] = steps

    plot_and_calculate(all_steps, "step")
