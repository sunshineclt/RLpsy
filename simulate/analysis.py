import csv
import json
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_and_calculate(data, data_name):
    data = np.array(data)
    plt.plot(data.mean(axis=0))
    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    else:
        plt.ylim([0, 10])
    plt.title(data_name + " under block condition")
    plt.show()

    print("All ", data_name, ":")
    print(data.mean(axis=0)[:36].mean())
    print(data.mean(axis=0)[36:72].mean())
    print(data.mean(axis=0)[72:108].mean())
    print(data.mean(axis=0)[108:].mean())


if __name__ == "__main__":
    numer_of_participant = 5
    all_data = [[] for _ in range(numer_of_participant)]
    all_steps = [[] for _ in range(numer_of_participant)]
    for lists in os.listdir("data/MF_randomized/"):
        path = os.path.join("data/MF_randomized/", lists)
        print("Loading ", path)
        split = lists.split("_")
        participant_id = int(split[0])

        rawFile = open(path, "r")
        reader = csv.DictReader(rawFile, delimiter="#")
        trials = []
        for row in reader:
            trial = row["trial_data"]
            if trial != "--":
                transformed_trial = json.loads(trial)
                trials.append(transformed_trial)
        trials = trials[:144]
        all_data[participant_id] = trials

        steps = []
        for trial in trials:
            steps.append(len(trial))

        all_steps[participant_id] = steps

    plot_and_calculate(all_steps, "step")
