import csv
import json
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_and_calculate(data, data_name):
    data = np.array(data)

    randomized_data = data[[0, 2, 4, 6, 8]]
    plt.plot(np.median(randomized_data, axis=0))
    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    else:
        plt.ylim([0, 10])
    plt.title(data_name + " under randomized condition")
    plt.show()

    block_data = data[[1, 3, 5, 7, 9]]
    plt.plot(np.median(block_data, axis=0))
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

    print("Randomized", data_name, ":")
    print(randomized_data.mean(axis=0)[:36].mean())
    print(randomized_data.mean(axis=0)[36:72].mean())
    print(randomized_data.mean(axis=0)[72:108].mean())
    print(randomized_data.mean(axis=0)[108:].mean())

    print("Block", data_name, ":")
    print(block_data.mean(axis=0)[:36].mean())
    print(block_data.mean(axis=0)[36:72].mean())
    print(block_data.mean(axis=0)[72:108].mean())
    print(block_data.mean(axis=0)[108:].mean())


if __name__ == "__main__":
    numer_of_participant = 10
    all_data = [[] for _ in range(numer_of_participant)]
    all_steps = [[] for _ in range(numer_of_participant)]
    all_times = [[] for _ in range(numer_of_participant)]
    all_normalized_times = [[] for _ in range(numer_of_participant)]
    for lists in os.listdir("data/"):
        path = os.path.join("data/", lists)
        print("Loading ", path)
        split = lists.split("_")
        participant_id = int(split[0])

        rawFile = open(path, "r")
        reader = csv.DictReader(rawFile, delimiter="#")
        trials = []
        length = []
        for row in reader:
            trial = row["trial_data"]
            if trial != "--":
                transformed_trial = json.loads(trial)
                trials.append(transformed_trial)
                length.append(len(transformed_trial))
        trials = trials[:144]
        all_data[participant_id] = trials
        plt.plot(length)
        plt.title("Participant " + split[0] + ": ")
        plt.vlines(36, 0, 50)
        plt.vlines(72, 0, 50)
        plt.vlines(108, 0, 50)
        plt.ylim([0, 50])
        plt.show()

        steps = []
        times = []
        normalized_times = []
        for trial in trials:
            steps.append(len(trial))
            times.append(0)
            for step in trial:
                times[-1] += step[3]
            normalized_times.append(times[-1] / steps[-1])
        all_steps[participant_id] = steps
        all_times[participant_id] = times
        all_normalized_times[participant_id] = normalized_times

    plot_and_calculate(all_steps, "step")
    plot_and_calculate(all_times, "time")
    plot_and_calculate(all_normalized_times, "normalized_time")
