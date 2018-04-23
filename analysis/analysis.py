import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.savitzky_golay import savitzky_golay

SAVITZKY_GOLAY_WINDOW = 15
SAVITZKY_GOLAY_ORDER = 1


def plot_and_calculate(data, data_name):
    data = np.array(data)

    randomized_data = data[::2, :]
    mean = np.mean(randomized_data, axis=0)
    mean = savitzky_golay(mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
    std = np.std(randomized_data, axis=0)
    plt.plot(mean, linewidth=3.0, label="mean")
    plt.fill_between(range(0, TRIAL_LENGTH), mean - 2 * std, mean + 2 * std, alpha=0.2)
    if data_name == "step":
        simulate_random_mean = np.load("random_randomized.npy")
        plt.plot(simulate_random_mean, linewidth=3.0, label="random")
        simulate_MF_mean = np.load("MF_randomized.npy")
        plt.plot(simulate_MF_mean, linewidth=3.0, label="MF")
    for participant in range(0, NUMBER_OF_PARTICIPANT, 2):
        plt.plot(savitzky_golay(data[participant], SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER), linewidth=0.5, label=str(participant))
    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    plt.legend(loc='upper right')
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    else:
        plt.ylim([0, 4])
    plt.title(data_name + " under randomized condition")
    plt.show()

    block_data = data[1::2, :]
    mean = np.mean(block_data, axis=0)
    mean = savitzky_golay(mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
    std = np.std(block_data, axis=0)
    plt.plot(mean, linewidth=3.0, label="mean")
    plt.fill_between(range(0, TRIAL_LENGTH), mean - 2 * std, mean + 2 * std, alpha=0.2)
    if data_name == "step":
        simulate_random_mean = np.load("random_block.npy")
        plt.plot(simulate_random_mean, linewidth=3.0, label="random")
        simulate_MF_mean = np.load("MF_block.npy")
        plt.plot(simulate_MF_mean, linewidth=3.0, label="MF")
    for participant in range(1, NUMBER_OF_PARTICIPANT, 2):
        plt.plot(savitzky_golay(data[participant], SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER), linewidth=0.5, label=str(participant))
    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    plt.legend(loc='upper right')
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    else:
        plt.ylim([0, 4])
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
    NUMBER_OF_PARTICIPANT = 18
    TRIAL_LENGTH = 144

    all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_steps = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_times = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_normalized_times = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
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
        all_data[participant_id] = trials_data

        # plt.plot(length)
        # plt.title("Participant " + split[0] + ": ")
        # plt.vlines(36, 0, 50)
        # plt.vlines(72, 0, 50)
        # plt.vlines(108, 0, 50)
        # plt.ylim([0, 50])
        # plt.show()

        steps = []
        times = []
        normalized_times = []
        for trial in trials_data:
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
