import csv
import json
import os

import numpy as np

from data_analysis.analysis_optimal import optimal_probability
from utils.draw import draw_metrics


def transform_and_plot(data, data_name):
    data = np.array(data)

    randomized_data = data[::2, :]
    draw_metrics(randomized_data,
                 data_name,
                 "%s under %s condition in participants" % (data_name, "randomized"),
                 extra_data_names=["MF_randomized", "optimal_randomized", "random_randomized"],
                 smooth=True,
                 save_npy=False,
                 draw_individual=True,
                 annotate_block_mean=True)

    block_data = data[1::2, :]
    draw_metrics(block_data,
                 data_name,
                 "%s under %s condition in participants" % (data_name, "block"),
                 extra_data_names=["MF_block", "optimal_block", "random_block"],
                 smooth=True,
                 save_npy=False,
                 draw_individual=True,
                 annotate_block_mean=True)


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144

    all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_steps = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_times = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_normalized_times = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities_inner = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities_outer = [[] for _ in range(NUMBER_OF_PARTICIPANT)]

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

        steps = []
        times = []
        normalized_times = []
        count = np.zeros(shape=[6, 3, 6])
        for trial in trials_data:
            steps.append(len(trial))
            times.append(0)
            for step in trial:
                times[-1] += step[3]
                count[step[0], step[1], step[2]] += 1
            normalized_times.append(times[-1] / steps[-1])
        # print(count, flush=True)
        all_steps[participant_id] = steps
        all_times[participant_id] = times
        all_normalized_times[participant_id] = normalized_times
        result = optimal_probability(participant_id, trials_data, is_simulate=False)
        all_optimal_probabilities[participant_id] = result[0]
        all_optimal_probabilities_inner[participant_id] = result[1]["inner"]
        all_optimal_probabilities_outer[participant_id] = result[1]["outer"]

    transform_and_plot(all_steps, "step")
    transform_and_plot(all_times, "time")
    transform_and_plot(all_normalized_times, "normalized_time")
    transform_and_plot(all_optimal_probabilities, "optimal")
    transform_and_plot(all_optimal_probabilities_inner, "optimal_inner")
    transform_and_plot(all_optimal_probabilities_outer, "optimal_outer")
