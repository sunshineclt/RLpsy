import csv
import json
import os

import numpy as np

from data_analysis.analysis_optimal import optimal_probability
from utils.draw import draw_metrics


def transform_and_plot(data, data_name, save_path=None):
    data = np.array(data)
    draw_metrics(data,
                 data_name,
                 "%s under %s condition in simulated algo %s" % (data_name, ("randomized" if randomized else "block"), SIMULATE_METHOD),
                 smooth=True,
                 trial_length=TRIAL_LENGTH,
                 save_npy=False,
                 save_path=save_path,
                 draw_individual=True,
                 annotate_block_mean=True,
                 show=True)


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 50
    TRIAL_LENGTH = 144
    SIMULATE_METHOD = "MB"
    randomized = True
    BASE_PATH = os.path.join("data", SIMULATE_METHOD, "randomized" if randomized else "block")

    all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_steps = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities_inner = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    all_optimal_probabilities_outer = [[] for _ in range(NUMBER_OF_PARTICIPANT)]

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
        result = optimal_probability(participant_id, trials_data, is_simulate=True, is_randomized=randomized)
        all_optimal_probabilities[participant_id] = result[0]
        all_optimal_probabilities_inner[participant_id] = result[1]["inner"]
        all_optimal_probabilities_outer[participant_id] = result[1]["outer"]

    transform_and_plot(all_steps, "step",
                       save_path=SIMULATE_METHOD + "_" + ("randomized" if randomized else "block") + "_step.npy")
    transform_and_plot(all_optimal_probabilities, "optimal",
                       save_path=SIMULATE_METHOD + "_" + ("randomized" if randomized else "block") + "_optimal.npy")
    transform_and_plot(all_optimal_probabilities_inner, "optimal_inner",
                       save_path=SIMULATE_METHOD + "_" + ("randomized" if randomized else "block") + "_optimal_inner.npy")
    transform_and_plot(all_optimal_probabilities_outer, "optimal_outer",
                       save_path=SIMULATE_METHOD + "_" + ("randomized" if randomized else "block") + "_optimal_outer.npy")
