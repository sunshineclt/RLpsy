import csv
import json
import os

import numpy as np

from data_analysis.analysis_optimal import optimal_probability
from utils.draw import draw_different_params


def transform_and_plot(data, data_name):
    draw_different_params(data,
                          data_name,
                          "%s under %s condition in simulated algo %s" % (data_name, ("randomized" if randomized else "block"), SIMULATE_METHOD),
                          smooth=True,
                          trial_length=TRIAL_LENGTH,
                          show=True)


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144
    SIMULATE_METHOD = "MF"
    randomized = True

    all_all_steps = {}
    all_all_optimal_probabilities = {}
    all_all_optimal_probabilities_inner = {}
    all_all_optimal_probabilities_outer = {}
    all_all_optimal_probabilities_last = {}
    for alpha in [i / 10 for i in range(1, 11, 1)]:

        BASE_PATH = os.path.join("data", SIMULATE_METHOD, str(alpha), "randomized" if randomized else "block")

        all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
        all_steps = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
        all_optimal_probabilities = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
        all_optimal_probabilities_inner = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
        all_optimal_probabilities_outer = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
        all_optimal_probabilities_last = [[] for _ in range(NUMBER_OF_PARTICIPANT)]

        for lists in os.listdir(BASE_PATH):
            path = os.path.join(BASE_PATH, lists)
            print("Loading ", path)
            split = lists.split("_")
            participant_id = int(split[0])
            if participant_id >= NUMBER_OF_PARTICIPANT:
                continue

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
            all_optimal_probabilities_last[participant_id] = result[1]["last"]

        all_all_steps[str(alpha)] = all_steps
        all_all_optimal_probabilities[str(alpha)] = all_optimal_probabilities
        all_all_optimal_probabilities_inner[str(alpha)] = all_optimal_probabilities_inner
        all_all_optimal_probabilities_outer[str(alpha)] = all_optimal_probabilities_outer
        all_all_optimal_probabilities_last[str(alpha)] = all_optimal_probabilities_last

    transform_and_plot(all_all_steps, "step")
    transform_and_plot(all_all_optimal_probabilities, "optimal")
    transform_and_plot(all_all_optimal_probabilities_inner, "optimal_inner")
    transform_and_plot(all_all_optimal_probabilities_outer, "optimal_outer")
    transform_and_plot(all_all_optimal_probabilities_last, "optimal_last")
