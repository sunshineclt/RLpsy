import csv
import json
import os

import numpy as np

from data_analysis.analysis_optimal import optimal_probability
from utils.draw import draw_different_params


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 50
    TRIAL_LENGTH = 144
    SIMULATE_METHOD = "MB"
    randomized = True

    all_reduction = {}
    for forward_planning in [1, 2, 3, 4, 5, 6, 7]:
        all_reduction[forward_planning] = {}
        for eta in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            all_reduction[forward_planning][eta] = {}
            for tau in [0.1, 0.5, 1, 5, 10, 100]:
                all_reduction[forward_planning][eta][tau] = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
                BASE_PATH = "data/MB/eta%.1f_tau%.1f_forward%d/" % (eta, tau, forward_planning) + (
                    "randomized" if randomized else "block") + "/"

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

                    steps = []
                    for trial in trials_data:
                        steps.append(len(trial))

                    all_reduction[forward_planning][eta][tau][participant_id].append(steps)
                    # result = optimal_probability(participant_id, trials_data, is_simulate=True, is_randomized=randomized)
                    # all_reduction["optimal"] = result[0]
                    # all_reduction["optimal_inner"] = result[1]["inner"]
                    # all_reduction["optimal_outer"] = result[1]["outer"]
                    # all_reduction["optimal_last"] = result[1]["last"]

    draw_different_params(all_reduction[1],
                          "step",
                          "step under %s condition in simulated algo %s forward 1" % (("randomized" if randomized else "block"), SIMULATE_METHOD),
                          smooth=True,
                          trial_length=TRIAL_LENGTH,
                          show=True)
