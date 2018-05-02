import csv
import json
import pickle
import os

from data_analysis.analysis_optimal import optimal_probability

if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 50
    TRIAL_LENGTH = 144
    SIMULATE_METHOD = "MF_forget"
    randomized = False

    all_reduction = {}
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        all_reduction[alpha] = {}
        for tau in [0.1, 0.5, 1, 5, 10, 100]:
            all_reduction[alpha][tau] = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
            BASE_PATH = "data/MF_forget/alpha%.1f_tau%.1f/" % (alpha, tau) + ("randomized" if randomized else "block") + "/"

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

                # all_reduction[eta][tau][participant_id] = steps
                result = optimal_probability(participant_id, trials_data, is_simulate=True,
                                             is_randomized=randomized)
                all_reduction[alpha][tau][participant_id] = result[0]
                # all_reduction[eta][tau][participant_id] = result[1]["inner"]
                # all_reduction[eta][tau][participant_id] = result[1]["outer"]
                # all_reduction[eta][tau][participant_id] = result[1]["last"]

    with open("optimal_MF_forget_block.pkl", "wb") as f:
        pickle.dump(all_reduction, f)