import csv
import json
import os

import numpy as np
import pandas

from data_analysis.analysis_optimal import optimal_probability
from utils.DataSaver import DataSaver


if __name__ == "__main__":
    NUMBER_OF_PARTICIPANT = 36
    TRIAL_LENGTH = 144
    NEED_UPDATE_DATA_FRAME = False

    saver = DataSaver(["step", "time", "normalized_time", "optimal",
                       "optimal_inner", "optimal_outer", "optimal_last"],
                      NUMBER_OF_PARTICIPANT,
                      TRIAL_LENGTH)
    all_data = [[] for _ in range(NUMBER_OF_PARTICIPANT)]
    optimal_data_frame = pandas.DataFrame(
        columns=["step", "reaction_time", "normalized_reaction_time", "optimal_p", "optimal_inner", "optimal_outer",
                 "optimal_last", "timestep", "block", "condition", "participant"])

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

        count = np.zeros(shape=[6, 3, 6])
        for trial_id, trial in enumerate(trials_data):
            saver.save_data("step", participant_id, trial_id, len(trial))
            time = 0
            for step in trial:
                time += step[3]
            saver.save_data("time", participant_id, trial_id, time)
            saver.save_data("normalized_time", participant_id, trial_id, time / len(trial))

        result = optimal_probability(participant_id, trials_data, is_simulate=False)
        saver.save_trials_data("optimal", participant_id, result[0])
        saver.save_trials_data("optimal_inner", participant_id, result[1]["inner"])
        saver.save_trials_data("optimal_outer", participant_id, result[1]["outer"])
        saver.save_trials_data("optimal_last", participant_id, result[1]["last"])
        if NEED_UPDATE_DATA_FRAME:
            for trial in range(TRIAL_LENGTH):
                block = trial // 36
                timestep = trial % 36
                optimal_data_frame = \
                    optimal_data_frame.append({"step": saver.get_trial_data("step", participant_id),
                                               "reaction_time": saver.get_trial_data("time", participant_id),
                                               "normalized_reaction_time": saver.get_trial_data("normalized_time", participant_id),
                                               "optimal_p": result[0][trial],
                                               "optimal_inner": result[1]["inner"][trial],
                                               "optimal_outer": result[1]["outer"][trial],
                                               "optimal_last": result[1]["last"][trial],
                                               "timestep": timestep,
                                               "trial": trial,
                                               "block": block,
                                               "condition": (
                                                   "random" if participant_id % 2 == 0 else "block"),
                                               "participant": participant_id},
                                              ignore_index=True)

    # SAVE SAVER
    saver.save_to_file("analysis_result.pkl")

    # SAVE DATA FRAME
    if NEED_UPDATE_DATA_FRAME:
        optimal_data_frame.to_csv("optimal_data_frame.csv")
