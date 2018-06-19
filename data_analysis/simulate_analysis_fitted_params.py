import csv
import json
import os

from simulate.simulate_MF import simulate_MF
from simulate.simulate_MB import simulate_MB
from simulate.simulate_MF_no_reinit import simulate_MF_no_reinit
from utils.DataSaver import DataSaver
from data_analysis.analysis_optimal import optimal_probability

MODEL_NAME = "MF_no_reinit"

f = open(MODEL_NAME + "_fit_result.csv", "r")
fit_result = csv.DictReader(f)

NUMBER_OF_PARTICIPANT = 36
TRIAL_LENGTH = 144
NUMBER_OF_REPEAT = 50
savers = []
for _ in range(NUMBER_OF_REPEAT):
    savers.append(DataSaver(["optimal",
                             "optimal_inner", "optimal_outer", "optimal_last"],
                            NUMBER_OF_PARTICIPANT,
                            TRIAL_LENGTH))

for participant_param in fit_result:
    participant_id = int(participant_param["participant"])
    # Use the fitted model to simulate
    BASE_PATH = "simulate_data/%s/%02d/" % (MODEL_NAME, participant_id)
    # simulate_MF(randomized=participant_id % 2 == 0,
    #             alpha=float(participant_param["alpha"]),
    #             tau=float(participant_param["tau"]),
    #             repeat=NUMBER_OF_REPEAT,
    #             gamma=float(participant_param["gamma"]),
    #             forget=float(participant_param["forget_MF"]),
    #             path=BASE_PATH,
    #             seed=participant_id)
    # simulate_MB(randomized=participant_id % 2 == 0,
    #             eta=float(participant_param["eta"]),
    #             tau=float(participant_param["tau"]),
    #             repeat=NUMBER_OF_REPEAT,
    #             gamma=float(participant_param["gamma"]),
    #             forget=float(participant_param["forget_MB"]),
    #             path=BASE_PATH,
    #             seed=participant_id)
    simulate_MF_no_reinit(randomized=participant_id % 2 == 0,
                          alpha=float(participant_param["alpha"]),
                          tau=float(participant_param["tau"]),
                          repeat=NUMBER_OF_REPEAT,
                          gamma=float(participant_param["gamma"]),
                          forget=float(participant_param["forget_MF"]),
                          path=BASE_PATH,
                          seed=participant_id)

    for file in os.listdir(BASE_PATH):
        path = os.path.join(BASE_PATH, file)
        print("Loading ", path)
        split = file.split("_")
        repeat_id = int(split[0])

        rawFile = open(path, "r")
        reader = csv.DictReader(rawFile, delimiter="#")
        trials_data = []
        for row in reader:
            trial = row["trial_data"]
            if trial != "--":
                transformed_trial = json.loads(trial)
                trials_data.append(transformed_trial)
        trials_data = trials_data[:TRIAL_LENGTH]

        result = optimal_probability(participant_id, trials_data, is_simulate=True,
                                     is_randomized=participant_id % 2 == 0)
        savers[repeat_id].save_trials_data("optimal", participant_id, result[0])
        savers[repeat_id].save_trials_data("optimal_inner", participant_id, result[1]["inner"])
        savers[repeat_id].save_trials_data("optimal_outer", participant_id, result[1]["outer"])
        savers[repeat_id].save_trials_data("optimal_last", participant_id, result[1]["last"])

for repeat in range(NUMBER_OF_REPEAT):
    savers[repeat].save_to_file(
        "simulate_analysis_result/" + MODEL_NAME + "_simulate_analysis_result" + str(repeat) + ".pkl")
