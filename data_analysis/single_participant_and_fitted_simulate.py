import numpy as np

from utils.DataSaver import DataSaver

participant_analysis_data = DataSaver.load_from_file("analysis_result.pkl")
simulate_analysis_data = {"MF": [],
                          "MB": []}
NUMBER_OF_REPEAT = 50
for i in range(NUMBER_OF_REPEAT):
    analysis_result = DataSaver.load_from_file("simulate_analysis_result/MF_simulate_analysis_result" + str(i) + ".pkl")
    simulate_analysis_data["MF"].append(analysis_result)
    analysis_result = DataSaver.load_from_file("simulate_analysis_result/MB_simulate_analysis_result" + str(i) + ".pkl")
    simulate_analysis_data["MB"].append(analysis_result)


for participant_id in range(36):
    for model in ["MF", "MB"]:
        metric_name = "optimal"
        simulate_data = []
        for i in range(NUMBER_OF_REPEAT):
            simulate_data.append(simulate_analysis_data[model][i].get_trial_data(metric_name, participant_id))
        simulate_data = np.array(simulate_data)
        mean_simulate_data = np.mean(simulate_data, axis=0)
        std_simulate_data = np.std(simulate_data, axis=0)
        lower = mean_simulate_data - 1.96 * std_simulate_data
        upper = mean_simulate_data + 1.96 * std_simulate_data
        participant_single_data = participant_analysis_data.get_trial_data(metric_name, participant_id)
        not_fall_in = np.sum(participant_single_data < lower) + np.sum(participant_single_data > upper)
        print("participant: %d, model: %s, not in 95%% CI count is %d" % (participant_id, model, not_fall_in))
