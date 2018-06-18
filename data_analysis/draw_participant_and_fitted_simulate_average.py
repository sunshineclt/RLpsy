from utils.DataSaver import DataSaver
from utils.draw import draw_participant_and_simulation

import numpy as np

import matplotlib.pyplot as plt

participant_analysis_data = DataSaver.load_from_file("analysis_result.pkl")
simulate_analysis_data = {"MF": [],
                          "MB": []}
NUMBER_OF_REPEAT = 50
for i in range(NUMBER_OF_REPEAT):
    analysis_result = DataSaver.load_from_file("simulate_analysis_result/MF_simulate_analysis_result" + str(i) + ".pkl")
    simulate_analysis_data["MF"].append(analysis_result)
    analysis_result = DataSaver.load_from_file("simulate_analysis_result/MB_simulate_analysis_result" + str(i) + ".pkl")
    simulate_analysis_data["MB"].append(analysis_result)


def draw(metric, method, simulate_data, participant_data, randomized):
    draw_participant_and_simulation(participant_data,
                                    simulate_data,
                                    metric,
                                    title=metric + "_in_" + method + "_method_" + ("randomized" if randomized else "block") + "_condition",
                                    save_path="participant_simulation_comparison_all/",
                                    save=False,
                                    show=False,
                                    multiple_simulation=True,
                                    smooth=False)


for metric_name in ["optimal", "optimal_inner", "optimal_outer", "optimal_last"]:
    plt.figure(figsize=[12, 8], dpi=80)
    averaged_simulate_data = []
    for i in range(NUMBER_OF_REPEAT):
        averaged_simulate_data.append(simulate_analysis_data["MF"][i].get_data(metric_name))
    averaged_simulate_data = np.array(averaged_simulate_data)
    plt.subplot(221)
    draw(metric_name, "MF", np.mean(averaged_simulate_data[:, ::2, :], axis=1), participant_analysis_data.get_data(metric_name)[::2, :], randomized=True)
    plt.subplot(222)
    draw(metric_name, "MF", np.mean(averaged_simulate_data[:, 1::2, :], axis=1), participant_analysis_data.get_data(metric_name)[1::2, :], randomized=False)
    averaged_simulate_data = []
    for i in range(NUMBER_OF_REPEAT):
        averaged_simulate_data.append(simulate_analysis_data["MB"][i].get_data(metric_name))
    averaged_simulate_data = np.array(averaged_simulate_data)
    plt.subplot(223)
    draw(metric_name, "MB", np.mean(averaged_simulate_data[:, ::2, :], axis=1),
         participant_analysis_data.get_data(metric_name)[::2, :], randomized=True)
    plt.subplot(224)
    draw(metric_name, "MB", np.mean(averaged_simulate_data[:, 1::2, :], axis=1),
         participant_analysis_data.get_data(metric_name)[1::2, :], randomized=False)
    plt.show()
