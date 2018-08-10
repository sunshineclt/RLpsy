# This script is to use all fitted params to do simulation on all trials

from utils.DataSaver import DataSaver
from utils.draw import draw_even_odd_simulation

import numpy as np

import matplotlib.pyplot as plt

model_names = ["MF_attention"]  # could be "MF", "MB", "MF_no_reinit", "MF_attention"
randomized = True
model_number = len(model_names)

simulate_analysis_data = {}
for name in model_names:
    simulate_analysis_data[name] = []


NUMBER_OF_REPEAT = 50
for i in range(NUMBER_OF_REPEAT):
    for name in model_names:
        analysis_result = DataSaver.load_from_file("simulate_analysis_result_all_" +
                                                   ("randomized" if randomized else "block") + "/" + name +
                                                   "_simulate_analysis_result" + str(i) + ".pkl")
        simulate_analysis_data[name].append(analysis_result)


def draw(metric, method, simulate_data_even, simulate_data_odd):
    draw_even_odd_simulation(simulate_data_even,
                             simulate_data_odd,
                             metric,
                             title=metric + "_in_" + method + "_method_" + ("randomized" if randomized else "block") +
                                   "_even_odd_comparison",
                             save_path="participant_simulation_comparison_even_odd/",
                             save=True,
                             show=True,
                             smooth=False)


for metric_name in ["optimal", "optimal_inner", "optimal_outer", "optimal_last"]:
    for index, name in enumerate(model_names):
        averaged_simulate_data = []
        for i in range(NUMBER_OF_REPEAT):
            averaged_simulate_data.append(simulate_analysis_data[name][i].get_data(metric_name))
        averaged_simulate_data = np.array(averaged_simulate_data)
        # plt.subplot(model_number * 100 + 10 + (index + 1))
        draw(metric_name, name,
             np.mean(averaged_simulate_data[:, ::2, :], axis=1),
             np.mean(averaged_simulate_data[:, 1::2, :], axis=1))
    plt.show()
