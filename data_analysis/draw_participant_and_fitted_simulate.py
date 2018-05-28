from utils.DataSaver import DataSaver
from utils.draw import draw_participant_and_simulation

import matplotlib.pyplot as plt

participant_analysis_data = DataSaver.load_from_file("analysis_result.pkl")
simulate_analysis_data = {"MF": DataSaver.load_from_file("MF_simulate_analysis_result.pkl"),
                          "MB": DataSaver.load_from_file("MB_simulate_analysis_result.pkl")}
MF_simulate_analysis_data = DataSaver.load_from_file("MF_simulate_analysis_result.pkl")
MB_simulate_analysis_data = DataSaver.load_from_file("MB_simulate_analysis_result.pkl")


def draw(metric, method, randomized=True):
    if randomized:
        draw_participant_and_simulation(participant_analysis_data.get_data(metric)[::2, :],
                                        simulate_analysis_data[method].get_data(metric)[::2, :],
                                        metric,
                                        title=metric + "_in_" + method + "_method_randomized_condition",
                                        save_path="participant_simulation_comparison_all/",
                                        save=True,
                                        show=True)
    else:
        draw_participant_and_simulation(participant_analysis_data.get_data(metric)[1::2, :],
                                        simulate_analysis_data[method].get_data(metric)[1::2, :],
                                        metric,
                                        title=metric + "_in_" + method + "_method_block_condition",
                                        save_path="participant_simulation_comparison_all/",
                                        save=True,
                                        show=True)


for metric_name in ["optimal", "optimal_inner", "optimal_outer", "optimal_last"]:
    # plt.figure(figsize=[10, 8], dpi=80)
    # plt.subplot(221)
    draw(metric_name, "MF", randomized=True)
    # plt.subplot(222)
    draw(metric_name, "MB", randomized=True)
    # plt.subplot(223)
    draw(metric_name, "MF", randomized=False)
    # plt.subplot(224)
    draw(metric_name, "MB", randomized=False)
    # plt.tight_layout()
    # plt.show()
