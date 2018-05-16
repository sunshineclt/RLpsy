from utils.DataSaver import DataSaver
from utils.draw import draw_participant_and_simulation

participant_analysis_data = DataSaver.load_from_file("analysis_result.pkl")
simulate_analysis_data = DataSaver.load_from_file("MF_simulate_analysis_result.pkl")


def draw(metric, randomized=True):
    if randomized:
        draw_participant_and_simulation(participant_analysis_data.get_data(metric)[::2, :],
                                        simulate_analysis_data.get_data(metric)[::2, :],
                                        metric,
                                        metric + " in participant and simulation",
                                        save_path="participant_simulation_comparison_all/")
    else:
        draw_participant_and_simulation(participant_analysis_data.get_data(metric)[1::2, :],
                                        simulate_analysis_data.get_data(metric)[1::2, :],
                                        metric,
                                        metric + " in participant and simulation",
                                        save_path="participant_simulation_comparison_all/")


draw("optimal", randomized=True)
draw("optimal_inner", randomized=True)
draw("optimal_outer", randomized=True)
draw("optimal_last", randomized=True)
draw("optimal", randomized=False)
draw("optimal_inner", randomized=False)
draw("optimal_outer", randomized=False)
draw("optimal_last", randomized=False)
