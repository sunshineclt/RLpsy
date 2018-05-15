from utils.draw import draw_metrics
from utils.DataSaver import DataSaver


def transform_and_plot(data, data_name):

    randomized_data = data[::2, :]
    draw_metrics(randomized_data,
                 data_name,
                 "%s under %s condition in participants" % (data_name, "randomized"),
                 # extra_data_names=["MF_randomized", "optimal_randomized", "random_randomized", "MB_randomized"],
                 smooth=True,
                 save_npy=False,
                 draw_individual=True,
                 annotate_block_mean=False)

    block_data = data[1::2, :]
    draw_metrics(block_data,
                 data_name,
                 "%s under %s condition in participants" % (data_name, "block"),
                 # extra_data_names=["MF_block", "optimal_block", "random_block", "MB_block"],
                 smooth=True,
                 save_npy=False,
                 draw_individual=True,
                 annotate_block_mean=False)


if __name__ == "__main__":
    saver = DataSaver.load_from_file("analysis_result.pkl")
    for metric in saver.data_names:
        transform_and_plot(saver.get_data(metric), metric)
