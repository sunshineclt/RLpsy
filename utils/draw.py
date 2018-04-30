import matplotlib.pyplot as plt
import numpy as np

from utils.savitzky_golay import savitzky_golay

SAVITZKY_GOLAY_WINDOW = 5
SAVITZKY_GOLAY_ORDER = 1


def draw_metrics(data,
                 data_name,
                 title,
                 extra_data_names=None,
                 smooth=True,
                 trial_length=144,
                 save_npy=False,
                 save_path=None,
                 draw_individual=False,
                 annotate_block_mean=True,
                 show=True):
    data_len = len(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / np.sqrt(data_len)
    if save_npy:
        np.save(save_path, mean)

    if smooth:
        mean = savitzky_golay(mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
    plt.plot(mean, linewidth=3.0, label="mean")
    plt.fill_between(range(0, trial_length), mean - 2 * std, mean + 2 * std, alpha=0.2)

    if extra_data_names and (data_name == "step" or data_name.find("optimal") != -1):
        for extra_data_name in extra_data_names:
            temp_mean = np.load("simulate_transformed_data/" + extra_data_name + "_" + data_name + ".npy")
            if smooth:
                temp_mean = savitzky_golay(temp_mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
            plt.plot(temp_mean, linewidth=3.0, label=extra_data_name.split("_")[0])

    if draw_individual:
        for participant in range(0, data_len):
            if smooth:
                plt.plot(savitzky_golay(data[participant], SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER),
                         linewidth=0.5)
            else:
                plt.plot(data[participant], linewidth=0.5)

    if annotate_block_mean:
        plt.text(0, 0, "mean: %.3f" % mean[:36].mean())
        plt.text(36, 0, "mean: %.3f" % mean[36:72].mean())
        plt.text(72, 0, "mean: %.3f" % mean[72:108].mean())
        plt.text(108, 0, "mean: %.3f" % mean[108:144].mean())

    plt.vlines(36, 0, 30)
    plt.vlines(72, 0, 30)
    plt.vlines(108, 0, 30)
    if data_name == "step":
        plt.ylim([0, 30])
    elif data_name == "time":
        plt.ylim([0, 50])
    elif data_name == "normalized_time":
        plt.ylim([0, 4])
    elif data_name.find("optimal") != -1:
        plt.ylim([0, 1])
    plt.legend()
    plt.title(title)

    if show:
        plt.show()
