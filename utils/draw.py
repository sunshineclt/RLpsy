import matplotlib.pyplot as plt
import numpy as np

from utils.savitzky_golay import savitzky_golay
import os
import shutil

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
    plt.fill_between(range(0, trial_length), mean - std, mean + std, alpha=0.2)

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

    plt.vlines(36, 0, 50)
    plt.vlines(72, 0, 50)
    plt.vlines(108, 0, 50)
    plt.xlim([0, 144])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("trial", fontsize=20)
    if data_name == "step":
        plt.ylim([0, 30])
        plt.ylabel("Step", fontsize=20)
    elif data_name == "time":
        plt.ylim([0, 50])
        plt.ylabel("time (s)", fontsize=20)
    elif data_name == "normalized_time":
        plt.ylim([0, 4])
        plt.ylabel("time (s)", fontsize=20)
    elif data_name.find("optimal") != -1:
        plt.ylim([0, 1])
        if data_name.find("inner") != -1:
            plt.ylabel("Inner Optimal Percentage", fontsize=20)
        elif data_name.find("outer") != -1:
            plt.ylabel("Outer Optimal Percentage", fontsize=20)
        elif data_name.find("last") != -1:
            plt.ylabel("Last Optimal Percentage", fontsize=20)
        else:
            plt.ylabel("Optimal Percentage", fontsize=20)
    # plt.legend(loc="upper right")
    # plt.title(title)
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    if show:
        plt.show()


def draw_different_params(data,
                          data_name,
                          title,
                          smooth=True,
                          trial_length=144,
                          show=True,
                          numer_of_participant=50):
    plt.figure(figsize=[40, 6], dpi=80)
    for index1, eta in enumerate(data):
        plt.subplot(1, 6, index1 + 1)
        for index2, tau in enumerate(data[eta]):
            one_param_data = data[eta][tau]
            mean = np.mean(one_param_data, axis=0)
            std = np.std(one_param_data, axis=0) / np.sqrt(numer_of_participant)

            if smooth:
                mean = savitzky_golay(mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
            plt.plot(mean, linewidth=3.0, label="eta: %.1f, tau: %.1f" % (eta, tau))
            plt.fill_between(range(0, trial_length), mean - std, mean + std, alpha=0.2)

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

    plt.tight_layout()
    plt.savefig(title + ".png")
    if show:
        plt.show()


def draw_participant_and_simulation(participant_data,
                                    simulation_data,
                                    data_name,
                                    title=None,
                                    smooth=True,
                                    trial_length=144,
                                    save=True,
                                    save_path=None,
                                    show=False,
                                    multiple_simulation=False):
    if show:
        plt.figure(figsize=[8, 6], dpi=80)
    mean = np.mean(participant_data, axis=0)
    std = np.std(participant_data, axis=0) / np.sqrt(18)
    simulation_mean = np.mean(simulation_data, axis=0)
    simulation_std = np.std(simulation_data, axis=0)
    if not multiple_simulation:
        simulation_std /= np.sqrt(18)
    if smooth:
        mean = savitzky_golay(mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
        simulation_mean = savitzky_golay(simulation_mean, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
    plt.plot(simulation_mean, linewidth=3.0, label="simulation")
    lower = simulation_mean - 1.96 * simulation_std
    upper = simulation_mean + 1.96 * simulation_std
    plt.fill_between(range(0, trial_length),
                     lower, upper,
                     alpha=0.2)
    plt.plot(mean, linewidth=3.0, label="participant")

    if multiple_simulation:
        count = [0, 0, 0, 0]
        for i in range(trial_length):
            if mean[i] < lower[i] or mean[i] > upper[i]:
                plt.vlines(i, 0, 0.1, colors="red")
                count[i // 36] += 1
        coord = 18
        for i in range(4):
            plt.text(coord, 0.2, str(count[i]))
            coord += 36
    else:
        plt.fill_between(range(0, trial_length),
                         mean - std, mean + std,
                         alpha=0.2)

    plt.vlines(36, 0, 50)
    plt.vlines(72, 0, 50)
    plt.vlines(108, 0, 50)
    plt.xlim([0, 144])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("trial", fontsize=20)
    if data_name == "step":
        plt.ylim([0, 30])
        plt.ylabel("Step", fontsize=20)
    elif data_name == "time":
        plt.ylim([0, 50])
        plt.ylabel("time (s)", fontsize=20)
    elif data_name == "normalized_time":
        plt.ylim([0, 4])
        plt.ylabel("time (s)", fontsize=20)
    elif data_name.find("optimal") != -1:
        plt.ylim([0, 1.2])
        if data_name.find("inner") != -1:
            plt.ylabel("Inner Optimal P", fontsize=20)
        elif data_name.find("outer") != -1:
            plt.ylabel("Outer Optimal P", fontsize=20)
        elif data_name.find("last") != -1:
            plt.ylabel("Last Optimal P", fontsize=20)
        else:
            plt.ylabel("Optimal P", fontsize=20)
    plt.legend(loc="upper right")
    if title:
        plt.title(title)
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    if save:
        plt.savefig(save_path + title + ".jpg")
    if show:
        plt.show()


def draw_even_odd_simulation(simulation_data_even,
                             simulation_data_odd,
                             data_name,
                             title=None,
                             smooth=True,
                             trial_length=144,
                             save=True,
                             save_path=None,
                             show=False):
    if show:
        plt.figure(figsize=[8, 6], dpi=80)
    simulation_mean_even = np.mean(simulation_data_even, axis=0)
    simulation_std_even = np.std(simulation_data_even, axis=0)
    simulation_mean_odd = np.mean(simulation_data_odd, axis=0)
    simulation_std_odd = np.std(simulation_data_odd, axis=0)
    if smooth:
        simulation_mean_even = savitzky_golay(simulation_mean_even, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)
        simulation_mean_odd = savitzky_golay(simulation_mean_odd, SAVITZKY_GOLAY_WINDOW, SAVITZKY_GOLAY_ORDER)

    plt.plot(simulation_mean_even, linewidth=3.0, label="even")
    lower = simulation_mean_even - 1.96 * simulation_std_even
    upper = simulation_mean_even + 1.96 * simulation_std_even
    plt.fill_between(range(0, trial_length),
                     lower, upper,
                     alpha=0.2)
    plt.plot(simulation_mean_odd, linewidth=3.0, label="odd")
    lower = simulation_mean_odd - 1.96 * simulation_std_odd
    upper = simulation_mean_odd + 1.96 * simulation_std_odd
    plt.fill_between(range(0, trial_length),
                     lower, upper,
                     alpha=0.2)

    plt.vlines(36, 0, 50)
    plt.vlines(72, 0, 50)
    plt.vlines(108, 0, 50)
    plt.xlim([0, 144])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("trial", fontsize=20)
    if data_name == "step":
        plt.ylim([0, 30])
        plt.ylabel("Step", fontsize=20)
    elif data_name.find("optimal") != -1:
        plt.ylim([0, 1.2])
        if data_name.find("inner") != -1:
            plt.ylabel("Inner Optimal P", fontsize=20)
        elif data_name.find("outer") != -1:
            plt.ylabel("Outer Optimal P", fontsize=20)
        elif data_name.find("last") != -1:
            plt.ylabel("Last Optimal P", fontsize=20)
        else:
            plt.ylabel("Optimal P", fontsize=20)
    plt.legend(loc="upper right")
    if title:
        plt.title(title)
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    if save:
        plt.savefig(save_path + title + ".jpg")
    if show:
        plt.show()
