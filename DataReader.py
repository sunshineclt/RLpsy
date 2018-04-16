import csv
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rawFile = open("raw_data/0_2018_Apr_09_2222.csv", "r")
    reader = csv.DictReader(rawFile, delimiter="#")
    all_data = []
    for row in reader:
        trial = row["trial_data"]
        if trial != "--":
            transformed_data = json.loads(trial)
            # print(transformed_data)
            all_data.append(transformed_data)

    steps = []
    times = []
    average = []
    for trial in all_data:
        steps.append(len(trial))
        times.append(0)
        for step in trial:
            times[-1] += step[3]
        average.append(times[-1] / steps[-1])

    plt.plot(steps)
    plt.show()
    plt.plot(times)
    plt.show()
    plt.plot(average)
    plt.show()
