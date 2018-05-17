from utils.DataSaver import DataSaver
import numpy as np
import matplotlib.pyplot as plt

participant_analysis_data = DataSaver.load_from_file("analysis_result.pkl")
step_data = participant_analysis_data.get_data("step")

mean = np.mean(step_data, axis=1)
sum = np.sum(step_data, axis=1)
plt.hist(mean)
plt.show()
plt.hist(sum)
plt.show()

for i in range(36):
    if sum[i] > 1324:
        print(i)
