import pandas as pd
from statsmodels.formula.api import gls
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("optimal_data_frame.csv")
print(data.head())

group = data.groupby(["block", "condition"])

model = gls("step ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = gls("reaction_time ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = gls("normalized_reaction_time ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())
# model = gls("step ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
# model = gls("reaction_time ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
# model = gls("normalized_reaction_time ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
model = gls("optimal_p ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())
#
# model = gls("optimal_p ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
model = gls("optimal_inner ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())
#
# model = gls("optimal_inner ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
model = gls("optimal_outer ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())
#
# model = gls("optimal_outer ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
model = gls("optimal_last ~ timestep + block", data[data["condition"] == "random"]).fit()
print(model.summary())
#
# model = gls("optimal_last ~ timestep + block", data[data["condition"] == "block"]).fit()
# print(model.summary())
#
# model = gls("optimal_p ~ timestep + block", data[(data["timestep"] <= 10) & (data["condition"] == "random")]).fit()
# print(model.summary())
#
# model = gls("optimal_p ~ timestep + block", data[(data["timestep"] <= 10) & (data["condition"] == "block")]).fit()
# print(model.summary())
#
# g = sns.lmplot(x="optimal_inner", y="optimal_outer", data=data, hue="condition")
# plt.show()
