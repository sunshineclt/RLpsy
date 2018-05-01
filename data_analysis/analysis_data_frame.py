import pandas as pd
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("optimal_data_frame.csv")
print(data.head())

group = data.groupby(["block", "condition"])


model = ols("optimal_p ~ step + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = ols("optimal_p ~ step + block", data[data["condition"] == "block"]).fit()
print(model.summary())

model = ols("optimal_inner ~ step + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = ols("optimal_inner ~ step + block", data[data["condition"] == "block"]).fit()
print(model.summary())

model = ols("optimal_outer ~ step + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = ols("optimal_outer ~ step + block", data[data["condition"] == "block"]).fit()
print(model.summary())

model = ols("optimal_last ~ step + block", data[data["condition"] == "random"]).fit()
print(model.summary())

model = ols("optimal_last ~ step + block", data[data["condition"] == "block"]).fit()
print(model.summary())

model = ols("optimal_p ~ step + block", data[(data["step"] <= 10) & (data["condition"] == "random")]).fit()
print(model.summary())

model = ols("optimal_p ~ step + block", data[(data["step"] <= 10) & (data["condition"] == "block")]).fit()
print(model.summary())

g = sns.lmplot(x="optimal_inner", y="optimal_outer", data=data, hue="condition")
plt.show()
