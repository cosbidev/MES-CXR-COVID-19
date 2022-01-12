from scipy.stats import pearsonr
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pylab import meshgrid
import scipy

report_dir = "./reports/normal_pneumonia_covid_other/plot"
color_dict = {"3":"C0","5":"C1","7":"C2","9":"C3","11":"C4","13":"C5","15":"C6","17":"C7","19":"C8"}

# Rank Correlation

rank_single_cv = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
rank_single_ev = [2, 1, 3, 6, 7, 5, 4, 8, 11, 13, 9, 12, 10, 16, 17, 20, 15, 18, 14, 19]
rank_mes_cv = [1, 4, 2, 3, 8, 9, 7, 6, 5]
rank_mes_ev = [1, 5, 3, 2, 9, 8, 6, 7, 4]

print(pearsonr(rank_single_cv, rank_single_ev))
print(pearsonr(rank_mes_cv, rank_mes_ev))


# Accuracy vs Diversity

data = pd.read_excel("./reports/normal_pneumonia_covid_other/optimization.xlsx", index_col="model")
data["k"] = data["k"].astype("str")
data = data.rename(columns={"mean Double-fault": "Diversity", "mean ACC": "Accuracy"})

sns.scatterplot(data=data, x="Diversity", y="Accuracy", hue="k", alpha=0.5, legend=False, palette=color_dict)
plt.savefig(os.path.join(report_dir, "div_acc_all.png"), dpi=1000)
plt.show()


data["F"] = data["Diversity"]+data["Accuracy"]
group = data.groupby(['k'], sort=False)['F'].idxmax()
data_group = data.loc[group]
data_group.loc[group.loc["3"], 'Accuracy'] = 0.806

def func(x1,x2):
 return ((1-x1)**2* + (1-x2)**2)

x = np.arange(data_group["Diversity"].min(), data_group["Diversity"].max(),0.0001)
y = np.arange(data_group["Accuracy"].min(), data_group["Accuracy"].max(),0.0001)
X, Y = meshgrid(x, y) # grid of point
Z = func(X, Y) # evaluation of the function on the grid

f, ax = plt.subplots()
cnt = ax.contour(Z, cmap=matplotlib.cm.Reds_r, vmin=abs(Z).min(), vmax=abs(Z).max(),
                 extent=[data_group["Diversity"].min()-0.001, data_group["Diversity"].max()+0.001, data_group["Accuracy"].min()-0.001, data_group["Accuracy"].max()+0.001],
                 levels=10, zorder=1) #counterf (per heatmap)
sns.scatterplot(data=data_group, x="Diversity", y="Accuracy", hue="k", s=100, linewidth=1, edgecolor="k", zorder=2, legend=False, palette=color_dict)
plt.savefig(os.path.join(report_dir, "div_acc_best.png"), dpi=1000)
plt.xlim((0.8348558926188276, 0.8524580381890977))
plt.ylim((0.7895287958115184, 0.812565445026178))
plt.show()


plt.figure(figsize=(0.8, 2.3))
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
plt.legend(markers, color_dict.keys(), numpoints=1, title=r"$k$", loc="center")
plt.axis('off')
plt.savefig(os.path.join(report_dir, "div_acc_legend.png"), dpi=1000)
plt.show()

# Drop Plot

results_1 = pd.read_excel("./reports/normal_pneumonia_covid_other/report.xlsx", sheet_name="Single 10 CV", index_col=0)
results_1["Eval"] = "CV"
results_1["Model"] = "CNN"
results_2 = pd.read_excel("./reports/normal_pneumonia_covid_other/report.xlsx", sheet_name="Single External CV", index_col=0)
results_2["Eval"] = "External"
results_2["Model"] = "CNN"
results_3 = pd.read_excel("./reports/normal_pneumonia_covid_other/report.xlsx", sheet_name="Pareto 10 CV", index_col=0)
results_3["Eval"] = "CV"
results_3["Model"] = "MES"
results_4 = pd.read_excel("./reports/normal_pneumonia_covid_other/report.xlsx", sheet_name="Pareto External CV", index_col=0)
results_4["Eval"] = "External"
results_4["Model"] = "MES"
results = pd.concat([results_1, results_2, results_3, results_4])


sns.boxplot(data=results, x="Eval", y="mean ACC", hue="Model")
plt.xlabel(None)
plt.ylabel("Accuracy (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "accuracy.png"), dpi=1000)
plt.show()


# T-test
print(scipy.stats.ttest_ind(results_1['mean ACC'], results_2['mean ACC']))
print(scipy.stats.ttest_ind(results_3['mean ACC'], results_4['mean ACC']))
print(scipy.stats.ttest_ind(results_1['mean ACC'], results_3['mean ACC']))
print(scipy.stats.ttest_ind(results_2['mean ACC'], results_4['mean ACC']))
print(scipy.stats.ttest_ind(results_1['mean ACC'], results_4['mean ACC']))
