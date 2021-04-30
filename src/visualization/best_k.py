import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

report_dir = "./reports"
figure_dir = os.path.join(report_dir, "figures", "diversity")

acc_dict = {"NC_AIforCovid": {"Top 3": 98.9011, "Top 5": 98.9011, "Top 7": 99.4506, "Top 9": 99.4506, "Top 11": 98.3516, "Top 13": 99.4505, "Top 15": 98.9011, "Top 17": 98.3516, "Top 19": 97.2527},
            "NC_COVIDX": {"Top 3": 97.7273, "Top 5": 98.8636, "Top 7": 98.3051, "Top 9": 94.0217, "Top 11": 97.7401, "Top 13": 97.7401, "Top 15": 97.1910, "Top 17": 97.1910, "Top 19": 96.6292},
            "NPC_AIforCovid": {"Top 3": 97.8102, "Top 5": 97.0803, "Top 7": 96.7153, "Top 9": 96.7153, "Top 11": 96.7153, "Top 13": 93.3504, "Top 15": 96.0000, "Top 17": 95.2899, "Top 19": 95.2899},
            "NPC_COVIDX": {"Top 3": 96.0452, "Top 5": 96.6102, "Top 7": 96.0452, "Top 9": 94.9438, "Top 11": 94.3820, "Top 13": 93.8202, "Top 15": 94.3820, "Top 17": 94.3820, "Top 19": 93.8202}}


task_list = ["NC", "NPC"]
task_dict = {"NC": "2 label", "NPC": "3 label"}
train_list = ["AIforCovid", "COVIDX"]
k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
measure_list = ["Q", "Correlation", "Disagreement", "Double-fault"]

# Acc dict
acc_dict = {}

report = pd.read_excel(os.path.join(report_dir, "normal_covid", "reports_top_k.xlsx"))
acc_dict["NC_AIforCovid"] = {"Top %i" % k: report[report["model"] == "Top %i" % k]["mean ACC"].values[0] for k in k_list}

report_dir = "./reports"
report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "reports_top_k.xlsx"))
acc_dict["NC_COVIDX"] = {"Top %i" % k: report[report["model"] == "Top %i" % k]["mean ACC"].values[0] for k in k_list}

report_dir = "./reports"
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "reports_top_k.xlsx"))
acc_dict["NPC_AIforCovid"] = {"Top %i" % k: report[report["model"] == "Top %i" % k]["mean ACC"].values[0] for k in k_list}

report_dir = "./reports"
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "reports_top_k.xlsx"))
acc_dict["NPC_COVIDX"] = {"Top %i" % k: report[report["model"] == "Top %i" % k]["mean ACC"].values[0] for k in k_list}

table_dir = "./data/interim/diversity"


alpha_list = [i/100 for i in range(0, 100)]

for alpha in tqdm(alpha_list):
    for measure in measure_list:
        best_list = []
        for task in task_list:
            diversity = pd.read_csv(os.path.join(table_dir, "%s_%s.csv" % (task, measure)), index_col="model")
            if measure != "Disagreement":
                diversity["mean %s" % measure] = 1-diversity["mean %s" % measure]
            for train in train_list:
                diversity_trian = diversity[(diversity["Train"] == train) & (diversity["Test"] == train)]
                f = alpha * pd.Series(acc_dict["%s_%s" % (task, train)]) + (1 - alpha) * diversity_trian["mean %s" % measure]
                best = f.sort_values(ascending=False).index[0]
                best_list.append(best)
        if len(set(best_list)) == 1:
            print("%s %s: %s" % (measure, alpha, best_list))

# Function (acc(k) − 1)^2 + (div(k) − 1)^2
measure = "Double-fault"
func_data = pd.DataFrame()
acc_data = pd.DataFrame()
div_data = pd.DataFrame()
for task in task_list:
    diversity = pd.read_csv(os.path.join(table_dir, "%s_%s.csv" % (task, measure)), index_col="model")
    diversity["mean %s" % measure] = 1 - diversity["mean %s" % measure]
    for train in train_list:
        diversity_trian = diversity[(diversity["Train"] == train) & (diversity["Test"] == train)]
        f = pd.DataFrame((1 - pd.Series(acc_dict["%s_%s" % (task, train)])/100)**2*0 + (1 - diversity_trian["mean %s" % measure])**2)
        f["Train"] = train
        f["Task"] = task_dict[task]
        func_data = func_data.append(f)

# Plot
sns.lineplot(data=func_data, x=func_data.index, y=0, hue="Train", style="Task", marker="o")
plt.xticks(list(range(len(k_list))), k_list)
plt.xlabel("k", weight="bold")
plt.ylabel("Function", weight="bold")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "optimization_function"), dpi=300)
plt.show()
