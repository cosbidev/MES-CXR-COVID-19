import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Line plot (K vs Accuracy) NC, NPC

report_dir = "./reports"
figure_dir = os.path.join(report_dir, "figures", "accuracy")

k_list = [3, 5, 7, 9]

acc_data = pd.DataFrame()

# Normal vs COVID-19
# Train AIforCovid
# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "normal_covid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "AIforCovid"
acc_data = acc_data.append(acc)

# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "normal_covid", "covidx", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "COVIDX"
acc_data = acc_data.append(acc)

# Test Brixia
report = pd.read_excel(os.path.join(report_dir, "normal_covid", "brixia", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "Brixia"
acc_data = acc_data.append(acc)

# Train COVIDX
# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "COVIDX"
acc_data = acc_data.append(acc)

# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "aiforcovid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "AIforCovid"
acc_data = acc_data.append(acc)

# Test Brixia
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "brixia", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "Brixia"
acc_data = acc_data.append(acc)

# Plot
sns.lineplot(data=acc_data, x="K", y="mean ACC", hue="Train", style="Test", marker="o")
plt.xticks(k_list)
plt.xlabel("K", weight="bold")
plt.ylabel("ACC", weight="bold")
plt.title("Healthy vs Covid-19", weight="bold")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "NC"), dpi=300)
plt.show()


# Normal vs Pneumonia vs COVID-19
acc_data = pd.DataFrame()
# Train AIforCovid
# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "AIforCovid"
acc_data = acc_data.append(acc)

# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "covidx", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "COVIDX"
acc_data = acc_data.append(acc)

# Test Brixia
report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "brixia", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "Brixia"
acc_data = acc_data.append(acc)

# Train COVIDX
# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "COVIDX"
acc_data = acc_data.append(acc)

# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "aiforcovid", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "AIforCovid"
acc_data = acc_data.append(acc)

# Test Brixia
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "brixia", "reports_top_k.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean ACC"]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "Brixia"
acc_data = acc_data.append(acc)

# Plot
sns.lineplot(data=acc_data, x="K", y="mean ACC", hue="Train", style="Test", marker="o")
plt.xticks(k_list)
plt.xlabel("K", weight="bold")
plt.ylabel("ACC", weight="bold")
plt.title("Healthy vs Pneumonia vs Covid-19", weight="bold")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "NPC"), dpi=300)
plt.show()
