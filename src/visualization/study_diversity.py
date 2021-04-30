import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Line plot (K vs Accuracy) NC, NPC

report_dir = "./reports"
figure_dir = os.path.join(report_dir, "figures", "diversity")
table_dir = "./data/interim/diversity"

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
measure_list = ["Q", "Correlation", "Disagreement", "Double-fault"]

#NC
for measure in measure_list:

    acc_data = pd.DataFrame()

    # Normal vs COVID-19
    # Train AIforCovid
    # Test AIforCovid
    report = pd.read_excel(os.path.join(report_dir, "normal_covid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "AIforCovid"
    acc_data = acc_data.append(acc)

    # Test COVIDX
    report = pd.read_excel(os.path.join(report_dir, "normal_covid", "covidx", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "COVIDX"
    acc_data = acc_data.append(acc)

    # Test Brixia
    report = pd.read_excel(os.path.join(report_dir, "normal_covid", "brixia", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "Brixia"
    acc_data = acc_data.append(acc)

    # Train COVIDX
    # Test COVIDX
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "COVIDX"
    acc_data = acc_data.append(acc)

    # Test AIforCovid
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "aiforcovid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "AIforCovid"
    acc_data = acc_data.append(acc)

    # Test Brixia
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_covid", "brixia", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "Brixia"
    acc_data = acc_data.append(acc)

    # Plot
    sns.lineplot(data=acc_data, x="K", y="mean %s" % measure, hue="Train", style="Test", marker="o")
    plt.xticks(k_list)
    plt.xlabel("K", weight="bold")
    plt.ylabel(measure, weight="bold")
    plt.title("Healthy vs Covid-19", weight="bold")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "diversity_NC_%s" % measure), dpi=300)
    plt.show()

    # Save
    acc_data.to_csv(os.path.join(table_dir, "NC_%s.csv" % measure))

#NPC
for measure in measure_list:
    acc_data = pd.DataFrame()

    # Normal vs COVID-19
    # Train AIforCovid
    # Test AIforCovid
    report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "AIforCovid"
    acc_data = acc_data.append(acc)

    # Test COVIDX
    report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "covidx", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "COVIDX"
    acc_data = acc_data.append(acc)

    # Test Brixia
    report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "brixia", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "AIforCovid"
    acc["Test"] = "Brixia"
    acc_data = acc_data.append(acc)

    # Train COVIDX
    # Test COVIDX
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "COVIDX"
    acc_data = acc_data.append(acc)

    # Test AIforCovid
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "aiforcovid", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "AIforCovid"
    acc_data = acc_data.append(acc)

    # Test Brixia
    report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "brixia", "report_diversity_topk.xlsx"))
    acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
    acc["K"] = k_list
    acc["Train"] = "COVIDX"
    acc["Test"] = "Brixia"
    acc_data = acc_data.append(acc)

    # Plot
    sns.lineplot(data=acc_data, x="K", y="mean %s" % measure, hue="Train", style="Test", marker="o")
    plt.xticks(k_list)
    plt.xlabel("K", weight="bold")
    plt.ylabel(measure, weight="bold")
    plt.title("Healthy vs Pneumonia vs Covid-19", weight="bold")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "diversity_NPC_%s" % measure), dpi=300)
    plt.show()

    # Save
    acc_data.to_csv(os.path.join(table_dir, "NPC_%s.csv" % measure))



# DOuble falut
measure = 'Double-fault'
acc_data = pd.DataFrame()

# Normal vs COVID-19
# Train AIforCovid
# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "normal_covid", "report_diversity_topk.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "AIforCovid"
acc["Task"] = "non-Covid-19 vs Covid-19"
acc_data = acc_data.append(acc)

# Train COVIDX
# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "report_diversity_topk.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "COVIDX"
acc["Task"] = "non-Covid-19 vs Covid-19"
acc_data = acc_data.append(acc)

# Normal vs COVID-19
# Train AIforCovid
# Test AIforCovid
report = pd.read_excel(os.path.join(report_dir, "normal_pneumonia_covid", "report_diversity_topk.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
acc["K"] = k_list
acc["Train"] = "AIforCovid"
acc["Test"] = "AIforCovid"
acc["Task"] = "Healthy vs Pneumonia vs Covid-19"
acc_data = acc_data.append(acc)

# Train COVIDX
# Test COVIDX
report = pd.read_excel(os.path.join(report_dir, "covidx_normal_pneumonia_covid", "report_diversity_topk.xlsx"))
acc = report[report["model"].isin(["Top %i" % k for k in k_list])][["model", "mean %s" % measure]]
acc["K"] = k_list
acc["Train"] = "COVIDX"
acc["Test"] = "COVIDX"
acc["Task"] = "Healthy vs Pneumonia vs Covid-19"
acc_data = acc_data.append(acc)

# Plot
sns.lineplot(data=acc_data, x="K", y="mean %s" % measure, hue="Train", style="Task", marker="o")
plt.xticks(k_list)
plt.xlabel("K", weight="bold")
plt.ylabel(measure, weight="bold")
plt.title("Healthy vs Pneumonia vs Covid-19", weight="bold")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "diversity_NPC_%s" % measure), dpi=300)
plt.show()

# Save
#aacc_data.to_csv(os.path.join(table_dir, "NPC_%s.csv" % measure))