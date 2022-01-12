import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
from tqdm import tqdm
import src.utils.util_model as util_model
import src.utils.util_data as util_data
import pandas as pd
import os
import collections
import numpy as np
import itertools

task = "normal_pneumonia_covid_other"
categories = sorted(["Normal", "Pneumonia", "COVID", "other"])

cv = 10
fold_list = list(range(cv))
fold_list = [0]

# Location of data
source = "../data"

report_dir = os.path.join("./reports", task)

model_dir = os.path.join("./models", task)

report_file_topk = os.path.join(report_dir, "report_topk.xlsx")
report_file_randomk = os.path.join(report_dir, "reports_random_k.xlsx")

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
top_list = [20]#, 10]

report = pd.read_excel(os.path.join(report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="mean ACC", ascending=False)

data_file = os.path.join("./data/processed", task, "data.xlsx")
data = pd.read_excel(data_file, index_col=0, dtype=list)

boxes = {}
for row in data.iterrows():
    boxes[row[0]] = eval(row[1]["all"])

batch_size = 1

# Generators
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

# Partition
partition = {"test": data.index.tolist()}
labels = data["label"].to_dict()

# Generators
image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

idx_to_class = {v: k for k, v in image_dataset.class_to_idx.items()}

# TOP K
print("***************TOP K***************")

results_frame = {}
acc_cols = []
acc_cat_cols = collections.defaultdict(lambda: [])
model_name_list = []
for k in k_list:
    model_name_list += ["Top %s" % k]
for fold in fold_list:
    acc_col = str(fold) + " ACC"
    acc_cols.append(acc_col)
    results_frame[acc_col] = []
    for cat in categories:
        cat_col = str(fold) + " ACC " + cat
        acc_cat_cols[cat].append(cat_col)
        results_frame[cat_col] = []
acc_cat_cols = dict(acc_cat_cols)


for fold in fold_list:
    print(fold)
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)
    for k in k_list:
        # top k models
        top_k_models = report["model"].iloc[:k].to_list()
        results = pd.DataFrame()
        results["Prediction"] = test[top_k_models].mode(axis=1)[0]
        results["True"] = test["True"]
        results["Acc"] = (results["Prediction"] == results["True"]) * 1
        results_frame["%s ACC" % fold].append(results["Acc"].sum() / len(results["Acc"]) * 100)
        for cat in results["True"].unique():
            cat_comb_results = results[results["True"] == cat]
            results_frame["%s ACC %s" % (fold, categories[cat])].append(cat_comb_results["Acc"].sum() / len(cat_comb_results["Acc"]) * 100)


results_frame = pd.DataFrame.from_dict(dict(results_frame))
for cat in categories[::-1]:
    results_frame.insert(loc=0, column='std ACC ' + cat, value=results_frame[acc_cat_cols[cat]].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC ' + cat, value=results_frame[acc_cat_cols[cat]].mean(axis=1))
results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file_topk, index=False)

# Random K
print("***************RANDOM K***************")

results_frame = {}
model_name_list = []
for k in k_list:
    for top in top_list:
        model_name_list.append("Random %s Top %s" % (k, top))
for fold in fold_list:
    results_frame["%s ACC mean" % str(fold)] = []
    results_frame["%s ACC std" % str(fold)] = []
    for cat in categories:
        results_frame["%s ACC %s mean" % (str(fold), cat)] = []
        results_frame["%s ACC %s std" % (str(fold), cat)] = []

for fold in fold_list:
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)

    for top in top_list:
        model_list = report["model"].iloc[:top].to_list()

        for k in k_list:
            k_results = collections.defaultdict(lambda: [])
            for comb in tqdm(itertools.combinations(model_list, k)):
                comb_results = pd.DataFrame()
                comb_results["Prediction"] = test[list(comb)].mode(axis=1)[0]
                comb_results["True"] = test["True"]
                comb_results["Acc"] = (comb_results["Prediction"] == comb_results["True"])*1
                k_results["ACC"].append(comb_results["Acc"].sum() / len(comb_results["Acc"]) * 100)
                for cat in comb_results["True"].unique():
                    cat_comb_results = comb_results[comb_results["True"] == cat]
                    k_results["ACC %s" % categories[cat]].append(cat_comb_results["Acc"].sum() / len(cat_comb_results["Acc"]) * 100)
            for a, l in k_results.items():
                results_frame["%s %s mean" % (str(fold), a)].append(np.mean(l))
                results_frame["%s %s std" % (str(fold), a)].append(np.std(l))

results_frame = pd.DataFrame.from_dict(dict(results_frame))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file_randomk, index=False)
