import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import pandas as pd
import os
import collections
import numpy as np
import itertools

task = "covidx_normal_pneumonia_covid"

cv = 10
fold_list = list(range(cv))
fold_list = [0]

report_dir = os.path.join("./reports", task)
#report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

report_file_divesity_topk = os.path.join(report_dir, "report_diversity_topk.xlsx")

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
measure_list = ["Q", "Correlation", "Disagreement", "Double-fault"]

report = pd.read_excel(os.path.join(report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="0 ACC", ascending=False)

# TOP K
results_frame = {}
acc_cat_cols = collections.defaultdict(lambda: [])
model_name_list = []
for k in k_list:
    model_name_list += ["Top %s" % k]
for fold in fold_list:
    for measure in measure_list:
        measure_col = "%i %s" % (fold, measure)
        acc_cat_cols[measure].append(measure_col)
        results_frame[measure_col] = []
acc_cat_cols = dict(acc_cat_cols)


for fold in fold_list:
    print(fold)
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)
    for k in k_list:
        # top k models
        top_k_models = report["model"].iloc[:k].to_list()
        measure_dict = collections.defaultdict(lambda: [])
        for model_1, model_2 in itertools.combinations(top_k_models, 2):
            N_11 = len(test[(test[model_1] == test["True"]) & (test[model_2] == test["True"])])
            N_10 = len(test[(test[model_1] == test["True"]) & (test[model_2] != test["True"])])
            N_01 = len(test[(test[model_1] != test["True"]) & (test[model_2] == test["True"])])
            N_00 = len(test[(test[model_1] != test["True"]) & (test[model_2] != test["True"])])
            # measures
            measure_dict["Q"].append(((N_11 * N_00) - (N_01 * N_10)) / ((N_11 * N_00) + (N_01 * N_10)))
            measure_dict["Correlation"].append(((N_11 * N_00) - (N_01 * N_10)) / np.sqrt((N_11 + N_10) * (N_01 + N_00) * (N_11 + N_01) * (N_10 + N_00)))
            measure_dict["Disagreement"].append((N_01 + N_10) / (N_11 + N_10 + N_01 + N_00))
            measure_dict["Double-fault"].append(N_00 / (N_11 + N_10 + N_01 + N_00))
        for measure in measure_dict:
            results_frame["%i %s" % (fold, measure)].append((2 / (k * (k-1))) * sum(measure_dict[measure]))



results_frame = pd.DataFrame.from_dict(dict(results_frame))
for measure in measure_list[::-1]:
    results_frame.insert(loc=0, column='std ' + measure, value=results_frame[acc_cat_cols[measure]].std(axis=1))
    results_frame.insert(loc=0, column='mean ' + measure, value=results_frame[acc_cat_cols[measure]].mean(axis=1))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file_divesity_topk, index=False)


#AIforCovid
report_dir = os.path.join("./reports", task, "aiforcovid")
#report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

trained_model_report_dir = os.path.join("./reports", task)
#trained_model_report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

report_file_divesity_topk = os.path.join(report_dir, "report_diversity_topk.xlsx")

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
measure_list = ["Q", "Correlation", "Disagreement", "Double-fault"]

report = pd.read_excel(os.path.join(trained_model_report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="0 ACC", ascending=False)

# TOP K
results_frame = {}
acc_cat_cols = collections.defaultdict(lambda: [])
model_name_list = []
for k in k_list:
    model_name_list += ["Top %s" % k]
for fold in fold_list:
    for measure in measure_list:
        measure_col = "%i %s" % (fold, measure)
        acc_cat_cols[measure].append(measure_col)
        results_frame[measure_col] = []
acc_cat_cols = dict(acc_cat_cols)


for fold in fold_list:
    print(fold)
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)
    for k in k_list:
        # top k models
        top_k_models = report["model"].iloc[:k].to_list()
        measure_dict = collections.defaultdict(lambda: [])
        for model_1, model_2 in itertools.combinations(top_k_models, 2):
            N_11 = len(test[(test[model_1] == test["True"]) & (test[model_2] == test["True"])])
            N_10 = len(test[(test[model_1] == test["True"]) & (test[model_2] != test["True"])])
            N_01 = len(test[(test[model_1] != test["True"]) & (test[model_2] == test["True"])])
            N_00 = len(test[(test[model_1] != test["True"]) & (test[model_2] != test["True"])])
            # measures
            measure_dict["Q"].append(((N_11 * N_00) - (N_01 * N_10)) / ((N_11 * N_00) + (N_01 * N_10)))
            measure_dict["Correlation"].append(((N_11 * N_00) - (N_01 * N_10)) / np.sqrt((N_11 + N_10) * (N_01 + N_00) * (N_11 + N_01) * (N_10 + N_00)))
            measure_dict["Disagreement"].append((N_01 + N_10) / (N_11 + N_10 + N_01 + N_00))
            measure_dict["Double-fault"].append(N_00 / (N_11 + N_10 + N_01 + N_00))
        for measure in measure_dict:
            results_frame["%i %s" % (fold, measure)].append((2 / (k * (k-1))) * sum(measure_dict[measure]))



results_frame = pd.DataFrame.from_dict(dict(results_frame))
for measure in measure_list[::-1]:
    results_frame.insert(loc=0, column='std ' + measure, value=results_frame[acc_cat_cols[measure]].std(axis=1))
    results_frame.insert(loc=0, column='mean ' + measure, value=results_frame[acc_cat_cols[measure]].mean(axis=1))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file_divesity_topk, index=False)


#Brixia
report_dir = os.path.join("./reports", task, "brixia")
#report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

trained_model_report_dir = os.path.join("./reports", task)
#trained_model_report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

report_file_divesity_topk = os.path.join(report_dir, "report_diversity_topk.xlsx")

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
measure_list = ["Q", "Correlation", "Disagreement", "Double-fault"]

report = pd.read_excel(os.path.join(trained_model_report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="0 ACC", ascending=False)

# TOP K
results_frame = {}
acc_cat_cols = collections.defaultdict(lambda: [])
model_name_list = []
for k in k_list:
    model_name_list += ["Top %s" % k]
for fold in fold_list:
    for measure in measure_list:
        measure_col = "%i %s" % (fold, measure)
        acc_cat_cols[measure].append(measure_col)
        results_frame[measure_col] = []
acc_cat_cols = dict(acc_cat_cols)


for fold in fold_list:
    print(fold)
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)
    for k in k_list:
        # top k models
        top_k_models = report["model"].iloc[:k].to_list()
        measure_dict = collections.defaultdict(lambda: [])
        for model_1, model_2 in itertools.combinations(top_k_models, 2):
            N_11 = len(test[(test[model_1] == test["True"]) & (test[model_2] == test["True"])])
            N_10 = len(test[(test[model_1] == test["True"]) & (test[model_2] != test["True"])])
            N_01 = len(test[(test[model_1] != test["True"]) & (test[model_2] == test["True"])])
            N_00 = len(test[(test[model_1] != test["True"]) & (test[model_2] != test["True"])])
            # measures
            measure_dict["Q"].append(((N_11 * N_00) - (N_01 * N_10)) / ((N_11 * N_00) + (N_01 * N_10)))
            measure_dict["Correlation"].append(((N_11 * N_00) - (N_01 * N_10)) / np.sqrt((N_11 + N_10) * (N_01 + N_00) * (N_11 + N_01) * (N_10 + N_00)))
            measure_dict["Disagreement"].append((N_01 + N_10) / (N_11 + N_10 + N_01 + N_00))
            measure_dict["Double-fault"].append(N_00 / (N_11 + N_10 + N_01 + N_00))
        for measure in measure_dict:
            results_frame["%i %s" % (fold, measure)].append((2 / (k * (k-1))) * sum(measure_dict[measure]))



results_frame = pd.DataFrame.from_dict(dict(results_frame))
for measure in measure_list[::-1]:
    results_frame.insert(loc=0, column='std ' + measure, value=results_frame[acc_cat_cols[measure]].std(axis=1))
    results_frame.insert(loc=0, column='mean ' + measure, value=results_frame[acc_cat_cols[measure]].mean(axis=1))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file_divesity_topk, index=False)
