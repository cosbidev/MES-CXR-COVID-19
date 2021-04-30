import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
from tqdm import tqdm
import pandas as pd
import os
import collections
import itertools
import numpy as np

task = "covidx_normal_covid"
categories = sorted(["Normal", "COVID"])

cv = 10
fold_list = list(range(cv))
fold_list = [0]

report_dir = os.path.join("./reports", task)
#report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

report_file = os.path.join(report_dir, "optimization.xlsx")

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
top_list = [20]#, 10]
measures = ["Q", "Correlation", "Disagreement", "Double-fault"]
# iterations: 1140+15504+77520+167960+167960+77520+1140+15504+20
# # hours: (1140+15504+77520+167960+167960+77520+1140+15504+20)/15/60/60

report = pd.read_excel(os.path.join(report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="0 ACC", ascending=False)

# All K combinations
results_frame = {}
acc_cols = []
acc_cat_cols = collections.defaultdict(lambda: [])
div_cols = collections.defaultdict(lambda: [])
for fold in fold_list:
    acc_col = "%s ACC" % str(fold)
    acc_cols.append(acc_col)
    results_frame[acc_col] = []
    for div in measures:
        div_col = "%s %s" % (str(fold), div)
        div_cols[div].append(div_col)
        results_frame[div_col] = []
    for cat in categories:
        cat_col = "%s ACC %s" % (str(fold), cat)
        acc_cat_cols[cat].append(cat_col)
        results_frame[cat_col] = []


model_name_list = []
k_model_list = []
for fold in fold_list:
    test = pd.read_excel(os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold), index_col=0)

    for top in top_list:
        model_list = report["model"].iloc[:top].to_list()

        for k in k_list:
            print("k=%i" % k)

            for comb in tqdm(itertools.combinations(model_list, k)):
                model_name_list.append(";".join(comb))
                k_model_list.append(k)

                comb_results = (test[list(comb)].mode(axis=1)[0] == test["True"])*1
                results_frame["%s ACC" % fold].append(comb_results.sum() / len(comb_results))
                for cat in test["True"].unique():
                    cat_comb_results = comb_results[test["True"] == cat]
                    results_frame["%s ACC %s" % (fold, categories[cat])].append(cat_comb_results.sum() / len(cat_comb_results))
                measure_dict = collections.defaultdict(lambda: [])
                for model_1, model_2 in itertools.combinations(comb, 2):
                    N_11 = len(test[(test[model_1] == test["True"]) & (test[model_2] == test["True"])])
                    N_10 = len(test[(test[model_1] == test["True"]) & (test[model_2] != test["True"])])
                    N_01 = len(test[(test[model_1] != test["True"]) & (test[model_2] == test["True"])])
                    N_00 = len(test[(test[model_1] != test["True"]) & (test[model_2] != test["True"])])
                    measure_dict["Q"].append(((N_11 * N_00) - (N_01 * N_10)) / ((N_11 * N_00) + (N_01 * N_10)))
                    measure_dict["Correlation"].append(((N_11 * N_00) - (N_01 * N_10)) / np.sqrt((N_11 + N_10) * (N_01 + N_00) * (N_11 + N_01) * (N_10 + N_00)))
                    measure_dict["Disagreement"].append((N_01 + N_10) / (N_11 + N_10 + N_01 + N_00))
                    measure_dict["Double-fault"].append(N_00 / (N_11 + N_10 + N_01 + N_00))
                for div in measure_dict:
                    if div != "Disagreement":
                        results_frame["%i %s" % (fold, div)].append(1 - ((2 / (k * (k - 1))) * sum(measure_dict[div])))
                    else:
                        results_frame["%i %s" % (fold, div)].append(((2 / (k * (k - 1))) * sum(measure_dict[div])))

results_frame = pd.DataFrame.from_dict(dict(results_frame))
for cat in categories[::-1]:
    results_frame.insert(loc=0, column='std ACC ' + cat, value=results_frame[acc_cat_cols[cat]].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC ' + cat, value=results_frame[acc_cat_cols[cat]].mean(axis=1))
for div in measures:
    results_frame.insert(loc=0, column='std %s' % div, value=results_frame[div_cols[div]].std(axis=1))
    results_frame.insert(loc=0, column='mean %s' % div, value=results_frame[div_cols[div]].mean(axis=1))
results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
results_frame.insert(loc=0, column='k', value=k_model_list)
results_frame.insert(loc=0, column='model', value=model_name_list)
for div in measures:
    results_frame["F %s" % div] = (1 - results_frame["mean ACC"])**2 + (1 - results_frame["mean %s" % div])**2
results_frame.to_excel(report_file, index=False)
