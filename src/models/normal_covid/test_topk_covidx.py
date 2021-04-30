import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
from tqdm import tqdm
from torch.nn import functional as F
import src.utils.util_model as util_model
import src.utils.util_data as util_data
import pandas as pd
import os
import collections

task = "normal_covid"
categories = sorted(["Normal", "COVID"])

k_list = [3, 5, 7, 9]

cv = 10
fold_list = list(range(cv))
fold_list = [0]

# Location of data
#source = "../data"
source = "../../../../warp10data/ESA/data/COVIDX/img"

#report_dir = os.path.join("./reports", task)
report_dir = os.path.join("../../../../warp10data/ESA/reports", task, "covidx")

#trained_model_report_dir = os.path.join("./reports", task)
trained_model_report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

#model_dir = os.path.join("./models", task)
model_dir = os.path.join("../../../../warp10data/ESA/models", task)

report_file = os.path.join(report_dir, "reports_top_k.xlsx")
util_data.delete_file(report_file)

report = pd.read_excel(os.path.join(trained_model_report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="mean ACC", ascending=False)

data_file = os.path.join("./data/processed", "covidx_normal_covid", "data.xlsx")
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

results_frame = {}
acc_cols = []
acc_cat_cols = collections.defaultdict(lambda: [])
model_name_list = []
for k in k_list:
    model_name_list += ["Top %s" % k, "Top %s prob" % k]
for fold in fold_list:
    acc_col = str(fold) + " ACC"
    acc_cols.append(acc_col)
    results_frame[acc_col] = []
    for cat in categories:
        cat_col = str(fold) + " ACC " + cat
        acc_cat_cols[cat].append(cat_col)
        results_frame[cat_col] = []
acc_cat_cols = dict(acc_cat_cols)

for k in k_list:
    # top k models
    model_list = report["model"].iloc[:k].to_list()
    for fold in fold_list:
        print(fold)

        # test file
        test_file = os.path.join('./data/processed', "covidx_normal_covid", 'all.txt')
        test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

        # Partition
        partition = {"test": test_set.index.tolist()}
        labels = test_set["label"].to_dict()

        # Generators
        image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model_dir_split = os.path.join(model_dir, str(cv), str(fold))
        top_k_model_acc = collections.defaultdict(lambda: 0)
        top_k_model_prob = collections.defaultdict(lambda: {label: 0 for label in categories})
        for model_name in model_list:
            print(model_name)
            # Load model
            model_file_name = model_name+".pt"
            if device.type == "cpu":
                model = torch.load(os.path.join(model_dir_split, model_file_name), map_location=torch.device('cpu'))
            else:
                model = torch.load(os.path.join(model_dir_split, model_file_name))
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                # Testing loop
                for img, targets, file_name in tqdm(dataloader):
                    img = img.to(device)
                    targets = targets.to(device)
                    # Raw model output
                    out = model(img.float())
                    probs = F.softmax(out, dim=1)
                    probs = probs[0].tolist()
                    for i, p in enumerate(probs):
                        if p > top_k_model_prob[file_name[0]][categories[i]]:
                            top_k_model_prob[file_name[0]][categories[i]] = p
                    # Iterate through each example
                    pred = out[0]
                    true = targets[0]
                    _, pred = pred.unsqueeze(0).topk(k=1, dim=1, largest=True, sorted=True)
                    pred = pred[0][0]
                    if pred == true:
                        top_k_model_acc[file_name[0]] += 1
        # Top k
        k_model_acc_dict = collections.defaultdict(lambda: 0)
        tot_dict = collections.defaultdict(lambda: 0)
        for file_name, n_acc in top_k_model_acc.items():
            label = data["label"].loc[file_name]
            tot_dict["all"] += 1
            tot_dict[label] += 1
            if n_acc > k//2:
                k_model_acc_dict["all"] += 1
                k_model_acc_dict[label] += 1
        for label, v in k_model_acc_dict.items():
            if label == "all":
                results_frame["%s ACC" % fold].append(v / tot_dict[label] * 100)
            else:
                results_frame["%s ACC %s" % (fold, label)].append(v / tot_dict[label] * 100)
        # Top k Prob
        k_model_acc_dict_prob = collections.defaultdict(lambda: 0)
        tot_dict_prob = collections.defaultdict(lambda: 0)
        for file_name, probs_dict in top_k_model_prob.items():
            label = data["label"].loc[file_name]
            pred = max(probs_dict, key=probs_dict.get)
            tot_dict_prob["all"] += 1
            tot_dict_prob[label] += 1
            if label == pred:
                k_model_acc_dict_prob["all"] += 1
                k_model_acc_dict_prob[label] += 1
        for label, v in k_model_acc_dict_prob.items():
            if label == "all":
                results_frame["%s ACC" % fold].append(v / tot_dict_prob[label] * 100)
            else:
                results_frame["%s ACC %s" % (fold, label)].append(v / tot_dict_prob[label] * 100)


results_frame = pd.DataFrame.from_dict(dict(results_frame))
for cat in categories[::-1]:
    results_frame.insert(loc=0, column='std ACC ' + cat, value=results_frame[acc_cat_cols[cat]].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC ' + cat, value=results_frame[acc_cat_cols[cat]].mean(axis=1))
results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
results_frame.insert(loc=0, column='model', value=model_name_list)
results_frame.to_excel(report_file, index=False)
