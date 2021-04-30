import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
from tqdm import tqdm
import pandas as pd
import os
import collections
import itertools
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from src.utils import util_data
import src.utils.util_model as util_model

task = "covidx_normal_covid"
categories = sorted(["Normal", "COVID"])
external_data = ["normal_covid", "brixia"]
source_dict = {"normal_covid": "../../../../warp10data/ESA/data", "brixia": "../../../../warp10data/ESA/data/Brixia/dicom_clean"}

cv = 10
fold = 0

#report_dir = os.path.join("./reports", task)
report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

#model_dir = os.path.join("./models", task)
model_dir = os.path.join("../../../../warp10data/ESA/models", task)

report_file = os.path.join(report_dir, "optimization.xlsx")
results_frame = pd.read_excel(report_file)

measures = ["Correlation"]
#measures = ["Q", "Correlation", "Disagreement", "Double-fault"]

# Generators
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

batch_size = 1

for div in measures:
    print(div)

    # report file
    external_report_file = os.path.join(report_dir, "%s_optimization_external.xlsx" % div)

    # External reportdirs
    external_results_frame = {"ACC": []}
    for cat in categories:
        external_results_frame["ACC %s" % cat] = []
    model_name_list = []
    external_list = []
    best_list = []

    # standardize ACC and DIV
    scaler = MinMaxScaler()
    results_frame[["mean ACC", "mean %s" % div]] = pd.DataFrame(scaler.fit_transform(results_frame[["mean ACC", "mean %s" % div]]), columns=["mean ACC", "mean %s" % div])
    results_frame["F %s" % div] = (1 - results_frame["mean ACC"]) ** 2 + (1 - results_frame["mean %s" % div]) ** 2

    # Best Function on external data
    best_f = results_frame[results_frame["F %s" % div] == results_frame["F %s" % div].min()]

    best_models = [comb.split(";") for comb in best_f["model"]]
    for model_list in best_models:
        for external in external_data:
            print(external)

            model_name_list.append(";".join(model_list))
            external_list.append(external)
            best_list.append("F")

            source = source_dict[external]

            data_file = os.path.join("./data/processed", external, "data.xlsx")
            data = pd.read_excel(data_file, index_col=0, dtype=list)

            boxes = {}
            for row in data.iterrows():
                boxes[row[0]] = eval(row[1]["all"])

            # test file
            test_file = os.path.join('./data/processed', external, 'all.txt')
            test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

            # Partition
            partition = {"test": test_set.index.tolist()}
            labels = test_set["label"].to_dict()

            # Generators
            image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            model_dir_split = os.path.join(model_dir, str(cv), str(fold))
            top_k_model_acc = collections.defaultdict(lambda: 0)
            for model_name in model_list:
                print(model_name)
                # Load model
                model_file_name = model_name + ".pt"
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
                if n_acc > len(model_list) // 2:
                    k_model_acc_dict["all"] += 1
                    k_model_acc_dict[label] += 1
            external_results_frame["ACC"].append(k_model_acc_dict["all"] / tot_dict["all"] * 100)
            for label in categories:
                external_results_frame["ACC %s" % label].append(k_model_acc_dict.get(label, np.nan) / tot_dict.get(label, np.nan) * 100)

    # Best ACC on external data
    best_acc = results_frame[results_frame["mean ACC"] == results_frame["mean ACC"].max()]

    best_models = [comb.split(";") for comb in best_acc["model"]]
    for model_list in best_models:
        for external in external_data:
            print(external)

            model_name_list.append(";".join(model_list))
            external_list.append(external)
            best_list.append("ACC")

            source = source_dict[external]

            data_file = os.path.join("./data/processed", external, "data.xlsx")
            data = pd.read_excel(data_file, index_col=0, dtype=list)

            boxes = {}
            for row in data.iterrows():
                boxes[row[0]] = eval(row[1]["all"])

            # test file
            test_file = os.path.join('./data/processed', external, 'all.txt')
            test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

            # Partition
            partition = {"test": test_set.index.tolist()}
            labels = test_set["label"].to_dict()

            # Generators
            image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

            model_dir_split = os.path.join(model_dir, str(cv), str(fold))
            top_k_model_acc = collections.defaultdict(lambda: 0)
            for model_name in model_list:
                print(model_name)
                # Load model
                model_file_name = model_name + ".pt"
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
                if n_acc > len(model_list) // 2:
                    k_model_acc_dict["all"] += 1
                    k_model_acc_dict[label] += 1
            external_results_frame["ACC"].append(k_model_acc_dict["all"] / tot_dict["all"] * 100)
            for label in categories:
                external_results_frame["ACC %s" % label].append(
                    k_model_acc_dict.get(label, np.nan) / tot_dict.get(label, np.nan) * 100)

    # Best DIV on external data
    best_div = results_frame[results_frame["mean %s" % div] == results_frame["mean %s" % div].max()]

    best_models = [comb.split(";") for comb in best_div["model"]]
    for model_list in best_models:
        for external in external_data:
            print(external)

            model_name_list.append(";".join(model_list))
            external_list.append(external)
            best_list.append("DIV")

            source = source_dict[external]

            data_file = os.path.join("./data/processed", external, "data.xlsx")
            data = pd.read_excel(data_file, index_col=0, dtype=list)

            boxes = {}
            for row in data.iterrows():
                boxes[row[0]] = eval(row[1]["all"])

            # test file
            test_file = os.path.join('./data/processed', external, 'all.txt')
            test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

            # Partition
            partition = {"test": test_set.index.tolist()}
            labels = test_set["label"].to_dict()

            # Generators
            image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

            model_dir_split = os.path.join(model_dir, str(cv), str(fold))
            top_k_model_acc = collections.defaultdict(lambda: 0)
            for model_name in model_list:
                print(model_name)
                # Load model
                model_file_name = model_name + ".pt"
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
                if n_acc > len(model_list) // 2:
                    k_model_acc_dict["all"] += 1
                    k_model_acc_dict[label] += 1
            external_results_frame["ACC"].append(k_model_acc_dict["all"] / tot_dict["all"] * 100)
            for label in categories:
                external_results_frame["ACC %s" % label].append(
                    k_model_acc_dict.get(label, np.nan) / tot_dict.get(label, np.nan) * 100)

    # Save
    external_results_frame = pd.DataFrame.from_dict(dict(external_results_frame))
    external_results_frame.insert(loc=0, column='External', value=external_list)
    external_results_frame.insert(loc=0, column='Best', value=best_list)
    external_results_frame.insert(loc=0, column='model', value=model_name_list)
    external_results_frame.to_excel(external_report_file)
