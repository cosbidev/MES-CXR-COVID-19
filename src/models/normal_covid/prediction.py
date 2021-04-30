import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
from tqdm import tqdm
import src.utils.util_model as util_model
import src.utils.util_data as util_data
import pandas as pd
import os

task = "normal_covid"
categories = sorted(["Normal", "COVID"])

cv = 10
fold_list = list(range(cv))
fold_list = [0]

# Location of data
#source = "../data"
source = "../../../../warp10data/ESA/data"

#report_dir = os.path.join("./reports", task)
report_dir = os.path.join("../../../../warp10data/ESA/reports", task)

#model_dir = os.path.join("./models", task)
model_dir = os.path.join("../../../../warp10data/ESA/models", task)

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
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

model_list = report["model"].to_list()
for fold in fold_list:
    print(fold)

    report_file = os.path.join(report_dir, "reports_prediction_%s.xlsx" % fold)
    util_data.delete_file(report_file)

    # test file
    test_file = os.path.join('./data/processed', task, str(cv), str(fold), 'test.txt')
    test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

    # Partition
    partition = {"test": test_set.index.tolist()}
    labels = test_set["label"].to_dict()

    # Generators
    image_dataset = util_model.Dataset(partition["test"], labels, boxes, source, "test")
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_dir_split = os.path.join(model_dir, str(cv), str(fold))
    results_frame = pd.DataFrame()
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
        predictions = {}
        truth = {}
        with torch.no_grad():
            # Testing loop
            for img, targets, file_name in tqdm(dataloader):
                img = img.to(device)
                targets = targets.to(device)
                # Raw model output
                out = model(img.float())
                pred = out[0]
                true = targets[0]
                _, pred = pred.unsqueeze(0).topk(k=1, dim=1, largest=True, sorted=True)
                pred = pred[0][0]
                predictions[file_name[0]] = pred.item()
                truth[file_name[0]] = true.item()
        results_frame[model_name] = pd.Series(predictions)
    results_frame["True"] = pd.Series(truth)
    results_frame.to_excel(report_file, index=True)
