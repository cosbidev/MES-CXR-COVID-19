import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import src.utils.util_data as util_data
import src.utils.util_model as util_model
import collections

task = "covidx_normal_covid"
categories = ["Normal", "COVID"]

cv = 10
cv_list = list(range(cv))
cv_list = [0]

# Location of data
#data_dir = "../data/COVIDX/img"
data_dir = "../../../../warp10data/ESA/data/COVIDX/img"

#model_dir = os.path.join("./models/", task)
model_dir = os.path.join("../../../../warp10data/ESA/models", task)
model_dir_cv = os.path.join(model_dir, str(cv))

#report_dir = os.path.join('./reports', task)
report_dir = os.path.join('../../../../warp10data/ESA/reports', task)
report_file = os.path.join(report_dir, '3report_'+str(cv)+'.xlsx')
util_data.delete_file(report_file)

#plot_dir = os.path.join("./reports/figures", task)
plot_dir = os.path.join("../../../../warp10data/ESA/figures", task)
plot_dir_cv = os.path.join(plot_dir, str(cv))

model_name = "3models"

report = pd.read_excel(os.path.join(report_dir, "report_%s.xlsx" % cv))
report = report.sort_values(by="mean ACC", ascending=False)
top_3_models = report["model"][:3].to_list()

num_epochs = 300
max_epochs_stop = 25

dim_list = [32, 32, 32]

mmhid = 96
label_dim = len(categories)

act = nn.Tanh()

# Change to fit hardware
batch_size = 16

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

# Results table
results_frame = {}
acc_cols = []
acc_cat_cols = collections.defaultdict(lambda: [])
for fold in cv_list:
    acc_col = str(fold) + " ACC"
    acc_cols.append(acc_col)
    results_frame[acc_col] = []
    for cat in categories:
        cat_col = str(fold) + " ACC " + cat
        acc_cat_cols[cat].append(cat_col)
        results_frame[cat_col] = []
acc_cat_cols = dict(acc_cat_cols)


for fold in cv_list:

    model_dir_split = os.path.join(model_dir_cv, str(fold))

    plot_dir_split = os.path.join(plot_dir_cv, str(fold))

    all_file = os.path.join(os.path.join('./data/processed', task, 'all.txt'))
    train_file = os.path.join('./data/processed', task, str(cv), str(fold), 'train.txt')
    val_file = os.path.join('./data/processed', task, str(cv), str(fold), 'val.txt')
    test_file = os.path.join('./data/processed', task, str(cv), str(fold), 'test.txt')
    all_set = pd.read_csv(all_file, delimiter=" ", index_col=0)
    train_set = pd.read_csv(train_file, delimiter=" ", index_col=0)
    val_set = pd.read_csv(val_file, delimiter=" ", index_col=0)
    test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

    box_file = os.path.join("./data/processed", task, "data.xlsx")
    box_data = pd.read_excel(box_file, index_col=0, dtype=list)

    partition = {"train": train_set.index.tolist(),
                 "val": val_set.index.tolist(),
                 "test": test_set.index.tolist()}
    labels = all_set["label"].to_dict()

    boxes = {}
    for row in box_data.iterrows():
        boxes[row[0]] = eval(row[1]["all"])

    # Empty lists
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # Iterate through each category
    for c in categories:
        # Number of each image
        train_imgs = train_set.index[train_set['label'] == c].tolist()
        valid_imgs = val_set.index[val_set['label'] == c].tolist()
        test_imgs = test_set.index[test_set['label'] == c].tolist()
        n_train.append(len(train_imgs))
        n_valid.append(len(valid_imgs))
        n_test.append(len(test_imgs))

        # Find stats for train images
        for i in train_imgs:
            img_categories.append(c)
            im = Image.open(os.path.join(data_dir, i))
            img_array = np.array(im)
            # Shape
            hs.append(img_array.shape[0])
            ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid,
                           'n_test': n_test}).sort_values('category')
    print(cat_df)

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })

    # Distribution of Images
    cat_df.set_index('category')['n_train'].plot.bar(color='r', edgecolor='k')
    plt.ylabel('Count')
    plt.title('Training Images by Category')
    plt.savefig(os.path.join(plot_dir, "Class_distr"))
    plt.show()

    # Distribution of Images Sizes
    img_dsc = image_df.groupby('category').describe()
    print(img_dsc)

    # Generators
    image_datasets = {step: util_model.Dataset(partition[step], labels, boxes, data_dir, step) for step in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    # Model Parameters
    model_paths = [os.path.join(model_dir_split, "%s.pt" % top_3_models[i]) for i in range(3)]

    # Load model
    if device.type == "cpu":
        models = [torch.load(model_path, map_location=torch.device('cpu')) for model_path in model_paths]
    else:
        models = [torch.load(model_path) for model_path in model_paths]

    # Vector Models
    for i in range(3):
        model_name_i = top_3_models[i]
        if model_name_i == "alexnet":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg11":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg11_bn":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg13":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg13_bn":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg16":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg16_bn":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg19":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "vgg19_bn":
            models[i].classifier[-1] = nn.Linear(in_features=4096, out_features=dim_list[i], bias=True)
        elif model_name_i == "resnet18":
            models[i].fc = nn.Linear(512, dim_list[i], bias=True)
        elif model_name_i == "resnet34":
            models[i].fc = nn.Linear(512, dim_list[i], bias=True)
        elif model_name_i == "resnet50":
            models[i].fc = nn.Linear(2048, dim_list[i], bias=True)
        elif model_name_i == "resnet101":
            models[i].fc = nn.Linear(2048, dim_list[i], bias=True)
        elif model_name_i == "resnet152":
            models[i].fc = nn.Linear(2048, dim_list[i], bias=True)
        elif model_name_i == "squeezenet1_0":
            models[i].classifier[1] = nn.Conv2d(512, dim_list[i], kernel_size=(1, 1), stride=(1, 1))
        elif model_name_i == "squeezenet1_1":
            models[i].classifier[1] = nn.Conv2d(512, dim_list[i], kernel_size=(1, 1), stride=(1, 1))
        elif model_name_i == "densenet121":
            models[i].classifier = nn.Linear(in_features=1024, out_features=dim_list[i], bias=True)
        elif model_name_i == "densenet169":
            models[i].classifier = nn.Linear(in_features=1664, out_features=dim_list[i], bias=True)
        elif model_name_i == "densenet161":
            models[i].classifier = nn.Linear(in_features=2208, out_features=dim_list[i], bias=True)
        elif model_name_i == "densenet201":
            models[i].classifier = nn.Linear(in_features=1920, out_features=dim_list[i], bias=True)
        # elif model_name_i == "inception_v3":
        # models[i].fc = nn.Linear(512, dim_list[i], bias=True)
        elif model_name_i == "googlenet":
            models[i].fc = nn.Linear(in_features=1024, out_features=dim_list[i], bias=True)
        elif model_name_i == "shufflenet_v2_x0_5":
            models[i].fc = nn.Linear(in_features=1024, out_features=dim_list[i], bias=True)
        elif model_name_i == "shufflenet_v2_x1_0":
            models[i].fc = nn.Linear(in_features=1024, out_features=dim_list[i], bias=True)
        # elif model_name_i == "shufflenet_v2_x2_0":
        # models.fc = nn.Linear(512, dim_list[i], bias=True)
        elif model_name_i == "mobilenet_v2":
            models[i].classifier[-1] = nn.Linear(in_features=1280, out_features=dim_list[i], bias=True)
        elif model_name_i == "resnext50_32x4d":
            models[i].fc = nn.Linear(2048, dim_list[i], bias=True)
        elif model_name_i == "wide_resnet50_2":
            models[i].fc = nn.Linear(2048, dim_list[i], bias=True)
        elif model_name_i == "mnasnet0_5":
            models[i].classifier[-1] = nn.Linear(in_features=1280, out_features=dim_list[i], bias=True)
        # elif model_name_i == "mnasnet0_75":
        # models[i].fc = nn.Linear(512, dim_list[i], bias=True)
        elif model_name_i == "mnasnet1_0":
            models[i].classifier[-1] = nn.Linear(in_features=1280, out_features=dim_list[i], bias=True)
        # elif model_name_i == "mnasnet1_3":
        # models[i].fc = nn.Linear(512, dim_list[i], bias=True)

    # Connect
    model = util_model.three_models(models[0], models[1], models[2], dim_list[0], dim_list[1], dim_list[2], mmhid, act, label_dim)
    model = model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Train model
    model_file_name = model_name+".pt"
    model, history = util_model.train_3model(model, criterion, optimizer, scheduler, model_file_name, dataloaders, dataset_sizes, model_dir_split, device, num_epochs, max_epochs_stop)

    # Plot trend
    model_plot_dir = os.path.join(plot_dir_split, model_name)
    util_data.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))
    plt.show()
    # Training results Accuracy
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'val_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(model_plot_dir, "Acc"))
    plt.show()

    # Evaluate the model on all the training data
    results, acc = util_model.evaluate_3model(model, dataloaders['test'], criterion, idx_to_class, device, topk=(1,))
    print(results)
    print(acc)

    # Update report
    results_frame[str(fold) + " ACC"].append(acc)
    for cat in categories:
        results_frame[str(fold) + " ACC " + str(cat)].append(results.loc[results["class"] == cat]["top1"].item())

    # Test results
    results = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])

    # Weighted column of test images
    results['weighted'] = results['n_test'] / results['n_test'].sum()

    # Create weighted accuracies
    for i in (1, 1):  # range of k
        results[f'weighted_top{i}'] = results['weighted'] * results[f'top{i}']

    # Find final accuracy accounting for frequencies
    top1_weighted = results['weighted_top1'].sum()
    # top2_weighted = results['weighted_top2'].sum()
    loss_weighted = (results['weighted'] * results['loss']).sum()

    print(f'Final test cross entropy per image = {loss_weighted:.4f}.')
    print(f'Final test top 1 weighted accuracy = {top1_weighted:.2f}%')

results_frame = pd.DataFrame.from_dict(results_frame)
for cat in categories[::-1]:
    results_frame.insert(loc=0, column='std ACC ' + cat, value=results_frame[acc_cat_cols[cat]].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC ' + cat, value=results_frame[acc_cat_cols[cat]].mean(axis=1))
results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
results_frame.insert(loc=0, column='model', value=[model_name])
results_frame.to_excel(report_file, index=False)
