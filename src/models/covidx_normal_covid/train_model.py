import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import src.utils.util_data as util_data
import src.utils.util_model as util_model
import collections

task = "covidx_normal_covid"
categories = ["Normal", "COVID"]

cv = 10
fold_list = list(range(cv))
fold_list = [0]

# Location of data
#source = "../data/COVIDX/img"
source = "../../../../warp10data/ESA/data/COVIDX/img"

#model_dir = os.path.join("./models", task)
model_dir = os.path.join("../../../../warp10data/ESA/models", task)
model_dir_cv = os.path.join(model_dir, str(cv))
util_data.create_dir(model_dir_cv)

#report_file = os.path.join('./reports', task, 'report_'+str(cv)+'.xlsx')
report_file = os.path.join('../../../../warp10data/ESA/reports', task, 'report_'+str(cv)+'.xlsx')
util_data.delete_file(report_file)
#report_file_temp = os.path.join('./reports', task, 'report_'+str(cv)+'_temp.xlsx')
report_file_temp = os.path.join('../../../../warp10data/ESA/reports', task, 'report_'+str(cv)+'_temp.xlsx')
util_data.delete_file(report_file_temp)

#plot_dir = os.path.join("./reports/figures", task)
plot_dir = os.path.join("../../../../warp10data/ESA/figures", task)
plot_dir_cv = os.path.join(plot_dir, str(cv))
util_data.create_dir(plot_dir_cv)

model_list = ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
              "squeezenet1_0", "squeezenet1_1", "densenet121", "densenet169", "densenet161", "densenet201",
              "googlenet",  "mobilenet_v2", "resnext50_32x4d", "wide_resnet50_2"]
# "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "mnasnet0_5", "mnasnet1_0"

# Change to fit hardware
batch_size = 32

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

data_file = os.path.join("./data/processed", task, "data.xlsx")
db = pd.read_excel(data_file, index_col="img", dtype=list)

# Results table
results_frame = {}
acc_cols = []
acc_cat_cols = collections.defaultdict(lambda: [])
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

    model_dir_split = os.path.join(model_dir_cv, str(fold))
    util_data.create_dir(model_dir_split)

    plot_dir_split = os.path.join(plot_dir_cv, str(fold))
    util_data.create_dir(plot_dir_split)

    all_file = os.path.join(os.path.join('./data/processed', task, 'all.txt'))
    train_file = os.path.join('./data/processed', task, str(cv), str(fold), 'train.txt')
    val_file = os.path.join('./data/processed', task, str(cv), str(fold), 'val.txt')
    test_file = os.path.join('./data/processed', task, str(cv), str(fold), 'test.txt')
    all_set = pd.read_csv(all_file, delimiter=" ", index_col=0)
    train_set = pd.read_csv(train_file, delimiter=" ", index_col=0)
    val_set = pd.read_csv(val_file, delimiter=" ", index_col=0)
    test_set = pd.read_csv(test_file, delimiter=" ", index_col=0)

    partition = {"train": train_set.index.tolist(),
                 "val": val_set.index.tolist(),
                 "test": test_set.index.tolist()}
    labels = all_set["label"].to_dict()

    boxes = {}
    for row in db.iterrows():
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
            im = Image.open(os.path.join(source, i))
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
    image_datasets = {step: util_model.Dataset(partition[step], labels, boxes, source, step) for step in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    # Get a batch of training data
    #inputs, classes = next(iter(dataloaders['train']))
    #print(inputs.shape, classes.shape)
    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)
    #util_model.imshow(out)

    # Finetuning the convnet
    for model_name in model_list:
        print("********************************************")
        print(model_name)
        model_file_name = model_name+".pt"
        if model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg11":
            model = models.vgg11(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg11_bn":
            model = models.vgg11_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg13":
            model = models.vgg13(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg13_bn":
            model = models.vgg13_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg16_bn":
            model = models.vgg16_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg19":
            model = models.vgg19(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        if model_name == "vgg19_bn":
            model = models.vgg19_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, len(class_names), bias=True)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, len(class_names), bias=True)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, len(class_names), bias=True)
            #remove maxpooling layer
            pool = True
            if pool == False:
                model.maxpool = util_model.Identity()
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=True)
            model.fc = nn.Linear(2048, len(class_names), bias=True)
        elif model_name == "resnet152":
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(2048, len(class_names), bias=True)
        elif model_name == "squeezenet1_0":
            model = models.squeezenet1_0(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        elif model_name == "squeezenet1_1":
            model = models.squeezenet1_1(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
        elif model_name == "densenet169":
            model = models.densenet169(pretrained=True)
            model.classifier = nn.Linear(in_features=1664, out_features=len(class_names), bias=True)
        elif model_name == "densenet161":
            model = models.densenet161(pretrained=True)
            model.classifier = nn.Linear(in_features=2208, out_features=len(class_names), bias=True)
        elif model_name == "densenet201":
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(in_features=1920, out_features=len(class_names), bias=True)
        #elif model_name == "inception_v3":
            #model = models.inception_v3(pretrained=True)
            #model.fc = nn.Linear(512, len(class_names), bias=True)
        elif model_name == "googlenet":
            model = models.googlenet(pretrained=True)
            model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
        elif model_name == "shufflenet_v2_x0_5":
            model = models.shufflenet_v2_x0_5(pretrained=True)
            model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
        elif model_name == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
        #elif model_name == "shufflenet_v2_x2_0":
            #model = models.shufflenet_v2_x2_0(pretrained=True)
            #model.fc = nn.Linear(512, len(class_names), bias=True)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        elif model_name == "resnext50_32x4d":
            model = models.resnext50_32x4d(pretrained=True)
            model.fc = nn.Linear(2048, len(class_names), bias=True)
        elif model_name == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=True)
            model.fc = nn.Linear(2048, len(class_names), bias=True)
        elif model_name == "mnasnet0_5":
            model = models.mnasnet0_5(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        #elif model_name == "mnasnet0_75":
            #model = models.mnasnet0_75(pretrained=True)
            #model.fc = nn.Linear(512, len(class_names), bias=True)
        elif model_name == "mnasnet1_0":
            model = models.mnasnet1_0(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        #elif model_name == "mnasnet1_3":
            #model = models.mnasnet1_3(pretrained=True)
            #model.fc = nn.Linear(512, len(class_names), bias=True)

        model = model.to(device)
        #summary(model, input_size=inputs.shape[1:], batch_size=batch_size)
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        # Train model
        model, history = util_model.train_model(model, criterion, optimizer_ft, exp_lr_scheduler, model_file_name,
                                                dataloaders, dataset_sizes, model_dir_split, device, num_epochs=300, max_epochs_stop=25)

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

        # Load model
        #model = torch.load("./models/"+model_file_name)

        # Evaluate the model on all the training data
        results, acc = util_model.evaluate(model, dataloaders['test'], criterion, idx_to_class, device, topk=(1, ))
        print(results)
        print(acc)

        # Update report
        results_frame[str(fold) + " ACC"].append(acc)
        for cat in categories:
            results_frame[str(fold) + " ACC " + str(cat)].append(results.loc[results["class"] == cat]["top1"].item())

        # Save temporary Results
        results_frame_temp = pd.DataFrame.from_dict(results_frame)
        for cat in categories[::-1]:
            results_frame_temp.insert(loc=0, column='std ACC ' + cat, value=results_frame_temp[acc_cat_cols[cat]].std(axis=1))
            results_frame_temp.insert(loc=0, column='mean ACC ' + cat, value=results_frame_temp[acc_cat_cols[cat]].mean(axis=1))
        results_frame_temp.insert(loc=0, column='std ACC', value=results_frame_temp[acc_cols].std(axis=1))
        results_frame_temp.insert(loc=0, column='mean ACC', value=results_frame_temp[acc_cols].mean(axis=1))
        results_frame_temp.insert(loc=0, column='model', value=model_list[:len(results_frame_temp)])
        results_frame_temp.to_excel(report_file_temp, index=False)

        # Test results
        results = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])

        # Weighted column of test images
        results['weighted'] = results['n_test'] / results['n_test'].sum()

        # Create weighted accuracies
        for i in (1, 1):# range of k
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
results_frame.insert(loc=0, column='model', value=model_list)
results_frame.to_excel(report_file, index=False)
