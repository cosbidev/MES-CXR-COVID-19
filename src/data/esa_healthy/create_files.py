import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
from tqdm import tqdm
import random
import pandas as pd
import src.utils.util_data as util_data
from itertools import chain

task = "esa_healthy"

cv = 10

data_file = os.path.join("./data/processed", task, "data.xlsx")
dest_dir = os.path.join("./data/processed", task)

db = pd.read_excel(data_file, header=0, index_col="img")
label_col = "label"
classes = list(db[label_col].unique())

if cv == 10:
    div = 10
    test_split = 1
    val_split = 2
    train_split = 7
if cv == 7:
    div = 7
    test_split = 1
    val_split = 2
    train_split = 4
if cv == 5:
    div = 5
    test_split = 1
    val_split = 1
    train_split = 3
if cv == 3:
    div = 10
    test_split = 3
    val_split = 2
    train_split = 5

all = []
# all  
with open(os.path.join(dest_dir, 'all.txt'), 'w') as file:
    file.write("img label\n")
    for img in db.index:
        label = db.loc[img, label_col] + "\n"
        row = img+" "+label
        file.write(row)
        all.append(row)

folds = [[]]*cv
for c in classes:
    patient_class = [img+" "+c+"\n" for img in db.index[db[label_col] == c].to_list()]
    # randomize
    random.seed(0)
    random.shuffle(patient_class)
    # create splits
    folds_class = list(util_data.chunks(patient_class, len(patient_class) // cv))
    if len(folds_class) != cv:
        del folds_class[-1]
    for i in range(cv):
        folds[i] = folds[i] + folds_class[i]

# create split dir
dest_dir = os.path.join(dest_dir, str(cv))
util_data.create_dir(dest_dir)
for i in range(cv):
    dest_dir_cv = os.path.join(dest_dir, str(i))
    util_data.create_dir(dest_dir_cv)

    train = list(chain.from_iterable(folds[0:train_split]))
    val = list(chain.from_iterable(folds[train_split:train_split+val_split]))
    test = list(chain.from_iterable(folds[train_split+val_split:train_split+val_split+test_split]))

    # train_CDI.txt #todo: rimouvi spazi
    with open(os.path.join(dest_dir_cv, 'train.txt'), 'w') as file:
        file.write("img label\n")
        for row in tqdm(train):
            file.write(row)
    # val_CDI.txt
    with open(os.path.join(dest_dir_cv, 'val.txt'), 'w') as file:
        file.write("img label\n")
        for row in tqdm(val):
            file.write(row)
    # test_CDI.txt
    with open(os.path.join(dest_dir_cv, 'test.txt'), 'w') as file:
        file.write("img label\n")
        for row in tqdm(test):
            file.write(row)

    # Shift folds by one
    folds = util_data.rotate(folds, 1)
