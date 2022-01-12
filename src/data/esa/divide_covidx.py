import pandas as pd
import os
import shutil
from tqdm import tqdm

data_dir = "../data/AI against COVID-19/COVIDX"

info_file = "../data/AI against COVID-19/all.txt"

info = pd.read_csv(info_file, sep=" ", header=None, index_col=1)

info = info[info[3] != "rsna"]

positive = info[info[2] == "positive"]
negative = info[info[2] == "negative"]

positive_dir = "../data/AI against COVID-19/positive"
negative_dir = "../data/AI against COVID-19/negative"
os.mkdir(positive_dir)
os.mkdir(negative_dir)

for file, row in tqdm(positive.iterrows()):
    shutil.copyfile(os.path.join(data_dir, file), os.path.join(positive_dir, file))

for file, row in tqdm(negative.iterrows()):
    shutil.copyfile(os.path.join(data_dir, file), os.path.join(negative_dir, file))
