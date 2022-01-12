import collections
import pandas as pd
import os
from tqdm import tqdm

data_dir = "C:/Users/guarr/Desktop"

label_list = ['Atelectasis',
             'Consolidation',
             'Infiltration',
             'Pneumothorax',
             'Edema',
             'Emphysema',
             'Fibrosis',
             'Effusion',
             'Pleural_Thickening',
             'Cardiomegaly',
             'Nodule',
             'Mass',
             'Hernia',
             'Lung Lesion',
             'Fracture',
             'Lung Opacity',
             'Enlarged Cardiomediastinum']

mapping = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"), index_col=0)

counter = collections.defaultdict(lambda: 0)
c = 0
for img_path in os.listdir(os.path.join(data_dir, "images")):
    img_label = mapping.loc[img_path]['Finding Labels']
    if img_label in label_list:
        c += 1
        counter[img_label] += 1
    else:
        os.remove(os.path.join(data_dir, "images", img_path))
