import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import collections
import pandas as pd


def get_data_info(img_dir):
    bit_dict = collections.defaultdict(lambda: 0)
    resolution_dict = {0: [], 1: []}
    compression_list = []
    range_list = []
    contrast_list = []
    for img_name in tqdm(os.listdir(img_dir)):
        filename, extension = os.path.splitext(img_name)
        if extension == ".dcm":
            dicom = pydicom.dcmread(os.path.join(img_dir, img_name))
            img = dicom.pixel_array.astype(float)
        else:
            img = Image.open(os.path.join(img_dir, img_name))
            img = np.array(img).astype(float)

        # Bit
        bit_list = [8, 12, 16]
        i = 0
        while True:
            bit = bit_list[i]
            if img.max() > 2 ** bit - 1:
                i += 1
            else:
                break
        bit_dict[bit] += 1

        # Resolution
        shape = img.shape
        resolution_dict[0].append(shape[0])
        resolution_dict[1].append(shape[1])

        # Compression
        compression_list.append(len(collections.Counter(img.flatten())) / (2 ** bit - 1))

        # Dynamic Range
        range_list.append((img.max() - img.min()) / (2 ** bit - 1))

        # Contrast
        contrast_list.append((img.max() - img.min()) / (img.max() + img.min()))

    # To Dataframe
    info = {}
    info["Bit"] = str(dict(bit_dict))
    info["Resolution"] = "%0.4f(%0.4f) x %0.4f(%0.4f)" % (np.mean(resolution_dict[0]), np.std(resolution_dict[0]), np.mean(resolution_dict[1]), np.std(resolution_dict[1]))
    info["Compression"] = "%0.4f(%0.4f)" % (np.mean(compression_list), np.std(compression_list))
    info["Range"] = "%0.4f(%0.4f)" % (np.mean(range_list), np.std(range_list))
    info["Contrast"] = "%0.4f(%0.4f)" % (np.mean(contrast_list), np.std(contrast_list))
    info = pd.Series(info)

    return info


info_data = pd.DataFrame()

# AIforCovid
data_name = "AIforCovid"
img_dir = "../data/tiff_analysis_images_raw"
info_data[data_name] = get_data_info(img_dir)

# COVIDX
data_name = "COVIDX"
img_dir = "../data/COVIDX/img"
info_data[data_name] = get_data_info(img_dir)

# RSNA
data_name = "RSNA"
img_dir = "../data/normalpatient"
info_data[data_name] = get_data_info(img_dir)

# Brixia
data_name = "Brixia"
img_dir = "../data/Brixia/dicom_clean"
info_data[data_name] = get_data_info(img_dir)

# Save
info_data.to_excel("./reports/data/info_data.xlsx")
