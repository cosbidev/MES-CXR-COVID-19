import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure, transform
import os
from tqdm import tqdm
import skimage.measure as measure
import pydicom
from PIL import Image


def masked_pred(img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def loadData(path, im_shape, max_img_label, label_dir):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X = []
    original_shape = []
    img_name = []
    for item in tqdm(os.listdir(path)[:max_img_label]):
        filename, extension = os.path.splitext(item)
        if extension == ".dcm":
            dicom = pydicom.dcmread(os.path.join(path, item))
            img = dicom.pixel_array.astype(int)
        else:
            img = Image.open(os.path.join(path, item))
            img = np.array(img).astype(int)
        img_name.append(os.path.join(label_dir, item))
        original_shape.append(img.shape)
        img = transform.resize(img, im_shape)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img = np.expand_dims(img, -1)
        img -= img.mean()
        img /= img.std()
        X.append(img)
    X = np.array(X)

    print('### Data loaded')
    print('\t{}'.format(path))
    print('\t{}'.format(X.shape))
    return X, original_shape, img_name


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    task = "esa"

    # Model path
    model_path = './models/segmentation/trained_model.hdf5'
    im_shape = (256, 256)
    UNet = load_model(model_path)

    # Path to the folder with images
    img_dir = "../data"
    img_dict = {"RSNA/normalpatient": "Normal",  # 2081
                "RSNA/patientPneumonia": "Normal",  # 2018
                "AI against COVID-19/negative": "Normal",  # 404
                "AI against COVID-19/positive": "COVID",  # 2358
                "AIforCOVID/imgs": "COVID",  # 820
                "AIforCOVID/imgs_r2": "COVID"} # 283
    max_img_label_dict = {"RSNA/normalpatient": 1500,  # 2081
                          "RSNA/patientPneumonia": 1500,  # 2018
                          "AI against COVID-19/negative": 404,  # 404
                          "AI against COVID-19/positive": 2358,  # 2358
                          "AIforCOVID/imgs": 820,  # 820
                          "AIforCOVID/imgs_r2": 283} # 283}

    # Bounding-box file
    bounding_box_file = os.path.join("./data/processed", task, "data.xlsx")
    bounding_box = pd.DataFrame(columns=["img", "dx", "sx", "all", "label"])

    for label_dir, label in img_dict.items():
        max_img_label = max_img_label_dict[label_dir]

        # Load test data
        X, original_shape, img_name = loadData(os.path.join(img_dir, label_dir), im_shape, max_img_label, label_dir)

        n_test = X.shape[0]
        inp_shape = X[0].shape

        # For inference standard keras ImageGenerator is used.
        test_gen = ImageDataGenerator(rescale=1.)

        i = 0
        # Bounding Boxes
        bounding_box_label = pd.DataFrame(columns=["img", "dx", "sx", "all"])
        dx_box = []
        sx_box = []
        all_box = []
        for xx in tqdm(X):
            img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
            pred = UNet.predict(np.expand_dims(xx, axis=0))[..., 0].reshape(inp_shape[:2])

            # Binarize masks
            pr = pred > 0.5

            # Remove regions smaller than 2% of the image
            pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

            # resize
            pr = transform.resize(pr, original_shape[i])
            # get box for single lungs
            lbl = measure.label(pr)
            props = measure.regionprops(lbl)
            # devo fare singolo
            if len(props) >= 2:
                box_1 = props[0].bbox
                box_2 = props[1].bbox
                if box_1[1] < box_2[1]:
                    dx_box.append(list(box_1))
                    sx_box.append(list(box_2))
                else:
                    dx_box.append(list(box_2))
                    sx_box.append(list(box_1))
                # get box for both lungs
                props = measure.regionprops(pr.astype("int64"))
                if len(props) == 1:
                    all_box.append(list(props[0].bbox))
                else:
                    all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])
            else:
                dx_box.append(None)
                sx_box.append(None)
                all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])

            i += 1
            if i == n_test:
                break

        # save excel with boxes
        bounding_box_label["img"] = img_name
        bounding_box_label["dx"] = dx_box
        bounding_box_label["sx"] = sx_box
        bounding_box_label["all"] = all_box
        bounding_box_label["label"] = [img_dict[label_dir]]*len(img_name)

        bounding_box = pd.concat([bounding_box, bounding_box_label], ignore_index=True)

    bounding_box.to_excel(bounding_box_file, index=False)
