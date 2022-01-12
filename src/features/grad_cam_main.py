import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import pandas as pd
import os.path as osp
import os

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from src.utils import util_model
from src.utils import util_data

from src.features.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
torch.backends.cudnn.enabled = False


def load_images(image_dir, boxes):
    images = []
    raw_images = []
    print("Images:")
    for i, image_file in enumerate(os.listdir(image_dir)):
        print("\t#{}: {}".format(i, image_file))
        image_path = os.path.join(image_dir, image_file)
        image = util_model.loader(image_path, box=boxes[image_file], step="test")
        images.append(image)
        raw_image = np.transpose(image.numpy(), (1, 2, 0))
        raw_image = ((raw_image - raw_image.min()) * (1/(raw_image.max() - raw_image.min()) * 255)).astype('uint8')
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable(all_file):
    classes = []
    with open(os.path.join(all_file)) as lines:
        next(lines)
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            classes.append(line)
    return classes


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def save_raw_image(filename, raw_image):
    cv2.imwrite(filename, np.uint8(raw_image.astype(np.float)))


source = "../data"

image_dir = os.path.join(source, "sub_tiff_analysis_images_raw")

output_dir = os.path.join("./data", 'interim/backpropagation')

topk = 1

model_dir = os.path.join('./models', "normal_covid", '10', '0')

model_list = ["alexnet",
"densenet121",
"densenet161",
"densenet169",
"densenet201",
"mobilenet_v2",
"resnet101",
"resnet152",
"resnet18",
"resnet34",
"resnet50",
"resnext50_32x4d",
"squeezenet1_0",
"squeezenet1_1",
"vgg11",
"vgg13",
"vgg16",
"vgg19",
"wide_resnet50_2"]

"""
Visualize model responses given multiple images
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Synset words
classes = get_classtable("./data/processed/normal_covid/all.txt")

# boxes
box_file = "./data/processed/normal_covid/data.xlsx"
box_data = pd.read_excel(box_file, index_col=0, dtype=list)
boxes = {}
for row in box_data.iterrows():
    boxes[row[0].replace("tiff_analysis_images_raw/", "")] = eval(row[1]["all"])

# Images
images, raw_images = load_images(image_dir, boxes)
images = torch.stack(images).to(device)

for model_name in model_list:
    print(model_name)

    model_path = os.path.join(model_dir, model_name + ".pt")

    if model_name == "alexnet":
        target_layer = "features"
    elif model_name == "densenet121":
        target_layer = "features"
    elif model_name == "densenet161":
        target_layer = "features"
    elif model_name == "densenet169":
        target_layer = "features"
    elif model_name == "densenet201":
        target_layer = "features"
    elif model_name == "googlenet":
        target_layer = "dropout"
    elif model_name == "mobilenet_v2":
        target_layer = "features"
    elif model_name == "resnet101":
        target_layer = "layer4"
    elif model_name == "resnet152":
        target_layer = "layer4"
    elif model_name == "resnet18":
        target_layer = "layer4"
    elif model_name == "resnet34":
        target_layer = "layer4"
    elif model_name == "resnet50":
        target_layer = "layer4"
    elif model_name == "resnext50_32x4d":
        target_layer = "layer4"
    elif model_name == "squeezenet1_0":
        target_layer = "features"
    elif model_name == "squeezenet1_1":
        target_layer = "features"
    elif model_name == "vgg11":
        target_layer = "features"
    elif model_name == "vgg13":
        target_layer = "features"
    elif model_name == "vgg16":
        target_layer = "features"
    elif model_name == "vgg19":
        target_layer = "features"
    elif model_name == "wide_resnet50_2":
        target_layer = "layer4"

    output_model_dir = os.path.join(output_dir, model_name)
    util_data.create_dir(output_model_dir)

    # Model from torchvision
    if device.type == "cpu":
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
    model.to(device)
    model.eval()

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(filename=osp.join(output_model_dir, "{}-{}-vanilla-{}.png".format(j, model_name, classes[ids[j, i]]),),
                          gradient=gradients[j],)

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(filename=osp.join(output_model_dir, "{}-{}-deconvnet-{}.png".format(j, model_name, classes[ids[j, i]]),),
                          gradient=gradients[j],)

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(output_model_dir, "{}-{}-guided-{}.png".format(j, model_name, classes[ids[j, i]]),),
                gradient=gradients[j],)

            # Grad-CAM
            save_gradcam(filename=osp.join(output_model_dir, "{}-{}-gradcam-{}-{}.tiff".format(j, model_name, target_layer, classes[ids[j, i]]),),
                         gcam=regions[j, 0],
                         raw_image=raw_images[j],
                         paper_cmap=False)

            # Guided Grad-CAM
            save_gradient(filename=osp.join(output_model_dir, "{}-{}-guided_gradcam-{}-{}.png".format(j, model_name, target_layer, classes[ids[j, i]]),),
                          gradient=torch.mul(regions, gradients)[j],)

            # Save Raw iamge
            save_raw_image(filename=osp.join(output_model_dir, "{}-{}-original-{}-{}.png".format(j, model_name, target_layer, classes[ids[j, i]]), ),
                          raw_image=raw_images[j])