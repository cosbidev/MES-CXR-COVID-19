import sys;

import pandas as pd

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torchvision
from keras.models import load_model

#import sys
#sys.path.extend(["./"])
## noinspection PyUnresolvedReferences
#from keras.preprocessing.image import ImageDataGenerator
import warnings
try:
    import accimage
except ImportError:
    accimage = None
warnings.filterwarnings("ignore")

# Data
data_dir = "C:/Users/guarr/Desktop/RX_da_testare/images/"

# Models' paths
seg_model_path = './models/Segmentation/trained_model.hdf5'
model_path_1 = "./models/normal_covid/10/0/vgg19.pt"
model_path_2 = "./models/normal_covid/10/0/resnext50_32x4d.pt"
model_path_3 = "./models/normal_covid/10/0/resnet50.pt"
model_path_1 = "./models/competition/mnasnet0_5.pt"
model_path_2 = "./models/competition/mobilenet_v2.pt"
model_path_3 = "./models/competition/resnet101.pt"
model_path_1 = "./models/esa/10/0/densenet169.pt"
model_path_2 = "./models/esa/10/0/densenet201.pt"
model_path_3 = "./models/esa/10/0/resnext50_32x4d.pt"
model_path_1 = "./models/esa_healthy/10/0/densenet121.pt"
model_path_2 = "./models/esa_healthy/10/0/resnet101.pt"
model_path_3 = "./models/esa_healthy/10/0/wide_resnet50_2.pt"
model_name_1 = os.path.basename(model_path_1).split(".")[0]
model_name_2 = os.path.basename(model_path_2).split(".")[0]
model_name_3 = os.path.basename(model_path_3).split(".")[0]
gradcam_pth = "./data/interim/predict/gradcam"
result_fpth = "./data/interim/predict/results.csv"

# Parameters
im_shape = (256, 256)
categories = sorted(["Normal", "COVID"])
batch_size = 1
target_layer = "avgpool"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    num_workers = 0
else:
    num_workers = 16

# Load models
UNet = load_model(seg_model_path)

if device.type == "cpu":
    model_1 = torch.load(model_path_1, map_location=torch.device('cpu'))
    model_2 = torch.load(model_path_2, map_location=torch.device('cpu'))
    model_3 = torch.load(model_path_3, map_location=torch.device('cpu'))
else:
    model_1 = torch.load(model_path_1)
    model_2 = torch.load(model_path_2)
    model_3 = torch.load(model_path_3)

model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_3 = model_3.to(device)
model_1.eval()
model_2.eval()
model_3.eval()




results = pd.DataFrame(index=categories)
for file in os.listdir(data_dir):
    print("*************************************")
    print(file)

    fpth = os.path.join(data_dir, file)

    # Load img file and data
    img, original_shape, img_name = loadData(fpth, im_shape)

    # Bounding Boxes
    df = pd.DataFrame([fpth], columns=['img'])
    bounding_box = get_bounding_box(df, img, UNet, im_shape, original_shape)
    boxes = {row[0]: row[1]["all"] for row in bounding_box.iterrows()}
    partition = {"all": bounding_box.index.tolist()}

    # Generators
    image_dataset = Dataset_prediction(partition["all"], boxes, "test")
    #print("\n=== Salva immagine croppata e normalizzata in \"C:\\Users\\e.cordelli\\Desktop\\DCM_Storage\\I.png\" ===")
    #transforms.ToPILImage()(image_dataset[0][0]).save("C:\\Users\\e.cordelli\\Desktop\\DCM_Storage\\I.png")
    #print("========================================================================================================\n")
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Prediction
    risk = []
    grad_cam_path = ''
    grad_cam_path_list = []


    #    data, file_name = next(dataloader)
    for data, file_name in tqdm(dataloader):
        data = data.to(device)

        # GradCam
        image = [data[0]]
        raw_image = np.transpose(data[0].numpy(), (1, 2, 0))
        raw_image = ((raw_image - raw_image.min()) * (1 / (raw_image.max() - raw_image.min()) * 255)).astype('uint8')

        image = torch.stack(image).to(device)

        gcam_1 = GradCAM(model=model_1)
        probs_1, ids_1 = gcam_1.forward(image)
        probs_1 = probs_1.tolist()[0]

        gcam_2 = GradCAM(model=model_2)
        probs_2, ids_2 = gcam_2.forward(image)
        probs_2 = probs_2.tolist()[0]

        gcam_3 = GradCAM(model=model_3)
        probs_3, ids_3 = gcam_3.forward(image)
        probs_3 = probs_3.tolist()[0]

    #        probs = list(np.array([probs_1, probs_2, probs_3]).mean(axis=0))
    #        results[file_name[0]] = probs

        print('----------')
        probs_1_ = np.array(probs_1)
        probs_2_ = np.array(probs_2)
        probs_3_ = np.array(probs_3)
        print('IDs: 0 = COVID, 1 = Normal')
        print('--- Expert 1 ---')
        print(probs_1_)
        print(ids_1.numpy()[0])
        print('--- Expert 2 ---')
        print(probs_2_)
        print(ids_2.numpy()[0])
        print('--- Expert 3 ---')
        print(probs_3_)
        print(ids_3.numpy()[0])
        print('--- Experts average [COVID, Normal] ---')
        probs_test = list(np.array([probs_1_[ids_1.numpy()[0]], probs_2_[ids_2.numpy()[0]], probs_3_[ids_3.numpy()[0]]]).mean(axis=0))
        print(probs_test)
        print('----------')

        probs = probs_test
        results[file_name[0]] = probs


results = results.T

results.index = [index.split("/")[-1] for index in results.index]

truth = pd.read_csv("C:/Users/guarr/Desktop/RX_da_testare/labels.csv", index_col=0, header=None)



