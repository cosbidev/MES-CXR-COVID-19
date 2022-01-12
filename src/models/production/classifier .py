#import os
import sys
sys.path.extend(["./"])
#import torch
import torchvision.transforms as transforms ###
import matplotlib.pyplot as plt ###
import numpy as np
import os
import pandas as pd
#from keras.models import load_model
from skimage import exposure
from tqdm import tqdm
from skimage.measure import label, regionprops
#from keras.preprocessing.image import ImageDataGenerator
from torch.nn import functional as F
import matplotlib.cm as cm
from skimage import morphology, io, transform
import cv2
from PIL import Image
import torch
from pydicom import dcmread
#from src.models.production.DCM_PACS_predict_init import *
#try:
#    import accimage
#except ImportError:
#    accimage = None
#warnings.filterwarnings("ignore")


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# def loadData(DCM_fpth, df, im_shape):
def loadData(DCM_fpth, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    # X = []
    # original_shape = []
    # img_name = []
    img_name = [DCM_fpth]
    # for i, item in tqdm(df.iterrows()):
    img = io.imread(DCM_fpth).astype(int)
    # img_name.append(item.item())
    original_shape = [img.shape]
    img = transform.resize(img, im_shape)
    try:
        img = np.mean(img, axis=2)
    except:
        pass
    img = np.expand_dims(img, -1)
    img -= img.mean()
    img /= img.std()
    # X.append(img)
    X = np.array([img])

    print('### Data loaded')
    print('\t{}'.format(X.shape))
    return X, original_shape, img_name


# def get_bounding_box(X, UNet, im_shape, original_shape):
def get_bounding_box(df, X, UNet, im_shape, original_shape):
    n_test = X.shape[0]
    inp_shape = X[0].shape
    bounding_box = pd.DataFrame(columns=["dx", "sx", "all"])
    i = 0
    dx_box = []
    sx_box = []
    all_box = []
    for xx in X:
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
        pred = UNet.predict(np.expand_dims(xx, axis=0))[..., 0].reshape(inp_shape[:2])
        # Binarize masks
        pr = pred > 0.5
        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        # resize
        pr = transform.resize(pr, original_shape[i])
        # get box for single lungs
        lbl = label(pr)
        props = regionprops(lbl)
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
            props = regionprops(pr.astype("int64"))
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

    # df with boxes
    bounding_box["dx"] = dx_box
    bounding_box["sx"] = sx_box
    bounding_box["all"] = all_box
    bounding_box.index = df["img"]

    return bounding_box


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


def get_box(img, box):
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    c_h = box[0] + l_h // 2
    c_w = box[1] + l_w // 2
    if l_h > l_w:
        img = img[box[0]:box[2], max(0, c_w - l_h // 2):c_w + l_h // 2]
    elif l_w > l_h:
        img = img[max(0, c_h - l_w // 2):c_h + l_w // 2, box[1]:box[3]]
    else:
        img = img[box[0]:box[2], box[1]:box[3]]
    return img


def normalize(img):
    # img = img / img.max()
    img -= img.mean()
    img /= img.std()
    return img


def loader(path, size=224, box=None, step="train"):
    with open(path, 'rb') as f:
        _, ext = os.path.splitext(path)
        if ext in [".dcm", ".dicom"]:
            img = dcmread(f)
            img = img.pixel_array.astype(float)
        else:
            img = Image.open(f)
            # to numpy
            img = np.array(img).astype(float)
        # to grayscale
        if img.ndim > 2:
            img = img.mean(2)
        # to 3 channels
        img = np.stack((img, img, img), axis=-1)
        # select box area
        if box:
            img = get_box(img, box)
        # normalize
        img = normalize(img)
        # resize
        img = cv2.resize(img, (size, size))
        # if step == "train":
        # img = augmentation(img)
        return to_tensor(img)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float()  # .div(255)
        else:
            return img.float()

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()  # .div(255)
    else:
        return img.float()


class Dataset_prediction(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, boxes, step):
        'Initialization'
        self.step = step
        self.list_IDs = list_IDs
        self.boxes = boxes

    def _find_classes(self, classes):
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # load box
        box = self.boxes[ID]
        # Load data and get label
        X = loader(ID, size=224, box=box, step=self.step)
        return X, ID


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def predict(fpth):
    print('\n--- Predicting image "' + fpth + '" ---')

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
    results = pd.DataFrame(index=categories)


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

        # Covid Risk
        prob_covid = probs[categories.index("COVID")]
        if prob_covid >= 0.5:
            if prob_covid >= 0.75:
                risk.append("+")
            else:
                risk.append("-")
        else:
            risk.append("")

        gcam_1.backward(ids=ids_1[:, [0]])
        regions = gcam_1.generate(target_layer=target_layer)

        img_name = '.'.join(os.path.basename(fpth).split('.')[:-1])
        grad_cam_path = os.path.join(gradcam_pth, img_name + '_' + categories[ids_1[0][0]] + '.png')

        grad_cam_path_list.append(grad_cam_path)
        save_gradcam(filename=grad_cam_path, gcam=regions[0, 0], raw_image=raw_image)

        print('--- Gradcam saved ---') #


