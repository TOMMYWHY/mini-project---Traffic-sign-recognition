import cv2, os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model_builder import Generalized_CNN
from utils import load_weights


BASE_SIZE = 72
CROP_SIZE = 64
PIXEL_MEANS = (0.485, 0.456, 0.406)
PIXEL_STDS = (0.229, 0.224, 0.225)
means = torch.tensor([PIXEL_MEANS[2] * 255, PIXEL_MEANS[1] * 255, PIXEL_MEANS[0] * 255]).cpu().view(1, 3, 1, 1)
stds = torch.tensor([PIXEL_STDS[2] * 255, PIXEL_STDS[1] * 255, PIXEL_STDS[0] * 255]).cpu().view(1, 3, 1, 1)

test_weights = './cnn_models/model_final.pth'
# test_weights = './cnn_models/model.pth'
root_path = './'

image_sources = {
    'img_source/00005_00027.jpeg': 17,
    'img_source/00000_00028.jpeg': 8,
    'img_source/00000_00029.jpeg': 37
}

def pre_process(img, trans):
    img_trans = trans(img)
    np_array = np.asarray(img_trans, dtype=np.uint8)
    np_input = np.rollaxis(np_array, 2)

    processed_tensor = torch.from_numpy(np_input).unsqueeze(0)
    processed_tensor = processed_tensor.float().sub_(means).div_(stds)
    return np_array, processed_tensor

def main():
    trans = transforms.Compose([
        transforms.Resize(BASE_SIZE, interpolation=Image.BILINEAR),
        transforms.CenterCrop(CROP_SIZE)
    ])

    model = Generalized_CNN()
    model.to(torch.device("cpu"))
    load_weights(model, test_weights)
    model.eval()

    plt.figure()
    num_img = len(image_sources)
    img_id = 0
    for img_path, label in image_sources.items():
        img_id += 1

        img  = Image.open(
            os.path.join(root_path, img_path)
        )

        resized_img, input = pre_process(img, trans)

        plt.subplot(2, num_img, img_id)
        plt.imshow(resized_img)
        plt.xticks([])
        plt.yticks([])
        plt.title('GT label: {}'.format(label))

        out = model(input)
        print("out",out)
        _, pred = out.topk(1, 1, True, True)
        pred = pred.t().view(-1)
        print('Predict img {} to class {} vs {} label.'.format(img_path.split('/')[-1].strip(), label, pred[0]))

        plt.subplot(2, num_img, img_id + num_img)
        plt.imshow(resized_img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Predict result: {}'.format(pred[0]))

    plt.show()

if __name__ == '__main__':
    main()
