import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

from datasets import ImageFolderList, fast_collate
from model_builder import Generalized_CNN
from utils import load_weights, AverageMeter, accuracy
import argparse
import sys
import os

test_batch_size = 128
test_loader_threads = 4
BASE_SIZE = 72
CROP_SIZE = 64
PIXEL_MEANS = (0.485, 0.456, 0.406)
PIXEL_STDS = (0.229, 0.224, 0.225)
means = torch.tensor([PIXEL_MEANS[2] * 255, PIXEL_MEANS[1] * 255, PIXEL_MEANS[0] * 255]).cpu().view(1, 3, 1, 1)
stds = torch.tensor([PIXEL_STDS[2] * 255, PIXEL_STDS[1] * 255, PIXEL_STDS[0] * 255]).cpu().view(1, 3, 1, 1)

test_weights = './cnn_models/model_final.pth'

def test(model, loader):
    # switch to evaluate mode
    model.eval()

    acc1 = AverageMeter()
    acc5 = AverageMeter()

    batch_idx = 0
    for inputs, targets in loader:
        with torch.no_grad():
            batch_idx += 1

            inputs = inputs.float().sub_(means).div_(stds)

            outputs = model(inputs)

            [top1, top5], idx = accuracy(outputs, targets, topk=[1, 5])
            acc1.update(top1[0], inputs.size(0))
            acc5.update(top5[0], inputs.size(0))

            suffix = '[acc1: {:4.2f}% | acc5: {:4.2f}%]'.format(acc1.avg, acc5.avg)
            print(suffix)

    return acc1.avg, acc5.avg

def main(argv):
    print("Cnn_test_Path-----",argv[1])
    Cnn_test_Path=argv[1]

    trans = transforms.Compose([transforms.Resize(BASE_SIZE, interpolation=Image.BILINEAR),
                                transforms.CenterCrop(CROP_SIZE)])

    test_set = ImageFolderList([Cnn_test_Path], transform=trans)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_loader_threads,
            pin_memory=True,
            sampler=None,
            collate_fn=fast_collate
        )

    model = Generalized_CNN()

    load_weights(model, test_weights)

    model.to(torch.device("cpu"))

    test_top1, test_top5 = test(model, test_loader)
    print('test_acc1: {:.4f}% | test_acc5: {:.4f}%'.format(test_top1, test_top5))



if __name__ == '__main__':
    main(sys.argv)
