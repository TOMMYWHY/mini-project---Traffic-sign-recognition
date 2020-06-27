import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
import time

from datasets import ImageFolderList, fast_collate
from model_builder import Generalized_CNN
# from utils import Optimizer, LearningRateScheduler, get_lr
from utils import get_lr
import sys


train_batch_size = 128
train_loader_threads = 4
BASE_SIZE = 72
CROP_SIZE = 64
PIXEL_MEANS = (0.485, 0.456, 0.406)
PIXEL_STDS = (0.229, 0.224, 0.225)
means = torch.tensor([PIXEL_MEANS[2] * 255, PIXEL_MEANS[1] * 255, PIXEL_MEANS[0] * 255]).cpu().view(1, 3, 1, 1)
stds = torch.tensor([PIXEL_STDS[2] * 255, PIXEL_STDS[1] * 255, PIXEL_STDS[0] * 255]).cpu().view(1, 3, 1, 1)
solver = {'OPTIMIZER': 'SGD',
          'BASE_LR': 0.1,
          'MAX_EPOCHS': 16,
          'MOMENTUM': 0.9,
          'WARM_UP_METHOD': 'LINEAR',
          'WEIGHT_DECAY': 0.0001,
          'WARM_UP_EPOCH': 0,
          'WARM_UP_FACTOR': 0.1,
          'LR_POLICY': 'STEP',
          'STEPS': [8, 12],
          'GAMMA': 0.1,
          'WEIGHT_DECAY_GN': 0.0,
          'BIAS_DOUBLE_LR': True,
          'BIAS_WEIGHT_DECAY': False,
          'LR_MULTIPLE': 1.0,
          'SNAPSHOT_EPOCHS': -1,
          'LOG_LR_CHANGE_THRESHOLD': 1.1,
          'LR_POW': 0.9}


def train(model, criterion, trainloader, optimizer):

    for epoch in range(solver['MAX_EPOCHS']):
        current_learning_rate = solver['BASE_LR']
        if epoch + 1 in solver['STEPS']:
            current_learning_rate = get_lr(epoch+1, solver)
            for ind, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = current_learning_rate
        print('learning rate for current epoch: {}'.format(optimizer.param_groups[0]['lr']))


        running_loss = 0.0
        epoch_start_time = time.time()
        for iteration, data in enumerate(trainloader, 0):
            iter_start_time = time.time()

            inputs, targets = data

            inputs = inputs.float().sub_(means).div_(stds)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time

            running_loss += loss.item()
            if iteration % 20 == 19:  # print every 20 mini-batches
                print('epoch: {} | iter: {} | lr: {} | loss: {} | iter time: {}'.format(
                    epoch + 1,
                    iteration + 1,
                    current_learning_rate,
                    running_loss / 20,
                    iter_time)
                )
                running_loss = 0.0

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print('epoch {} time cost:'.format(epoch_time))
        if (epoch + 1) % 4 == 0:
            torch.save(model.state_dict(), 'cnn_models/model_epoch{}.pth'.format(epoch + 1))
            print('Saved trained model at epoch {} to cnn_models/model_epoch{}.pth'.format((epoch + 1), (epoch + 1)))
    torch.save(model.state_dict(), 'cnn_models/model_final.pth')


def main(argv):
    print("Cnn_tain_Path-----",argv[1])
    Cnn_tain_Path=argv[1]

    trans = transforms.Compose([transforms.Resize(BASE_SIZE, interpolation=Image.BILINEAR),
                                transforms.CenterCrop(CROP_SIZE)])

    tain_set = ImageFolderList([Cnn_tain_Path], transform=trans)

    train_loader = torch.utils.data.DataLoader(
        tain_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_loader_threads,
        pin_memory=True,
        sampler=None,
        collate_fn=fast_collate
    )

    model = Generalized_CNN()
    model.to(torch.device("cpu"))
    print(model)

    criterion = nn.CrossEntropyLoss().to(torch.device("cpu"))

    optimizer = torch.optim.SGD(model.parameters(), lr=solver['BASE_LR'], momentum=solver['MOMENTUM'])

    train(model, criterion, train_loader, optimizer)


if __name__ == '__main__':
    main(sys.argv)
