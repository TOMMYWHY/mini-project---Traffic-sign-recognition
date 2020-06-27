import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, make_dataset, DatasetFolder

class ImageFolderList(DatasetFolder):
    def __init__(self, root_list, transform=None, target_transform=None, loader=default_loader):
        if not isinstance(root_list, (list, tuple)):
            raise RuntimeError("dataset_list should be a list of strings, got {}".format(root_list))

        super(ImageFolderList, self).__init__(
            root_list[0], loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform
        )
        if len(root_list) > 1:
            for root in root_list[1:]:
                classes, class_to_idx = self._find_classes(root)
                for k in class_to_idx.keys():
                    class_to_idx[k] += len(self.classes)
                samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
                self.classes += classes
                self.class_to_idx.update(class_to_idx)
                self.samples += samples
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = 64
    h = 64
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

class DataPreFetcher(object):
    def __init__(self, loader, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([means[0] * 255, means[1] * 255, means[2] * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([stds[0] * 255, stds[1] * 255, stds[2] * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target