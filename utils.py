import torch
import torch.nn as nn
import torch.nn.functional as F
from _collections import OrderedDict
from bisect import bisect_right
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # print(output, target)
        maxk = max(topk)
        # print(maxk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)

        res = []
        for k in topk:
            # print(k)
            # print(correct[:k])
            # print(correct[:k].view(-1))
            # print(correct[:k].view(-1).float())
            # print(correct[:k].view(-1).float().sum(0, keepdim=True))
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # print(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.mul_(100.0 / batch_size))
        # print('====', res)
        return res, pred

def mismatch_params_filter(s):
    l = []
    for i in s:
        if i.split('.')[-1] in ['num_batches_tracked', 'running_mean', 'running_var']:
            continue
        else:
            l.append(i)
    return l

def strip_prefix_if_present(state_dict, prefix='module.'):
    """
    This function is taken from the maskrcnn_benchmark repo.
    It can be seen here:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/model_serialization.py
    """
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped_state_dict[key[len(prefix):]] = value
            # stripped_state_dict[key.replace(prefix, "")] = value
        else:
            pass
    return stripped_state_dict

def align_and_update_state_dicts(model_state_dict, weights_dict):
    """
    This function is taken from the maskrcnn_benchmark repo.
    It can be seen here:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/model_serialization.py

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(list(model_state_dict.keys()))
    weights_keys = sorted(list(weights_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in model_keys for j in weights_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(weights_keys))
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size_model = max([len(key) for key in model_keys]) if model_keys else 1
    max_size_weights = max([len(key) for key in weights_keys]) if weights_keys else 1
    match_keys = set()
    for idx_model, idx_weights in enumerate(idxs.tolist()):
        if idx_weights == -1:
            continue
        key_model = model_keys[idx_model]
        key_weights = weights_keys[idx_weights]
        model_state_dict[key_model] = weights_dict[key_weights]
        match_keys.add(key_model)
        print(
            '{: <{}} loaded from {: <{}} of shape {}'.format(key_model, max_size_model, key_weights, max_size_weights,
                                                             tuple(weights_dict[key_weights].shape))
        )
    mismatch_keys = set(model_keys) - match_keys
    return model_state_dict, mismatch_keys

def load_weights(model, weights_path):
    try:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))['model']
    except:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
    model_state_dict = model.state_dict()
    model_state_dict, mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict)
    model.load_state_dict(model_state_dict)
    print('The mismatch keys: {}.'.format(list(mismatch_params_filter(sorted(mismatch_keys)))))
    print('Loading from weights: {}.'.format(weights_path))

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def make_norm(c, norm='bn', group=1, eps=1e-5):
    if norm == 'bn':
        return nn.BatchNorm2d(c, eps=eps)
    elif norm == 'gn':
        assert c % group == 0
        return nn.GroupNorm(group, c, eps=eps)
    elif norm == 'none':
        return None
    else:
        return nn.BatchNorm2d(c, eps=eps)

def get_lr(epoch, solver):
    base_lr = solver['BASE_LR']
    new_lr = base_lr * solver['GAMMA'] ** bisect_right(solver['STEPS'], epoch)
    print('Change learning rate from {} to {} at epoch {}'.format(base_lr, new_lr, epoch))
    return new_lr