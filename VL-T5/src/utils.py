import re
import numpy as np
import torch
import collections
import logging
import torch.nn.functional as F
import torch.nn as nn

import os
import json
import errno
from collections import defaultdict, Counter
def get_area(pos):
    """
    Args
        pos: [B, N, 4]
            (x1, x2, y1, y2)

    Return
        area : [B, N]
    """
    # [B, N]
    height = pos[:, :, 3] - pos[:, :, 2]
    width = pos[:, :, 1] - pos[:, :, 0]
    area = height * width
    return area

def get_relative_distance(pos):
    """
    Args
        pos: [B, N, 4]
            (x1, x2, y1, y2)

    Return
        out : [B, N, N, 4]
    """
    # B, N = pos.size()[:-1]

    # [B, N, N, 4]
    relative_distance = pos.unsqueeze(1) - pos.unsqueeze(2)

    return relative_distance


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_iou(anchors, gt_boxes):
    """
    anchors: (N, 4) torch floattensor
    gt_boxes: (K, 4) torch floattensor
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)

    if gt_boxes.size() == (4,):
        gt_boxes = gt_boxes.view(1, 4)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
        (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) *
        (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))




def assert_array_eq(real, expected):
    EPS = 1e-7
    assert (np.abs(real-expected) < EPS).all(), \
        '{} (true) vs {} (expected)'.format(real, expected)


def json_keys2int(x):
    return {int(k): v for k, v in x.items()}


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def append_bias(train_dset, eval_dset, answer_voc_size):
    """
        Compute the bias:
        The bias here is just the expected score for each answer/question type
    """
    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

class Tracker:
    """ Keep track of results over time, while having access to
        monitors to display information about them.
    """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value