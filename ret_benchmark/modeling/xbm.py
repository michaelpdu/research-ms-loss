# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import tqdm
from ret_benchmark.data.build import build_memory_data


class XBM:
    def __init__(self, cfg, model):
        self.use_gpu = torch.cuda.is_available()
        self.ratio = cfg.MEMORY.RATIO
        # init memory
        self.feats = list()
        self.labels = list()
        self.indices = list()
        model.train()
        for images, labels, indices in build_memory_data(cfg):
            with torch.no_grad():
                feat = model(images.cuda() if self.use_gpu else images)
                self.feats.append(feat)
                self.labels.append(labels.cuda() if self.use_gpu else labels)
                self.indices.append(indices.cuda() if self.use_gpu else indices)
        self.feats = torch.cat(self.feats, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.indices = torch.cat(self.indices, dim=0)
        # if memory_ratio != 1.0 -> random sample init queue_mask to mimic fixed queue size
        if self.ratio != 1.0:
            rand_init_idx = torch.randperm(int(self.indices.shape[0] * self.ratio)).cuda()
            self.queue_mask = self.indices[rand_init_idx]

    def enqueue_dequeue(self, feats, indices):
        self.feats.data[indices] = feats
        if self.ratio != 1.0:
            # enqueue
            self.queue_mask = torch.cat((self.queue_mask, indices.cuda() if self.use_gpu else indices), dim=0)
            # dequeue
            self.queue_mask = self.queue_mask[-int(self.indices.shape[0] * self.ratio):]

    def get(self):
        if self.ratio != 1.0:
            return self.feats[self.queue_mask], self.labels[self.queue_mask]
        else:
            return self.feats, self.labels
