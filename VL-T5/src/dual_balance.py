import argparse
import math
import os
import random
import shutil
import time
import torch
import json
import torch.nn
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import geotorch
class BCP(nn.Module):

    def __init__(self,feature_num, class_num):
        super(BCP, self).__init__()
        self.feature_num = feature_num
        self.class_num = class_num
        self.rotate = nn.Linear(class_num, feature_num, bias=False)
        self.register_buffer("ETF", self.generate_ETF(dim=class_num))
        geotorch.orthogonal(self.rotate, "weight")
    def generate_ETF(self, dim):
        return torch.eye(dim, dim) - torch.ones(dim, dim) / dim
    def prototypes(self):
        p = self.rotate.weight @ self.ETF
        p = p.cuda()
        return p
    def forward(self, x):
        logit = x @ self.rotate.weight @ self.ETF
        return logit

class MFA(nn.Module):

    def __init__(self, feat_in, eps, max_iter, dis, reduction='none'):
        super(MFA, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis
        self.BN_H = nn.BatchNorm1d(feat_in)

    def feature(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x

    def forward(self, x, y):

        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8).cuda()
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if self.dis == 'cos':
            E = 1 - d_cosine(x_col.cuda(), y_lin.cuda())
        elif self.dis == 'euc':
            E = torch.mean((torch.abs(x_col - y_lin)) ** 2, -1)

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).cuda().squeeze()

        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).cuda().squeeze()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()

        actual_nits = 0
        thresh = 1e-1

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(E, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(E, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(E, U, V))
        cost = torch.sum(pi * E, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    def M(self, E, u, v):

        return (-E + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

class cross_entropy_loss(nn.Module):
    '''
    multi-label classification with cross entropy loss
    '''
    def __init__(self):
        super().__init__()
        pass
    def forward(self,logits,labels):
        nll = F.log_softmax(logits,dim=-1)
        loss = -nll * labels
        return loss.sum(dim=-1).mean()


