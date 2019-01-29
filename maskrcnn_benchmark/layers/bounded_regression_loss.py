# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import pdb
# TODO maybe push this to nn?
def bounded_regression_loss(input_t, input_s, target, beta=0.05, size_average=True):
    nt = torch.norm(input_t - target,dim=1)
    ns = torch.norm(input_s - target,dim=1)
    cond = ns - beta > nt
    loss = torch.where(cond, ns, torch.full_like(ns, 0))
    if size_average:
        return loss.mean()
    return loss.sum()
