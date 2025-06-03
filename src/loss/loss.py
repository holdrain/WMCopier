import torch
import torch.nn.functional as F
from torch import nn

from loss.focal_frequency_loss import FocalFrequencyLoss


class DDIMLoss(nn.Module):
    def __init__(self, ffl_w, ffl_w_start_step, **kwargs):
        super().__init__()
        if ffl_w > 0.0:
            self.ffl_loss = FocalFrequencyLoss(**kwargs)
        self.ffl_w = ffl_w
        self.ffl_w_start_step = ffl_w_start_step

    def forward(self, step, pred, target, matrix=None, **kwargs):
        mse_loss = F.mse_loss(pred, target)
        if self.ffl_w > 0 and step >= self.ffl_w_start_step:
            ffl_loss = self.ffl_loss(pred, target, matrix=matrix)
            return mse_loss + self.ffl_w * ffl_loss,mse_loss,ffl_loss
        else:
            return mse_loss,mse_loss,torch.tensor(0)