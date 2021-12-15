import torch
import torch.nn as nn


class LinearModel(torch.nn.Module):
    # Building of the full model for constructing the extractor of features
    def __init__(self):

        super(LinearModel, self).__init__()
        self.calib_params = nn.Linear(1, 1)

    def forward(self, x):

        calib_x = self.calib_params(x)

        return calib_x

