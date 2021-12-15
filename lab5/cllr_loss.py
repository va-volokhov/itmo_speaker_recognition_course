import torch
import numpy as np
import torch.nn as nn


class CllrLoss(nn.Module):

    def __init__(self, ptar=0.01):
        '''
        Calibration loss
        https://github.com/alumae/sv_score_calibration/blob/master/calibrate_scores.py
        :param ptar:
        '''

        super(CllrLoss, self).__init__()
        self.ptar = ptar
        self.alpha = np.log(ptar / (1 - ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            """-log(sigmoid(log_odds))"""
            return torch.log1p(torch.exp(-lodds))


        return 0.5 * (self.ptar * torch.mean(negative_log_sigmoid(target_llrs + self.alpha)) + (1 - self.ptar) * torch.mean(
            negative_log_sigmoid(-(nontarget_llrs + self.alpha)))) / np.log(2)