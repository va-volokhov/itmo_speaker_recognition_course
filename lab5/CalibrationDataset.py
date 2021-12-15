import torch

import numpy as np

from torch.utils.data import Dataset


class CalibrationDataset(Dataset):

    def __init__(self, target_scores, impostor_scores, N_parts=100):
        super(CalibrationDataset, self).__init__()

        self.target_scores = target_scores
        self.impostor_scores = impostor_scores
        self.L_tar = len(target_scores)
        self.L_imp = len(impostor_scores)
        self.N = N_parts
        self.n_tar_sampling = self.L_tar//N_parts
        self.n_imp_sampling = self.L_imp//N_parts


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        target_scores = self.target_scores[np.random.choice(self.L_tar, self.n_tar_sampling)]
        impostor_scores = self.impostor_scores[np.random.choice(self.L_imp, self.n_imp_sampling)]

        return target_scores, impostor_scores