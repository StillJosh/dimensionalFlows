# half_moons.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 19.03.24
import normflows as nf
import torch.utils.data
from sklearn.decomposition import PCA
from src.utils import utils


class HalfMoons(torch.utils.data.Dataset):

    def __init__(self, num_samples, sub_dims=None):
        if sub_dims is None:
            sub_dims = [1, 2]

        self.num_samples = num_samples

        self.target = nf.distributions.TwoModes(2, 0.1)
        self.data = utils.rejection_sampling_2d(lambda x: torch.exp(self.target.log_prob(x)), num_samples, (-3, 3),
                                                (-2, 2))

        self.data_reduced = {self.data.shape[-1]: self.data}

        self.return_dim = 2


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_reduced[self.return_dim][idx]
