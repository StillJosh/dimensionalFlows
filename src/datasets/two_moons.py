# half_moons.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 19.03.24
import normflows as nf
import torch.utils.data
from sklearn.decomposition import PCA
from src.utils import utils
from sklearn.datasets import make_moons

class TwoMoons(torch.utils.data.Dataset):

    def __init__(self, num_samples):


        self.num_samples = num_samples

        self.target = None
        self.data = make_moons(n_samples=num_samples, noise=0.1)[0]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data_reduced = {self.data.shape[-1]: self.data}
        self.data_reduced_train = self.data_reduced

        self.return_dim = 2


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_reduced[self.return_dim][idx]
