import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset
import torch

class BSDS300(Dataset):

    def __init__(self, **kwargs):

        self.root = Path('/Users/joshuastiller/Code/private/data/data')
        f = h5py.File(self.root / 'BSDS300/BSDS300.hdf5', 'r')

        self.x_train = torch.tensor(f['train'], dtype=torch.float32)
        self.x_val = torch.tensor(f['validation'], dtype=torch.float32)
        self.x_test = torch.tensor(f['test'], dtype=torch.float32)

        self.n_dims = self.x_train.shape[1]

        self.phase = 'train'

        self.data_reduced_train = {self.x_train.shape[-1]: self.x_train}
        self.data_reduced_val = {self.x_val.shape[-1]: self.x_val}
        self.data_reduced_test = {self.x_test.shape[-1]: self.x_test}

        self.return_dim = self.data.shape[-1]

    @property
    def data(self):
        if self.phase == 'train':
            return self.x_train
        elif self.phase == 'val':
            return self.x_val
        elif self.phase == 'test':
            return self.x_test
        else:
            raise ValueError('Invalid phase')

    @property
    def data_reduced(self):
        if self.phase == 'train':
            return self.data_reduced_train
        elif self.phase == 'val':
            return self.data_reduced_val
        elif self.phase == 'test':
            return self.data_reduced_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_reduced[self.return_dim][idx]



def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    # print data.head()
    # data = data.as_matrix()
    # # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    #
    # i = 0
    # # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)

    data = np.load(root_path)
    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):

    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test
