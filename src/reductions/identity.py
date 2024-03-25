# identity.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 24.03.24
import numpy as np


class IdentityReduction:
    """
    A reduction that does nothing.
    """

    def __init__(self):
        self.components_ = None

    def fit(self, x):
        self.components_ = np.eye(x.shape[1])


    def fit_transform(self, x):
        self.fit(x)
        return x

    def transform(self, x):

        return x