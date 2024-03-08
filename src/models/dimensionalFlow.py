# dimensionalFlow.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24

import torch
from torch import nn


class DimensionalFlow(nn.Module):

    def __init__(self, flows, flow_dims):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.flow_dims = flow_dims + [0]
        self.train_flow = 1

    def forward_kld(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z0 = torch.zeros_like(x, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            z = z[:, :self.flow_dims[i - 1]]
            z0 = z0[:, self.flow_dims[i - 1]:]

        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def part_forward_kld(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z0 = torch.zeros((len(x), self.flow_dims[self.train_flow-1]), device=x.device)
        z = x
        for i in range(self.train_flow - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z[:, :self.flow_dims[i]])
            log_q += log_det

            z0[:, self.flow_dims[i - 1]:self.flow_dims[i]] = z[:, self.flow_dims[i - 1]:]

        log_q += self.flows[i].q0.log_prob(z0)
        return -torch.mean(log_q)

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        dim_x = x.shape[1]
        z = x
        for i in range(self.flow_dims.index(dim_x), -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det
        log_q += self.flows[i].q0.log_prob(z)
        return log_q
