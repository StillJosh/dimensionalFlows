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
        self.flow_dims = [sum(flow_dims[:i + 1]) for i in range(len(flow_dims))]
        self.current_flow = 1

        self.pcas = {}

    def full_forward_kld(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            z0 = z[:, self.flow_dims[i - 1]:]
            z = z[:, :self.flow_dims[i - 1]]

            log_q += self.flows[i].q0.log_prob(z0)

        return -torch.mean(log_q)

    def forward_kld_alt(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(self.current_flow - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            z0 = z[:, self.flow_dims[i] - 1:]
            z = z[:, :self.flow_dims[i] - 1]

            log_q += self.flows[i].q0.log_prob(z0)

        return -torch.mean(log_q)

    def log_prob_alt(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(self.current_flow - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            z0 = z[:, self.flow_dims[i] - 1:]
            z = z[:, :self.flow_dims[i] - 1]

            log_q += self.flows[i].q0.log_prob(z0)

        return log_q

    def part_inverse(self, x, end_flow=None, with_grad=False):
        if end_flow is None:
            end_flow = self.current_flow

        z = x
        with torch.set_grad_enabled(with_grad):
            for i in range(len(self.flows) - 1, end_flow - 1, -1):
                z = z[:, :self.flow_dims[i]]
                z = self.flows[i].inverse(z)

        return z

    def forward_kld(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z = x

        for i in range(self.current_flow - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            if self.flow_dims[i] in self.pcas.keys():
                z = (z - z.mean(axis=0)) @ self.pcas[self.flow_dims[i]]

            if i > 0:
                z0 = z[:, self.flow_dims[i - 1]:]
                z = z[:, :self.flow_dims[i - 1]]
            else:
                z0 = z

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
        z = x

        with torch.no_grad():
            for i in range(len(self.flows) - 1, self.current_flow, -1):
                z = self.flows[i].inverse(z)
                z = z[:, :self.flow_dims[i] - 1]

        for i in range(self.current_flow - 1, -1, -1):
            z, log_det = self.flows[i].inverse_and_log_det(z)
            log_q += log_det

            if self.flow_dims[i] in self.pcas.keys():
                z = (z - z.mean(axis=0)) @ self.pcas[self.flow_dims[i]]

            if i > 0:
                z0 = z[:, self.flow_dims[i - 1]:]
                z = z[:, :self.flow_dims[i - 1]]
            else:
                z0 = z

            log_q += self.flows[i].q0.log_prob(z0)

        return log_q