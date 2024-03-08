# flows.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 08.03.24
import normflows as nf
import torch


def norm_flow1d(q0, latent_size, K=2):
    flows = []
    for i in range(K):
        flows += [nf.flows.Planar(1, act="leaky_relu")]

    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm


def resid_flow(q0, latent_size, K=2, hidden_units=128, hidden_layers=3):
    flows = []
    for i in range(K):
        net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                   init_zeros=True, lipschitz_const=0.9)
        flows += [nf.flows.Residual(net, reduce_memory=True)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm


def norm_flow(q0, latent_size, K=2):
    b = torch.Tensor([0 if i % 2 == 0 else 1 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm
