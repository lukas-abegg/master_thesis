import torch
from torch import nn


'''
GRU gating layer used in Stabilizing transformers in RL.
Note that all variable names follow the notation from section: "Gated-Recurrent-Unit-type gating" 
in https://arxiv.org/pdf/1910.06764.pdf
'''


class GRUGate(nn.Module):

    def __init__(self, d_model):
        # d_model is dimension of embedding for each token as input to layer (want to maintain this in the gate)
        super(GRUGate, self).__init__()

        # TODO: DEBUG Make sure intitialize bias of linear_w_z to -3
        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model,
                                    d_model)  ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)  # Manually setting this bias to allow starting with markov process
            # Note -2 is the setting used in the paper stable transformers

    def forward(self, x, y):
        ### Here x,y follow from notation in paper
        # TODO: DEBUG MAKE SURE THIS IS APPLIED ON PROPER AXIS
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))  # MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r * x))  # Note elementwise multiplication of r and x
        return (1. - z) * x + z * h_hat
