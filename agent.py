import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class MlpAgent(nn.Module):
    def __init__(self, ob_space, ac_space): #pylint: disable=W0613

        super(MlpAgent, self).__init__()
        self.initial_state = None
        self.output_shape = ac_space
        self.input_shape = ob_space

        self.pi_h = nn.Sequential(
                                 nn.Linear(ob_space, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 )
        self.mean = nn.Linear(64, ac_space)
        self.logstd = nn.Parameter(torch.zeros(ac_space))

        self.vf_h = nn.Sequential(
                               nn.Linear(ob_space, 64),
                               nn.Tanh(),
                               nn.Linear(64, 64),
                               nn.Tanh(),
                               )
        self.vf = nn.Linear(64, 1)

        self.init_()
        self.pd = None
        self.v = None

    def init_(self):
        for m in self.pi_h:
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                for name, param in m.named_parameters():
                    if name == 'weight':
                        nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
                    if name == 'bias':
                        nn.init.constant_(param, 0.0)

        for name, param in self.mean.named_parameters():
            if name == 'weight':
                nn.init.orthogonal_(param, gain=0.01)
            if name == 'bias':
                nn.init.constant_(param, 0.0)

        nn.init.constant_(self.logstd, 0.0)

        for m in self.vf_h:
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                for name, param in m.named_parameters():
                    if name == 'weight':
                        nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
                    if name == 'bias':
                        nn.init.constant_(param, 0.0)

        for name, param in self.vf.named_parameters():
            if name == 'weight':
                nn.init.orthogonal_(param, gain=1.0)
            if name == 'bias':
                nn.init.constant_(param, 0.0)

    def forward(self, obs):
        mean = self.mean(self.pi_h(obs))
        logstd = self.logstd
        std = torch.exp(logstd)
        self.pd = Normal(mean, std)

        vf = self.vf(self.vf_h(obs))
        self.v = torch.sum(vf, dim=1) # vf has a shape of (nenv, 1), sum operation is just to reduct the last dimension of '1', for compatibility between nenv = 1 and nenv > 1.

    def statistics(self, obs, A):
        self.forward(obs)
        return torch.sum(- self.pd.log_prob(A), dim=1), torch.sum(self.pd.entropy(), dim=1), self.v

    def step(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs).float()
        with torch.no_grad():
            self.forward(obs)
            act = self.pd.sample()
            neglogp = torch.sum(-self.pd.log_prob(act), dim=1)
        return act.numpy(), self.v.numpy(), self.initial_state, neglogp.numpy()
