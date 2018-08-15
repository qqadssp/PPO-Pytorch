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

        self.mean = nn.Sequential(
                                 nn.Linear(ob_space, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, ac_space),
                                 )
        self.logstd = nn.Parameter(torch.zeros(ac_space))

        self.vf = nn.Sequential(
                               nn.Linear(ob_space, 64),
                               nn.Tanh(),
                               nn.Linear(64, 64),
                               nn.Tanh(),
                               nn.Linear(64, 1),
                               )

        self.pd = None
        self.v = None

    def forward(self, obs):
        mean = self.mean(obs)
        logstd = self.logstd
        std = torch.exp(logstd)
        self.pd = Normal(mean, std)

        vf = self.vf(obs)
        self.v = vf.squeeze()

    def statistics(self, obs, A):
        self.forward(obs)
        return torch.sum(- self.pd.log_prob(A), dim=1), torch.sum(self.pd.entropy(), dim=1), self.v

    def step(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        with torch.no_grad():
            self.forward(obs)
            act = self.pd.sample()
            neglogp = torch.sum(-self.pd.log_prob(act), dim=1)
        return act.numpy(), self.v.numpy(), self.initial_state, neglogp.numpy()


class DiagGaussianPd(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)
    def neglogp(self, x):
        return 0.5 * torch.sum(torch.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * troch.to_float(tf.shape(x)[-1]) \
               + troch.sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return torch.sum(other.logstd - self.logstd + (torch.square(self.std) + torch.square(self.mean - other.mean)) / (2.0 * torch.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return torch.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * torch.random_normal(torch.shape(self.mean))
    def logp(self, x):
        return - self.neglogp(x)

