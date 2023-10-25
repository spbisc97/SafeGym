# @title CriticNets
from torch import nn
import torch


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        action = action.unsqueeze(-1) if action.dim() == 1 else action
        state_action = torch.cat([state, action], dim=1)
        return self.critic(state_action)


# @title SafeyNets


class SafetyCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(SafetyCritic, self).__init__()
        self.safety_critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Outputs a single safety value
            nn.Sigmoid(),  # Sigmoid squashes the safety value between 0 and 1
        )

    def forward(self, state, action):
        action = action.unsqueeze(-1) if action.dim() == 1 else action
        state_action = torch.cat([state, action], dim=1)
        return self.safety_critic(state_action)


# @title ActorNets
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256, std=-0.5):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)

    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp()
        return mean, std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()  # standard deviation
        normal = torch.distributions.Normal(
            mean, std
        )  # create a normal distribution
        z = normal.rsample()  # sample an action aroud the mean
        action = torch.tanh(
            z
        )  # restrict action space, needed for pendulum[-1,1]

        # Calculate log probability (log_prob)
        log_prob = normal.log_prob(z) - torch.log(
            1 - action.pow(2) + 1e-9
        )  # corrects the tanh
        # torch.log(1-0) gives 0, so no correction if we are in the center
        # torch.log(0-1e-9) gives really low negative number,
        log_prob = log_prob.sum(1, keepdim=True)  # single action here

        return action, log_prob
