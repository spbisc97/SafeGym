# @title SACAgent
import torch
from torch import nn, optim
import numpy as np
from safegym.models.nets.nets import Actor, Critic
from safegym.models.replaybuffer.replaybuffer import ReplayBuffer


class SACAgent:
    def __init__(
        self,
        env,
        gamma,
        tau,
        alpha,
        actor_learning_rate,
        critic_learning_rate,
        buffer_size,
        auto_entropy_tuning=True,
        gradient_steps=1,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau
        self.log_alpha = np.log(alpha)
        self.gradient_steps = gradient_steps

        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.target_critic1 = Critic(self.obs_dim, self.action_dim).to(
            self.device
        )
        self.target_critic2 = Critic(self.obs_dim, self.action_dim).to(
            self.device
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=critic_learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=critic_learning_rate
        )
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.auto_entropy_tuning = auto_entropy_tuning
        self.alpha_optimizer = None

        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.001)
        else:
            self.log_alpha = np.log(alpha)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state=state)
        action = action.cpu().detach().numpy()
        return action[0]

    def train(self, batch_size):
        if self.replay_buffer.__len__() < batch_size:
            print("returned")
            return
        for _ in range(self.gradient_steps):
            state, action, reward, next_state, done = (
                self.replay_buffer.sample(batch_size)
            )
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = (
                torch.FloatTensor(np.float32(done))
                .unsqueeze(1)
                .to(self.device)
            )
            with torch.no_grad():
                next_state_action, next_state_log_pi = self.actor.sample(
                    next_state
                )
                q1_next, q2_next = (
                    self.critic1(next_state, next_state_action),
                    self.critic2(next_state, next_state_action),
                )
                q_next = torch.min(q1_next, q2_next)
                q_next_vals = q_next - self.alpha * next_state_log_pi
                target_q = reward + (1 - done) * self.gamma * q_next_vals
            q1, q2 = (self.critic1(state, action), self.critic2(state, action))
            q1_loss = nn.functional.mse_loss(q1, target_q)
            q2_loss = nn.functional.mse_loss(q2, target_q)

            self.critic1_optimizer.zero_grad()
            q1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            q2_loss.backward()
            self.critic2_optimizer.step()
            pi, log_pi = self.actor.sample(state)
            q1_pi, q2_pi = (self.critic1(state, pi), self.critic2(state, pi))
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = -(q_pi - self.alpha * log_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if self.auto_entropy_tuning and self.alpha_optimizer is not None:
                alpha_loss = -(
                    self.alpha * (log_pi + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            for param, target_param in zip(
                self.critic1.parameters(), self.target_critic1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic2.parameters(), self.target_critic2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_alpha(self):
        return self.alpha
