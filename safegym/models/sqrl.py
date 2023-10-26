import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from safegym.models.nets.nets import (
    Critic,
    SafetyCritic,
    Actor,
)
from safegym.models.replaybuffer.replaybuffer import ReplayBuffer


# @title SQRLAgent
class SQRLAgent:
    def __init__(
        self,
        env,
        gamma,
        safe_gamma,
        tau,
        alpha,
        epsilon_safe,
        actor_learning_rate,
        critic_learning_rate,
        buffer_size,
        auto_entropy_tuning=True,
        nu=0.1,
        auto_nu_tuning=True,
        gradient_steps=1,
        target_entropy="auto",
        actor_hidden_size=256,
        critic_hidden_size=256,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.gamma = gamma
        self.safe_gamma = safe_gamma
        self.tau = tau
        self.epsilon_safe = epsilon_safe
        self.gradient_steps = gradient_steps

        self.actor = Actor(
            self.obs_dim, self.action_dim, hidden_size=actor_hidden_size
        ).to(self.device)
        self.critic1 = Critic(
            self.obs_dim, self.action_dim, hidden_size=critic_hidden_size
        ).to(self.device)
        self.critic2 = Critic(
            self.obs_dim, self.action_dim, hidden_size=critic_hidden_size
        ).to(self.device)

        self.target_critic1 = Critic(
            self.obs_dim, self.action_dim, hidden_size=critic_hidden_size
        ).to(self.device)
        self.target_critic2 = Critic(
            self.obs_dim, self.action_dim, hidden_size=critic_hidden_size
        ).to(self.device)

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

        self.safety_critic = SafetyCritic(
            self.obs_dim, self.action_dim, hidden_size=critic_hidden_size
        ).to(self.device)
        self.safety_critic_optimizer = optim.Adam(
            params=self.safety_critic.parameters(), lr=critic_learning_rate
        )

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.auto_entropy_tuning = auto_entropy_tuning
        self.auto_nu_tuning = auto_nu_tuning

        # add entropy tuning

        if self.auto_entropy_tuning:
            if target_entropy == "auto":
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.env.action_space.shape).to(self.device)
                ).item()
            else:
                self.target_entropy = (
                    torch.Tensor((target_entropy,)).to(self.device).item()
                )
                # set the target to -1 for each action

            self.log_alpha = torch.tensor(
                np.log(alpha), requires_grad=True, device=self.device
            )  # maybe init with sme value
            # one alpha for all
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=critic_learning_rate
            )  # dnk needs square
        else:
            self.log_alpha = torch.tensor(
                np.log(alpha), requires_grad=False, device=self.device
            )
        nu = nu if nu > 0 else 0.1
        self.log_nu = torch.tensor(
            (np.log(nu)), requires_grad=self.auto_nu_tuning, device=self.device
        )
        if self.log_nu.requires_grad:
            self.log_nu_optimizer = optim.Adam(
                [self.log_nu], lr=critic_learning_rate
            )

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)

        # If action is a tensor, convert it to a numpy array.
        action = action.cpu().detach().numpy()
        return action[0]

    def train(self, batch_size, pretrain=False):
        if pretrain:
            self.pretrain(batch_size)
        else:
            self.finetune(batch_size)
        pass

    def pretrain(self, batch_size):
        if self.replay_buffer.__len__() < batch_size:
            print("returned")
            return
        for it in range(self.gradient_steps):
            # step 1 sample a batch
            state, action, reward, next_state, done, unsafe = (
                self.replay_buffer.safety_sample(batch_size)
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
            unsafe = (
                torch.FloatTensor(np.float32(unsafe))
                .unsqueeze(1)
                .to(self.device)
            )

            # compute target q val
            with torch.no_grad():
                # get policy action and compute target value
                next_state_action, next_state_log_pi = self.actor.sample(
                    next_state
                )
                q1_next, q2_next = self.critic1(
                    next_state, next_state_action
                ), self.critic2(next_state, next_state_action)
                q_next = torch.min(q1_next, q2_next)
                # no entropy term
                q_next_vals = q_next - self.alpha * next_state_log_pi
                target_q = reward + (1 - done) * self.gamma * q_next_vals

            # update the critics
            q1, q2 = self.critic1(state, action), self.critic2(
                state, action
            )  # current q values
            # q1_loss = 0.5 * (q1-target_q.detach()).pow(2).mean()
            q1_loss = nn.functional.mse_loss(q1, target_q)  # sum?
            # q2_loss = 0.5 * (q2-target_q.detach()).pow(2).mean()
            q2_loss = nn.functional.mse_loss(q2, target_q)

            self.critic1_optimizer.zero_grad()
            q1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            q2_loss.backward()
            self.critic2_optimizer.step()
            # finish update the critics

            # update the actior , shoud i do if after the critic? yes
            pi, log_pi = self.actor.sample(state)
            q1_pi, q2_pi = self.critic1(state, pi), self.critic2(
                state, pi
            )  # maybe use the value taken before,but sb3 and algo calculates those here!
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = -(q_pi - self.alpha * log_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # finish update teh actor

            # update entropy coeff
            if self.auto_entropy_tuning:
                # keep the log_pi from before?
                alpha_loss = -(
                    self.log_alpha * (log_pi + self.target_entropy).detach()
                ).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # soft update target
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

            # update safety
            predicted_safety = self.safety_critic(state, action)
            known_safety = self.safety_critic(next_state, next_state_action)
            target_safety = (
                unsafe + (1 - unsafe) * self.safe_gamma * known_safety
            )
            safe_loss = nn.functional.mse_loss(predicted_safety, target_safety)

            self.safety_critic_optimizer.zero_grad()
            safe_loss.backward()
            self.safety_critic_optimizer.step()

    def finetune(self, batch_size):
        if self.replay_buffer.__len__() < batch_size:
            print("returned")
            return
        for it in range(self.gradient_steps):
            # step 1 sample a batch
            state, action, reward, next_state, done, unsafe = (
                self.replay_buffer.safety_sample(batch_size)
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
            unsafe = (
                torch.FloatTensor(np.float32(unsafe))
                .unsqueeze(1)
                .to(self.device)
            )

            # compute target q val
            with torch.no_grad():
                # get policy action and compute target value
                next_state_action, next_state_log_pi = self.actor.sample(
                    next_state
                )
                q1_next, q2_next = self.critic1(
                    next_state, next_state_action
                ), self.critic2(next_state, next_state_action)
                q_next = torch.min(q1_next, q2_next)
                # no entropy term
                q_next_vals = q_next - self.alpha * next_state_log_pi
                target_q = reward + (1 - done) * self.gamma * q_next_vals

            # update the critics
            q1, q2 = self.critic1(state, action), self.critic2(
                state, action
            )  # current q values
            # q1_loss = 0.5 * (q1-target_q.detach()).pow(2).mean()
            q1_loss = nn.functional.mse_loss(q1, target_q)  # sum?
            # q2_loss = 0.5 * (q2-target_q.detach()).pow(2).mean()
            q2_loss = nn.functional.mse_loss(q2, target_q)

            self.critic1_optimizer.zero_grad()
            q1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            q2_loss.backward()
            self.critic2_optimizer.step()
            # finish update the critics

            # update the actior , shoud i do if after the critic? yes
            pi, log_pi = self.actor.sample(state)
            q1_pi, q2_pi = self.critic1(state, pi), self.critic2(
                state, pi
            )  # maybe use the value taken before,but sb3 and algo calculates those here!
            q_pi = torch.min(q1_pi, q2_pi)
            predicted_safety = self.safety_critic(state, action)
            # target entropy term is constant
            # but also epsiln safe is constant
            # +self.target_entropy self.epsilon_safe
            actor_loss = -(
                +q_pi
                - self.alpha * (log_pi)  # +self.target_entropy
                + self.nu * (self.epsilon_safe - predicted_safety).detach()
            ).mean()
            # unpack -q_pi       +self.alpha*(log_pi+target_entropy)
            # +self.nu*(self.epsiolon_safe-predicted_safety)
            # i could probably just use this loss fo all! with just one backward?

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # finish update teh actor

            # update entropy coeff
            if self.auto_entropy_tuning:
                # keep the log_pi from before?
                alpha_loss = -(
                    self.log_alpha * (log_pi + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # update ni param
            if self.auto_nu_tuning:
                nu_loss = (
                    self.log_nu
                    * ((self.epsilon_safe) - (predicted_safety)).detach()
                ).mean()  # log nu would be right?
                self.log_nu_optimizer.zero_grad()
                nu_loss.backward()
                self.log_nu_optimizer.step()

            # soft update target
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

            # update safety
            # predicted_safety = self.safety_critic(state,action) already done before in the algo
            known_safety = self.safety_critic(next_state, next_state_action)
            target_safety = (
                unsafe + (1 - unsafe) * self.safe_gamma * known_safety
            )
            safe_loss = nn.functional.mse_loss(predicted_safety, target_safety)

            self.safety_critic_optimizer.zero_grad()
            safe_loss.backward()
            self.safety_critic_optimizer.step()

    def update_replay_buffer(
        self, state, action, reward, next_state, done, unsafe
    ):
        self.replay_buffer.push(
            state, action, reward, next_state, done, unsafe
        )

    @property
    def nu(self):
        return self.log_nu.exp()

    @property
    def alpha(self):
        return self.log_alpha.exp()
