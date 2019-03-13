from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from memorys import ReplayBuffer
from models import Actor, Critic

torch.manual_seed(7)
class DDPG(object):

    def __init__(self, n_s,
                 n_a,
                 a_bound,
                 gamma=0.99,
                 memory_size=10000,
                 tau=0.01,
                 lr_a=0.001,
                 lr_c=0.002,
                 batch_size=64,
                 var=3,
                 var_decay=0.9995
                 ):
        self.n_s = n_s
        self.n_a = n_a
        self.a_bound = a_bound
        self.gamma = gamma
        self.memory_size = memory_size
        self.tau = tau
        self.batch_size = batch_size
        self.var = var
        self.var_decay = var_decay

        # memory
        self.replay_buffer = ReplayBuffer(n_s, n_a, memory_size)
        # actor
        self.eval_actor = Actor(n_s, n_a, a_bound)
        self.target_actor = deepcopy(self.eval_actor)
        self.actor_optim = torch.optim.Adam(self.eval_actor.parameters(), lr=lr_a)

        # critic
        self.eval_critic = Critic(n_s, n_a)
        self.target_critic = deepcopy(self.eval_critic)
        self.critic_optim = torch.optim.Adam(self.eval_critic.parameters(), lr=lr_c)

    def choose_action(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        action = self.eval_actor(s).detach().numpy()[0]
        a = np.clip(np.random.normal(action, self.var), -self.a_bound, self.a_bound)
        return a

    def step(self, s, a, r, s_, done):
        self.store(s, a, r, s_, done)
        if self.replay_buffer.memory_count < self.memory_size:
            return
        # start learn
        self._learn()

    def _learn(self):
        # get batch
        mini_batch = self.replay_buffer.sample(self.batch_size)
        b_s = torch.FloatTensor(mini_batch[:, :self.n_s])
        b_a = torch.FloatTensor(mini_batch[:, self.n_s:self.n_s + self.n_a])
        b_r = torch.FloatTensor(mini_batch[:, self.n_s + self.n_a:self.n_s + self.n_a + 1])
        b_s_ = torch.FloatTensor(mini_batch[:, -self.n_s:])
        b_done = torch.FloatTensor(mini_batch[:, self.n_s + self.n_a + 1:self.n_s + self.n_a + 2])
        # learn
        self.update_critic(b_s, b_a, b_r, b_s_, b_done)
        self.update_actor(b_s)
        self.var *= self.var_decay

    def update_critic(self, s, a, r, s_, done):
        with torch.no_grad():
            target_next_a = self.target_actor(s_)
            next_a = self.target_critic(s_, target_next_a)
            target_q = r + self.gamma * next_a * (1.0 - done)
        eval_q = self.eval_critic(s, a)
        critic_loss = F.mse_loss(eval_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self._soft_update(self.eval_critic, self.target_critic)

    def update_actor(self, s):
        action = self.eval_actor(s)
        actor_loss = -self.eval_critic(s, action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._soft_update(self.eval_actor, self.target_actor)

    def store(self, s, a, r, s_, done):
        self.replay_buffer.store(s, a, r, s_, done)

    def _soft_update(self, eval_net, target_net):
        for eval, target in zip(eval_net.parameters(), target_net.parameters()):
            target.data.copy_(self.tau * eval.data + (1.0 - self.tau) * target.data)

    # save all net
    def save(self, name):
        torch.save(self.eval_actor, '{}_actor.pt'.format(name))
        torch.save(self.eval_critic, '{}_critic.pt'.format(name))

    # load all net
    def load(self, name):
        actor = torch.load('{}_actor.pt'.format(name))
        critic = torch.load('{}_critic.pt'.format(name))
        return actor, critic
