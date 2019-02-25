import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from compat import use_gpu, FloatTensor, LongTensor, numpy
from memory import Memory
from models import Net


class Agent(object):

    def __init__(self,
                 n_s,
                 n_a,
                 hiddens=(128, 64),
                 epsilon=1.0,
                 epsilon_min=0.005,
                 epsilon_decay=0.05,
                 gamma=0.99,
                 batch_size=64,
                 memory_capacity=100000,
                 lr=0.001,
                 is_dueling=False,
                 is_prioritize=True,
                 replace_iter=100,
                 is_soft=False,
                 tau=0.01,
                 e=0.01,
                 a=0.6,
                 b=0.4):
        self.n_s = n_s
        self.n_a = n_a
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replace_iter = replace_iter
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.is_soft = is_soft
        self.is_prioritize = is_prioritize
        self.tau = tau
        if use_gpu:
            self.eval_net = Net(n_s, n_a, hiddens=hiddens, is_dueling=is_dueling).cuda()
            self.target_net = Net(n_s, n_a, hiddens=hiddens, is_dueling=is_dueling).cuda()
        else:
            self.eval_net = Net(n_s, n_a, hiddens=hiddens, is_dueling=is_dueling)
            self.target_net = Net(n_s, n_a, hiddens=hiddens, is_dueling=is_dueling)
        if is_prioritize:
            self.memory = Memory(memory_capacity, e, a, b)
        else:
            self.memory = np.zeros((memory_capacity, self.n_s * 2 + 2))
        self.memory_count = 0
        self.learn_count = 0

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def act(self, s):
        if np.random.random() <= self.epsilon:
            # random
            return np.random.randint(self.n_a)
        else:
            # max
            s = FloatTensor(s)
            action_value = self.eval_net(s)
            a = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
            return a

    def step(self, s, a, r, s_, done):
        if self.is_prioritize:
            # experience = s, a, r, s_, done
            experience = np.hstack((s, [a, r], s_))
            self.memory.store(experience)
            self.memory_count += 1
            if np.count_nonzero(self.memory.tree.tree) > self.batch_size:
                tree_idx, batch, ISWeights_mb = self.memory.sample(
                    self.batch_size)
                self.learn(batch, tree_idx, ISWeights_mb)
        else:
            transition = np.hstack((s, [a, r], s_))
            # replace the old memory with new memory
            index = self.memory_count % self.memory_capacity
            self.memory[index, :] = transition
            self.memory_count += 1
            if self.memory_count < self.memory_capacity:
                return
            # sample batch transitions
            sample_index = np.random.choice(self.memory_capacity, self.batch_size)
            batch = self.memory[sample_index, :]
            self.learn(batch)

    def learn(self, batch, tree_idx=None, ISWeights_mb=None):
        b_s = torch.squeeze(FloatTensor(batch[:, :self.n_s]), 0)
        b_a = torch.squeeze(LongTensor(batch[:, self.n_s:self.n_s + 1]), 0)
        b_r = torch.squeeze(FloatTensor(batch[:, self.n_s + 1:self.n_s + 2]), 0)
        b_s_ = torch.squeeze(FloatTensor(batch[:, -self.n_s:]), 0)
        temp = self.eval_net(b_s)
        eval_q = torch.gather(temp, 1, b_a)
        next_max_from_eval = self.eval_net(b_s_)
        next_max_from_eval_index = next_max_from_eval.max(1)[1].unsqueeze(1)
        next_actions = self.target_net(b_s_).detach()
        next_max = next_actions.gather(1, next_max_from_eval_index)
        target_q = b_r + self.gamma * next_max  # * (1 - b_done)
        abs_errors = numpy(torch.sum(torch.abs(target_q - eval_q), dim=1))
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.is_prioritize:
            self.memory.batch_update(tree_idx=tree_idx, abs_errors=abs_errors)
        self.update()
        self.learn_count += 1

    def update(self):
        next_epsilon = self.epsilon * self.epsilon_decay
        if next_epsilon <= self.epsilon_min:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = next_epsilon
        if self.is_soft:
            for target_param, local_param in zip(
                    self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            if self.learn_count % self.replace_iter == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())

    # save all net
    def save(self, name):
        torch.save(self.eval_net, name)

    # load all net
    def load(self, name):
        return torch.load(name)
