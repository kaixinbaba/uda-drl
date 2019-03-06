import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):

    def __init__(self, n_s, n_a, a_bound):
        super(Actor, self).__init__()
        h1 = 128
        h2 = 64
        self.a_bound = a_bound
        self.hidden1 = torch.nn.Linear(n_s, h1)
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2 = torch.nn.Linear(h1, h2)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.out = torch.nn.Linear(h2, n_a)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = torch.tanh(self.out(x))
        result = x * self.a_bound
        return result


class Critic(torch.nn.Module):

    def __init__(self, n_s, n_a):
        super(Critic, self).__init__()
        h1 = 128
        h2 = 64
        self.hidden1 = torch.nn.Linear(n_s + n_a, h1)
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2 = torch.nn.Linear(h1, h2)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.out = torch.nn.Linear(h2, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = F.relu(self.hidden1(torch.cat((s, a), 1)))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
