import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_size, output_size, hiddens=(128, 64), activate_func=F.relu, is_dueling=False):
        super(Net, self).__init__()
        node_size = [input_size]
        node_size += hiddens
        node_size.append(output_size)
        self.node_size = node_size
        self.is_dueling = is_dueling
        for i in range(1, len(self.node_size) - 1):
            name = 'layer{}'.format(i)
            setattr(self, name, nn.Linear(node_size[i - 1], node_size[i]))
            getattr(self, name).weight.data.normal_(0, 0.1)
        self.A = nn.Linear(node_size[-2], node_size[-1])
        self.A.weight.data.normal_(0, 0.1)
        self.activate_func = activate_func
        if self.is_dueling:
            self.V = nn.Linear(node_size[-2], 1)
            self.V.weight.data.normal_(0, 0.1)

    def forward(self, x):
        input_from = x
        for i in range(1, len(self.node_size) - 1):
            input_from = self.activate_func(getattr(self, 'layer{}'.format(i))(input_from))
        if self.is_dueling:
            A = self.A(input_from)
            V = self.V(input_from)
            return V + (A - A.mean())
        else:
            return self.A(input_from)

