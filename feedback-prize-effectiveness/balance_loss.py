import torch
import torch.nn as nn


class Balanced_CE_loss(torch.nn.Module):

    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1 - torch.sum(target[i]) / target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]))
            y = torch.max(torch.log(1 - input[i]), torch.tensor([-100.0]))
            l = -(beta * target[i] * x + (1 - beta) * (1 - target[i]) * y)
            loss += torch.sum(l)
        return loss
