import numpy as np


def choose(preds):
    ret = list()
    test_len = len(preds[0])
    for i in range(test_len):
        dis = 10000
        choice = 0
        for j in range(3):
            distance = np.sqrt(
                np.sum(np.square(preds[j][i] - preds[(j + 1) % 2][i])))
            if distance < dis:
                dis = distance
                choice = j  # 0: 0 to 1, 1: 1 to 2, 2: 2 to 0
        ret.append((preds[choice][i] + preds[(choice + 1) % 2][i]) / 2)
    return np.array(ret)


a = np.random.rand(10, 3)
b = np.random.rand(10, 3)
c = np.random.rand(10, 3)
print(choose([a, b, c]))

# imbalance problem
# two classes label: discourse_effectiveness, discourse_type
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=30)


class Balanced_CE_loss(torch.nn.Module):

    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        for i in range(input.shape[0]):
            beta = 1 - torch.sum(target[i]) / target.shape[1]
            for j in range(input.shape[1]):
                loss += -(beta * target[i][j] * torch.log(input[i][j]) +
                          (1 - beta) *
                          (1 - target[i][j]) * torch.log(1 - input[i][j]))
        return loss


discourse_types = ['']