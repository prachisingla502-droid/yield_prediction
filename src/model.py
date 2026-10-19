"""YieldNet: a feed-forward neural network for regression.

Difference from a classifier (viva question 1): the last layer is ONE neuron
with NO activation (identity), so it can output any real number. A classifier
would end in K neurons + softmax (or 1 neuron + sigmoid) and output class
probabilities, trained with cross-entropy. Here we train with mean squared error.
"""
import torch
from torch import nn


class YieldNet(nn.Module):
    def __init__(self, n_features, hidden_layers=(64, 32)):
        super().__init__()
        layers, width = [], n_features
        for h in hidden_layers:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))   # regression head: 1 neuron, linear output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
