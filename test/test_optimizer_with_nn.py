#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from torch import nn

import curvetorch as curve


def make_dataset(seed=42):
    rng = np.random.RandomState(seed)
    N = 100
    D = 2

    X = rng.randn(N, D) * 2

    # center the first N/2 points at (-2,-2)
    mid = N // 2
    X[:mid, :] = X[:mid, :] - 2 * np.ones((mid, D))

    # center the last N/2 points at (2, 2)
    X[mid:, :] = X[mid:, :] + 2 * np.ones((mid, D))

    # labels: first N/2 are 0, last N/2 are 1
    Y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    return x, y


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred


def ids(v):
    return '{} {}'.format(v[0].__name__, v[1:])

optimizers = [
    (curve.CurveSGD, {'lr': 0.002}, 8000),
]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(optimizer_config):
    torch.manual_seed(42)
    x_data, y_data = make_dataset()
    model = LogisticRegression()

    loss_fn = nn.BCELoss()
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)
    init_loss = None

    for _ in range(iterations):
        if init_loss is None:
            y_pred = model(x_data)
            loss = loss_fn(y_pred, y_data)
            init_loss = loss

        def closure():
            y_pred = model(x_data)
            loss = loss_fn(y_pred, y_data)
            optimizer.zero_grad()
            loss.backward(create_graph=True)
            return loss
        optimizer.step(closure)

    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)

    assert init_loss.item() > 2.0 * loss.item()