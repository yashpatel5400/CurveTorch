#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np

import curvetorch as curve


def basic(x, y):
    return x * x + y * y


def rosenbrock(x, y):
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2


def quadratic(x, y):
    a = 1.0
    b = 1.0
    return (x ** 2) / a + (y ** 2) / b


cases = [
    (basic, (-0.3, 0.5), (0, 0)),
    # (rosenbrock, (1.5, 1.5), (1, 1)),
    # (quadratic, (0.5, 0.5), (0, 0)),
]


def ids(v):
    n = '{} {}'.format(v[0].__name__, v[1:])
    return n


optimizers = [
    (curve.CurveSGD, {'lr': 0.0015}, 100),
]


@pytest.mark.parametrize('case', cases, ids=ids)
@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_benchmark_function(case, optimizer_config):
    func, initial_state, min_loc = case
    optimizer_class, config, iterations = optimizer_config

    x = torch.Tensor(initial_state).requires_grad_(True)
    x_min = torch.Tensor(min_loc)
    optimizer = optimizer_class([x], **config)
    x0s = []
    x1s = []

    fs = []
    for _ in range(iterations):
        x0, x1 = x
        
        fs.append(func(x0, x1))
        def closure():
            optimizer.zero_grad()
            f = func(x0, x1)
            f.backward(retain_graph=True, create_graph=True)
            return f
        optimizer.step(closure)

        x0s.append(float(x0.detach().numpy()))
        x1s.append(float(x1.detach().numpy()))

    plt.plot(range(iterations), fs)
    plt.show()

    f_vec = np.vectorize(func)
    a, b = np.meshgrid(np.linspace(-2, 2, 300),
                       np.linspace(-2, 2, 300))

    plt.contour(a, b, f_vec(a, b))

    for i in range(len(x0s)-1):
        plt.plot(x0s[i:i+2], x1s[i:i+2],
                 alpha=float(i) / (len(x0s)-1),
                 color="blue")
    plt.show()

    assert torch.allclose(x, x_min, atol=0.01)

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()
