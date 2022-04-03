#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np

import curvetorch as curve


def ackley(x, y):
    if isinstance(x, torch.Tensor):
        return -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2))) - torch.exp(0.5 * (torch.cos(2 * np.pi * x)+torch.cos(2 * np.pi * y))) + np.e + 20
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20

def basic(x, y):
    return x * x + y * y

def rosenbrock(x, y):
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2

def quadratic(x, y):
    a = 1.0
    b = 1.0
    return (x ** 2) / a + (y ** 2) / b

def camel(x, y): 
   return 4 * x ** 2 - 2.1 * x ** 4 + (x ** 6) / 3 + x * y - 4 * y ** 2 + 4 * y ** 4

def mccor(x, y):
    if isinstance(x, torch.Tensor):
        return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

def rastrigin(x, y):
    if isinstance(x, torch.Tensor):
        return (x ** 2 - 1 * torch.cos(2 * np.pi * x)) + \
            (y ** 2 - 1 * torch.cos(2 * np.pi * y)) + 2
    return (x ** 2 - 1 * np.cos(2 * np.pi * x)) + \
        (y ** 2 - 1 * np.cos(2 * np.pi * y)) + 2

cases = [
    (basic, (1.5, 1.5), (1, 1)),
    (ackley, (-0.3, 0.5), (0, 0)), # known failure case
    (rosenbrock, (1.5, 1.5), (1, 1)),
    (quadratic, (0.5, 0.5), (0, 0)),
    (camel, (0.5, 0.5), (-0.0898, 0.7126)),
    (mccor, (-0.5, -0.5), (-0.54719, -1.54719)),
    (rastrigin, (-0.5, -0.5), (0, 0)),
]


def ids(v):
    n = '{} {}'.format(v[0].__name__, v[1:])
    return n


optimizers = [
    (curve.CurveSGD, {'lr': 0.0015}, 15000),
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

    _, axs = plt.subplots(1, 2)

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

    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss vs. Iteration")
    axs[0].plot(range(iterations), fs)
    
    f_vec = np.vectorize(func)
    a, b = np.meshgrid(np.linspace(-2, 2, 300),
                       np.linspace(-2, 2, 300))

    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("Trajectory Plot")
    axs[1].contour(a, b, f_vec(a, b), levels=15)

    for i in range(len(x0s)-1):
        axs[1].plot(x0s[i:i+2], x1s[i:i+2],
                 alpha=float(i) / (len(x0s)-1),
                 color="red")
    plt.show()

    assert torch.allclose(x, x_min, atol=0.01)

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()
