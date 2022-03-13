#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torch.optim.optimizer import Optimizer

__all__ = ('CurveSGD',)


class CurveSGD(Optimizer):
    r"""Implements Self-Tuning Stochastic Optimization with
    Curvature-Aware Gradient Filtering algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        kappa: ratio of long to short step (default: 1000)
        xi: statistical advantage parameter (default: 10)
        small_const: any value <=1 (default: 0.7)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.CurveSGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
     __ https://arxiv.org/abs/1704.08227
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta_r=0.999,
        beta_sigma=0.999,
        beta_alpha=0.999,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        defaults = dict(
            lr=lr,
            beta_r=beta_r,
            beta_sigma=beta_sigma,
            beta_alpha=beta_alpha,
        )
        super(CurveSGD, self).__init__(params, defaults)

    def step(self, closure = None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta_r = group['beta_r']
            beta_sigma = group['beta_sigma']
            beta_alpha = group['beta_alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if grad.is_sparse:
                    msg = (
                        'CurveSGD does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Exponential moving average of function values
                    state['func_exp_avg'] = torch.zeros_like(
                        loss, memory_format=torch.preserve_format
                    )
                    state['func_exp_var'] = torch.zeros_like(
                        loss, memory_format=torch.preserve_format
                    )

                    # Exponential moving average of gradient values
                    state['grad_exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state['grad_exp_var'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                func_exp_avg, func_exp_var, grad_exp_avg, grad_exp_var = (
                    state['func_exp_avg'],
                    state['func_exp_var'],
                    state['grad_exp_avg'],
                    state['grad_exp_var'],
                )

                # Decay the first and second moment running average coefficient
                func_exp_avg.mul_(beta_r).add_(loss, alpha=1 - beta_r)
                func_exp_var.mul_(beta_r).addcmul_(loss, loss, value=1 - beta_r)

                grad_exp_avg.mul_(beta_sigma).add_(grad, alpha=1 - beta_sigma)
                grad_exp_var.mul_(beta_sigma).addcmul_(grad, grad, value=1 - beta_sigma)

                state['step'] += 1
                
                p.data.add_(d_p, alpha=-group['lr'])

        return loss