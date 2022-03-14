#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch.optim.optimizer import Optimizer

__all__ = ('CurveSGD',)


class CurveSGD(Optimizer):
    r"""Implements Self-Tuning Stochastic Optimization with
    Curvature-Aware Gradient Filtering algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.CurveSGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
     __ https://arxiv.org/pdf/2011.04803.pdf
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

    def get_hessian_prod(self, params, grads, delta):
        """Get an estimate of Hessian product.
        This is done by computing the Hessian vector product with the stored delta
        vector at the current gradient point, to estimate Hessian trace by
        computing the gradient of <gradsH, s>.
        """

        # Check backward was called with create_graph set to True
        if grads.grad_fn is None:
            msg = (
                'Gradient tensor {:} does not have grad_fn. When '
                'calling loss.backward(), make sure the option '
                'create_graph is set to True.'
            )
            raise RuntimeError(msg.format(i))

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        hvs = torch.autograd.grad(
            grads, params, grad_outputs=delta, only_inputs=True, retain_graph=True
        )

        return hvs[0]

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

                if d_p.is_sparse:
                    msg = (
                        'CurveSGD does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['t'] = 0

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

                    # Exponential moving average of gradient values
                    state['hess_exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state['hess_exp_var'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    state['delta_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Kalman Filter states
                    state['m_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state['P_t'] = torch.eye(p.grad.size()[0]).mul(1e4)
                    state['u_t'] = 0
                    state['s_t'] = 1e4

                state['t'] += 1
                
                func_exp_avg = state['func_exp_avg']
                func_exp_var = state['func_exp_var']
                grad_exp_avg = state['grad_exp_avg']
                grad_exp_var = state['grad_exp_var']
                hess_exp_avg = state['hess_exp_avg']
                hess_exp_var = state['hess_exp_var']
                delta_t = state['delta_t']

                h_delta = self.get_hessian_prod(p, p.grad, delta_t)
                beta_delta = 1 - 1 / state['t'] # non-smoothed running average/variance

                # Decay the first and second moment running average coefficient
                func_exp_avg.mul_(beta_r).add_(loss, alpha=1 - beta_r)
                func_exp_var.mul_(beta_r).addcmul_(loss, loss, value=1 - beta_r)

                grad_exp_avg.mul_(beta_sigma).add_(p.grad, alpha=1 - beta_sigma)
                grad_exp_var.mul_(beta_sigma).addcmul_(p.grad, p.grad, value=1 - beta_sigma)

                hess_exp_avg.mul_(beta_delta).add_(h_delta, alpha=1 - beta_delta)
                hess_exp_var.mul_(beta_delta).addcmul_(h_delta, h_delta, value=1 - beta_delta)

                sigma_t = torch.mean(grad_exp_var)
                q_t = torch.mean(hess_exp_var)

                # Match notation from paper for convenience
                y_t = func_exp_avg
                r_t = func_exp_var
                g_t = grad_exp_avg
                Sigma_t = torch.eye(p.grad.size()[0]).mul(sigma_t)
                b_t = hess_exp_avg
                Q_t = torch.eye(p.grad.size()[0]).mul(q_t)

                # Kalman Filter update
                m_t = state['m_t']
                P_t = state['P_t']
                u_t = state['u_t']
                s_t = state['s_t']

                m_t_minus = m_t + h_delta
                P_t_minus = P_t + Q_t 
                K_t = P_t_minus.matmul((P_t_minus + Sigma_t).inverse())

                m_t = (torch.eye(p.grad.size()[0]) - K_t).matmul(m_t_minus) + K_t.matmul(g_t)
                P_t = (torch.eye(p.grad.size()[0]) - K_t).matmul(P_t_minus).matmul((torch.eye(p.grad.size()[0]) - K_t).t()) \
                        + K_t.matmul(Sigma_t).matmul(K_t.t())

                delta_t = m_t.mul(group['lr'])

                state['m_t'] = m_t
                state['P_t'] = P_t
                state['delta_t'] = delta_t

                # Use filtered gradient estimate for update step
                p.data.sub_(delta_t)

        return loss