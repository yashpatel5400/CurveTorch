#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch.optim.optimizer import Optimizer

import numpy as np
from scipy.optimize import minimize

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
        Arguments:
            params: iterable of parameters to optimize or dicts defining
                parameter groups
            grads: gradient of parameters
            delta: vector to be multiplied against the Hessian (right multiplied)
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

    def _get_prob_improve_num_den(self, alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t):
        alpha = alpha[0]
        numerator = -alpha * delta_t.matmul(m_t) + alpha ** 2 / 2 * delta_t.t().matmul(B_delta)
        denominator = 2 * s_t + alpha ** 2 * delta_t.t().matmul(P_t).matmul(delta_t) \
            + alpha ** 4 / 4 * delta_t.t().matmul(Q_t).matmul(delta_t)
        numerator = numerator.detach().numpy()
        denominator = np.sqrt(denominator.detach().numpy())[0]
        return numerator, denominator

    def prob_improve(self, alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t):
        """Get an estimate of improvement probability assuming alpha step size.
        This is done as a subroutine procedure to determine the optimal
        step size within after running filtering on the function and gradient values.
        Intended to be used in conjunction with an optimization procedure (i.e scipy.optimize)
        assuming all parameters fixed except alpha.
        Arguments:
            alpha: value of step size
            delta_t: Gradient change
            m_t: Kalman filtered gradient mean
            B_delta: Hessian-vector product
            s_t: Kalman filtered function mean
            P_t: Kalman filtered gradient covariance
            Q_t: Covariance of Hessian-vector product
        """
        numerator, denominator = self._get_prob_improve_num_den(alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t)
        return numerator / denominator

    def prob_improve_grad(self, alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t):
        """Get an estimate of improvement probability gradient. See prob_improve for docs
        Arguments:
            alpha: value of step size
            delta_t: Gradient change
            m_t: Kalman filtered gradient mean
            B_delta: Hessian-vector product
            s_t: Kalman filtered function mean
            P_t: Kalman filtered gradient covariance
            Q_t: Covariance of Hessian-vector product
        """
        numerator, denominator = self._get_prob_improve_num_den(alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t)
        alpha = alpha[0]
        numerator_grad = delta_t.matmul(m_t) + alpha * delta_t.t().matmul(B_delta)
        denominator_grad = 1 / (2 * denominator) * (2 * alpha * delta_t.t().matmul(P_t).matmul(delta_t) \
            + alpha ** 3 * delta_t.t().matmul(Q_t).matmul(delta_t))
        numerator_grad = numerator_grad.detach().numpy()
        denominator_grad = denominator_grad.detach().numpy()

        return (denominator * numerator_grad - numerator * denominator_grad) / denominator ** 2

    def mean_var_ewa(self, ema, emvar, x, beta):
        alpha = 1 - beta
        delta = x - ema
        ema_new = ema.add(delta.mul(alpha))
        emvar_new = emvar.add(delta.mul(delta).mul(alpha)).mul(1 - alpha)
        return ema_new, emvar_new

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

                    state['delta_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Exponential moving average of function values
                    state['func_exp_avg'] = loss.clone()
                    state['func_exp_var'] = torch.zeros((1))

                    # Exponential moving average of gradient values
                    state['grad_exp_avg'] = p.grad.clone()
                    state['grad_exp_var'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Exponential moving average of Hessian values
                    state['hess_exp_avg'] = self.get_hessian_prod(p, p.grad, state['delta_t']).clone()
                    state['hess_exp_var'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Kalman Filter states
                    state['m_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state['P_t'] = torch.eye(p.grad.size()[0]).mul(1e4)
                    state['u_t'] = 0
                    state['s_t'] = 1e4

                func_exp_avg = state['func_exp_avg']
                func_exp_var = state['func_exp_var']
                grad_exp_avg = state['grad_exp_avg']
                grad_exp_var = state['grad_exp_var']
                hess_exp_avg = state['hess_exp_avg']
                hess_exp_var = state['hess_exp_var']
                delta_t = state['delta_t']

                if state['t'] != 0:
                    beta_delta = 1 - 1 / state['t'] # non-smoothed running average/variance

                    func_exp_avg, func_exp_var = self.mean_var_ewa(func_exp_avg, func_exp_var, loss, beta_r)
                    grad_exp_avg, grad_exp_var = self.mean_var_ewa(grad_exp_avg, grad_exp_var, loss, beta_sigma)
                    hess_exp_avg, hess_exp_var = self.mean_var_ewa(hess_exp_avg, hess_exp_var, loss, beta_delta)

                B_delta = self.get_hessian_prod(p, p.grad, delta_t)
                
                sigma_t = torch.mean(grad_exp_var)
                q_t = torch.mean(hess_exp_var)

                # Match notation from paper for convenience
                y_t = func_exp_avg
                r_t = func_exp_var
                g_t = grad_exp_avg
                Sigma_t = torch.eye(p.grad.size()[0]).mul(sigma_t)
                b_t = hess_exp_avg
                Q_t = torch.eye(p.grad.size()[0]).mul(q_t)

                # Kalman Filter update for f
                u_t = state['u_t']
                s_t = state['s_t']
                m_t = state['m_t']
                P_t = state['P_t']

                # steps for Kalman filter
                # compute u_t_minus
                u_t_minus = u_t + m_t.t().matmul(delta_t) + 1 / 2 * delta_t.t().matmul(B_delta)
                c_t = s_t + delta_t.t().matmul(P_t).matmul(delta_t) + 1 / 4 * delta_t.t().matmul(Q_t).matmul(delta_t) + r_t
                lambda_t = max((y_t - u_t_minus) ** 2 - c_t, 0)
                s_t_minus = lambda_t + c_t - r_t

                mix_t = s_t_minus / (s_t_minus + r_t)
                u_t = (1 - mix_t) * u_t_minus + mix_t * y_t
                s_t = (1 - mix_t) ** 2 * s_t_minus + mix_t ** 2 * r_t

                # Kalman Filter update for grad f
                m_t_minus = m_t + B_delta
                P_t_minus = P_t + Q_t 
                K_t = P_t_minus.matmul((P_t_minus + Sigma_t).inverse())

                m_t = (torch.eye(p.grad.size()[0]) - K_t).matmul(m_t_minus) + K_t.matmul(g_t)
                P_t = (torch.eye(p.grad.size()[0]) - K_t).matmul(P_t_minus).matmul((torch.eye(p.grad.size()[0]) - K_t).t()) \
                        + K_t.matmul(Sigma_t).matmul(K_t.t())

                prob_improve_closure = lambda alpha : self.prob_improve(alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t)
                prob_improve_grad_closure = lambda alpha : self.prob_improve_grad(alpha, delta_t, m_t, B_delta, s_t, P_t, Q_t)
                if state['t'] == 0:
                    lr = group['lr']
                else:
                    lr = minimize(prob_improve_closure, group['lr'], jac=prob_improve_grad_closure, method='BFGS')
                    print(lr)

                delta_t = m_t.mul(lr)

                state['t'] += 1
                
                state['u_t'] = u_t
                state['s_t'] = s_t
                state['m_t'] = m_t
                state['P_t'] = P_t

                state['func_exp_avg'] = func_exp_avg
                state['func_exp_var'] = func_exp_var
                state['grad_exp_avg'] = grad_exp_avg
                state['grad_exp_var'] = grad_exp_var
                state['hess_exp_avg'] = hess_exp_avg
                state['hess_exp_var'] = hess_exp_var
                state['delta_t']      = delta_t

                # Use filtered gradient estimate for update step
                p.data.sub_(delta_t)

        return loss