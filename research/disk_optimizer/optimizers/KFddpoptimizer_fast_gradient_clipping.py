from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

from .KFoptimizer_fast_gradient_clipping import KF_DPOptimizerFastGradientClipping


logger = logging.getLogger(__name__)
logger.disabled = True


class KF_DistributedDPOptimizerFastGradientClipping(KF_DPOptimizerFastGradientClipping):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` compatible with
    distributed data processing
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        kappa=0.7,
        gamma=0.5,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            kappa=kappa,
            gamma=gamma,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        # Noise only gets added to the first worker
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(self, closure=required) -> Optional[float]:
        if self.kf_compute_grad_at_original:
            loss = self._compute_two_closure(closure)
        else:
            loss = self._compute_one_closure(closure)

        if self.pre_step():
            tmp_states = []
            first_step = False
            for p in self.params:
                grad = p.grad
                state = self.state[p]
                if "kf_d_t" not in state:
                    state = dict()
                    first_step = True
                    state["kf_d_t"] = torch.zeros_like(p.data).to(p.data)
                    state["kf_m_t"] = grad.clone().to(p.data)
                state["kf_m_t"].lerp_(grad, weight=self.kappa)
                p.grad = state["kf_m_t"].clone().to(p.data)
                state["kf_d_t"] = -p.data.clone().to(p.data)
                if first_step:
                    tmp_states.append(state)
            self.reduce_gradients()
            self.original_optimizer.step()
            for p in self.params:
                if first_step:
                    tmp_state = tmp_states.pop(0)
                    self.state[p]["kf_d_t"] = tmp_state["kf_d_t"]
                    self.state[p]["kf_m_t"] = tmp_state["kf_m_t"]
                    del tmp_state
                self.state[p]["kf_d_t"].add_(p.data, alpha=1.0)
        return loss
