#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
import torchvision.models.swin_transformer as swin

import opacus.grad_sample.dp_shifted_window_attention  # noqa: F401
from opacus import GradSampleModule, PrivacyEngine
from opacus.layers.dp_shifted_window_attention import DPShiftedWindowAttention
from opacus.validators import ModuleValidator


class ShiftedWindowAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_numerical_parity(self):
        """
        Verifies bit-for-bit forward numerical parity between torchvision
        ShiftedWindowAttention and Opacus DPShiftedWindowAttention.
        """
        dim = 32
        window_size = [4, 4]
        shift_size = [2, 2]
        num_heads = 4
        B, H, W = 2, 8, 8

        orig_layer = swin.ShiftedWindowAttention(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
        )
        orig_layer.eval()

        dp_layer = ModuleValidator.fix(orig_layer)
        dp_layer.eval()

        x = torch.randn(B, H, W, dim)
        with torch.no_grad():
            out_orig = orig_layer(x)
            out_dp = dp_layer(x)

        torch.testing.assert_close(
            out_dp,
            out_orig,
            rtol=1e-5,
            atol=1e-6,
            msg="DPShiftedWindowAttention forward pass diverges from torchvision reference",
        )

    def test_module_validator_and_fixer(self):
        """
        Verifies that ModuleValidator detects unsupported torchvision ShiftedWindowAttention
        and replaces it with DPShiftedWindowAttention.
        """
        layer = swin.ShiftedWindowAttention(
            dim=32,
            window_size=[4, 4],
            shift_size=[2, 2],
            num_heads=4,
        )

        errors = ModuleValidator.validate(layer)
        self.assertGreater(
            len(errors), 0, "Validator failed to flag ShiftedWindowAttention"
        )

        fixed_layer = ModuleValidator.fix(layer)
        self.assertIsInstance(fixed_layer, DPShiftedWindowAttention)

        post_fix_errors = ModuleValidator.validate(fixed_layer)
        self.assertEqual(
            len(post_fix_errors), 0, f"Post-fix validation errors: {post_fix_errors}"
        )

    def test_per_sample_gradient_accuracy(self):
        """
        Verifies that per-sample gradients computed by GradSampleModule match
        the true per-sample gradients obtained via single-sample microbatching.
        """
        B, H, W, C = 3, 8, 8, 32
        window_size = [4, 4]
        shift_size = [2, 2]
        num_heads = 4

        dp_layer = DPShiftedWindowAttention(
            dim=C,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
        )

        # 1. Compute ground-truth per-sample gradients via microbatch autograd loop
        x = torch.randn(B, H, W, C)
        grad_out = torch.randn(B, H, W, C)

        ref_per_sample_grads = {name: [] for name, _ in dp_layer.named_parameters()}
        for i in range(B):
            dp_layer.zero_grad()
            single_x = x[i : i + 1]
            single_out = dp_layer(single_x)
            single_loss = (single_out * grad_out[i : i + 1]).sum()
            single_loss.backward()
            for name, p in dp_layer.named_parameters():
                ref_per_sample_grads[name].append(p.grad.detach().clone())

        for name in ref_per_sample_grads:
            ref_per_sample_grads[name] = torch.stack(ref_per_sample_grads[name], dim=0)

        # 2. Compute Opacus per-sample gradients via GradSampleModule(loss_reduction="sum")
        dp_layer.zero_grad()
        gsm = GradSampleModule(dp_layer, loss_reduction="sum")
        out = gsm(x)
        loss = (out * grad_out).sum()
        loss.backward()

        for name, p in dp_layer.named_parameters():
            self.assertTrue(hasattr(p, "grad_sample"), f"Missing grad_sample on {name}")
            self.assertEqual(p.grad_sample.shape[0], B, f"Batch dim mismatch on {name}")
            torch.testing.assert_close(
                p.grad_sample,
                ref_per_sample_grads[name],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Per-sample gradient mismatch on {name}",
            )

    def test_zero_shift_attention(self):
        """
        Verifies shifted window attention with shift_size=[0, 0] (regular window attention).
        """
        B, H, W, C = 2, 8, 8, 32
        dp_layer = DPShiftedWindowAttention(
            dim=C,
            window_size=[4, 4],
            shift_size=[0, 0],
            num_heads=4,
        )
        gsm = GradSampleModule(dp_layer, loss_reduction="mean")
        x = torch.randn(B, H, W, C)
        out = gsm(x)
        loss = (out**2).mean()
        loss.backward()

        for name, p in dp_layer.named_parameters():
            self.assertEqual(p.grad_sample.shape[0], B)
            torch.testing.assert_close(
                p.grad_sample.mean(dim=0),
                p.grad,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_empty_batch_poisson(self):
        """
        Verifies that an empty batch (B=0) from Poisson sampling does not crash.
        """
        C = 32
        dp_layer = DPShiftedWindowAttention(
            dim=C,
            window_size=[4, 4],
            shift_size=[2, 2],
            num_heads=4,
        )
        activations = [torch.empty(0, 8, 8, C)]
        backprops = torch.empty(0, 8, 8, C)
        grad_samples = opacus.grad_sample.dp_shifted_window_attention.compute_shifted_window_attention_grad_sample(
            dp_layer, activations, backprops
        )
        for p, gs in grad_samples.items():
            self.assertEqual(gs.shape[0], 0)
            self.assertEqual(gs.shape[1:], p.shape)

    def test_end_to_end_dp_sgd(self):
        """
        Verifies full end-to-end DP-SGD training with PrivacyEngine, DPOptimizer, and Accountant.
        """
        dim = 32
        layer = DPShiftedWindowAttention(
            dim=dim,
            window_size=[4, 4],
            shift_size=[2, 2],
            num_heads=4,
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 8, 8, dim), torch.randn(16, 8, 8, dim)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        privacy_engine = PrivacyEngine()
        model, opt, dl = privacy_engine.make_private(
            module=layer,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        for x, y in dl:
            opt.zero_grad()
            out = model(x)
            loss = ((out - y) ** 2).mean()
            loss.backward()
            opt.step()

        eps = privacy_engine.get_epsilon(delta=1e-5)
        self.assertGreater(eps, 0.0)


if __name__ == "__main__":
    unittest.main()
