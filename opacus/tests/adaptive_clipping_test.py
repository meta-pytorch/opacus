import unittest

import torch
import torch.nn as nn
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.privacy_engine import PrivacyEngine
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BaseAdaClipTest:
    """Base test class for AdaClipDPOptimizer with different privacy engines."""

    # Subclasses should set this
    ENGINE_CLASS = None

    def setUp(self):
        self.DATA_SIZE = 100
        self.BATCH_SIZE = 10
        self.LR = 0.1

        # Create simple dataset
        self.data = torch.randn(self.DATA_SIZE, 10)
        self.labels = torch.randint(0, 5, (self.DATA_SIZE,))
        self.dataset = TensorDataset(self.data, self.labels)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.BATCH_SIZE, drop_last=False
        )

    def tearDown(self):
        """Clean up controller if needed."""
        if hasattr(self, "controller") and self.controller is not None:
            self.controller.cleanup()

    def _make_private(self, model, optimizer, **kwargs):
        """
        Wrapper to handle both PrivacyEngine types.

        Returns: (model, optimizer, dataloader, controller_or_none)
        """
        privacy_engine = self.ENGINE_CLASS()

        # Check if this is controller-based engine
        is_controller_based = isinstance(
            privacy_engine, PrivacyEngineGradSampleController
        )

        if is_controller_based:
            # Controller-based engine with return_controller=True
            controller, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                return_controller=True,
                **kwargs,
            )
            return model, optimizer, dataloader, controller
        else:
            # Standard engine
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model, optimizer=optimizer, data_loader=self.dataloader, **kwargs
            )
            return model, optimizer, dataloader, None

    def test_adaclip_optimizer_initialization(self):
        """Test that AdaClipDPOptimizer can be initialized."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        # Make private with AdaClip optimizer
        # Note: noise_multiplier must be < 2 * unclipped_num_std (AdaClip constraint)
        unclipped_num_std = 1.0
        model, optimizer, dataloader, self.controller = self._make_private(
            model=model,
            optimizer=optimizer,
            noise_multiplier=0.5,  # < 2 * unclipped_num_std (1.0)
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        # Verify optimizer is AdaClipDPOptimizer
        self.assertIsInstance(optimizer, AdaClipDPOptimizer)

        # Verify AdaClip-specific attributes exist
        self.assertTrue(hasattr(optimizer, "target_unclipped_quantile"))
        self.assertTrue(hasattr(optimizer, "clipbound_learning_rate"))
        self.assertTrue(hasattr(optimizer, "max_clipbound"))
        self.assertTrue(hasattr(optimizer, "min_clipbound"))
        self.assertTrue(hasattr(optimizer, "unclipped_num"))
        self.assertTrue(hasattr(optimizer, "sample_size"))

    def test_adaclip_clipbound_updates(self):
        """Test that adaptive clipping actually updates the clipping bound."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        model, optimizer, dataloader, self.controller = self._make_private(
            model=model,
            optimizer=optimizer,
            noise_multiplier=0.0,  # No noise for clearer results
            max_grad_norm=1.0,  # Initial clip bound
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.05,
        )

        criterion = nn.CrossEntropyLoss()
        initial_clipbound = optimizer.max_grad_norm
        clipbounds = [initial_clipbound]

        # Train for several steps and track clipbound changes
        for i, (x, y) in enumerate(dataloader):
            if i >= 5:
                break

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Record clipbound after step
            clipbounds.append(optimizer.max_grad_norm)

        # Verify that clipbound changed during training
        unique_clipbounds = set(f"{cb:.6f}" for cb in clipbounds)
        self.assertGreater(
            len(unique_clipbounds),
            1,
            f"Clipbound should change over time. Got values: {clipbounds}",
        )

        # Verify clipbound stays within bounds
        for cb in clipbounds:
            self.assertGreaterEqual(cb, 0.01)  # min_clipbound
            self.assertLessEqual(cb, 10.0)  # max_clipbound

    def test_adaclip_unclipped_tracking(self):
        """Test that AdaClip correctly tracks unclipped gradient counts."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        unclipped_num_std = 1.0
        model, optimizer, dataloader, self.controller = self._make_private(
            model=model,
            optimizer=optimizer,
            noise_multiplier=0.8,  # < 2 * unclipped_num_std (0.8 < 1.0)
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        criterion = nn.CrossEntropyLoss()

        # Train one step
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Call optimizer.step() which triggers clip_and_accumulate
        # that sets sample_size and unclipped_num
        optimizer.step()

        # After step, clipbound should have been updated
        self.assertIsNotNone(optimizer.max_grad_norm)

        # Verify sample_size is positive
        sample_size = (
            float(optimizer.sample_size)
            if torch.is_tensor(optimizer.sample_size)
            else optimizer.sample_size
        )
        self.assertGreater(
            sample_size, 0, "Sample size should be positive after training step"
        )

        # Compute unclipped fraction (convert to float if tensor)
        # Note: unclipped_num can be negative due to DP noise in AdaClip
        unclipped_num = (
            float(optimizer.unclipped_num)
            if torch.is_tensor(optimizer.unclipped_num)
            else optimizer.unclipped_num
        )
        unclipped_frac = unclipped_num / sample_size
        # Due to DP noise, unclipped_frac may be slightly outside [0, 1]
        # Just verify it's been set and is a reasonable value
        self.assertGreater(
            unclipped_frac, -0.5, "Unclipped fraction should not be too negative"
        )
        self.assertLess(
            unclipped_frac, 1.5, "Unclipped fraction should not be too large"
        )

        # Do another step to verify counters work across multiple steps
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # After another step, counters should still be valid
        sample_size2 = (
            float(optimizer.sample_size)
            if torch.is_tensor(optimizer.sample_size)
            else optimizer.sample_size
        )
        self.assertGreater(sample_size2, 0)

    def test_adaclip_convergence_behavior(self):
        """Test that AdaClip converges toward target quantile."""
        torch.manual_seed(42)
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        target_quantile = 0.7
        unclipped_num_std = 0.5
        model, optimizer, dataloader, self.controller = self._make_private(
            model=model,
            optimizer=optimizer,
            noise_multiplier=0.8,  # < 2 * unclipped_num_std (0.8 < 1.0)
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=target_quantile,
            clipbound_learning_rate=0.1,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        criterion = nn.CrossEntropyLoss()
        unclipped_fractions = []

        # Train for multiple steps
        for i, (x, y) in enumerate(dataloader):
            if i >= 10:
                break

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # Record unclipped fraction before step
            if optimizer.sample_size > 0:
                unclipped_frac = float(optimizer.unclipped_num) / optimizer.sample_size
                unclipped_fractions.append(unclipped_frac)

            optimizer.step()

        # Average unclipped fraction should be reasonably close to target
        # (not exact due to noise and limited steps)
        if len(unclipped_fractions) > 5:
            avg_unclipped = sum(unclipped_fractions[-5:]) / 5
            # Should be within reasonable range of target
            self.assertGreater(avg_unclipped, target_quantile - 0.3)
            self.assertLess(avg_unclipped, target_quantile + 0.3)

    def test_adaclip_vs_fixed_clipping(self):
        """Test that AdaClip behaves differently from fixed clipping."""
        torch.manual_seed(42)

        # Train with AdaClip
        model1 = SimpleNet()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=self.LR)

        model1, optimizer1, dataloader1, controller1 = self._make_private(
            model=model1,
            optimizer=optimizer1,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.05,
        )

        # Train with fixed clipping
        torch.manual_seed(42)
        model2 = SimpleNet()
        # Handle both wrapped (GradSampleModule) and unwrapped models
        state_dict1 = model1.state_dict()
        # If wrapped, state_dict has _module. prefix, need to remove it
        if any(key.startswith("_module.") for key in state_dict1.keys()):
            state_dict1 = {k.replace("_module.", ""): v for k, v in state_dict1.items()}
        model2.load_state_dict(state_dict1)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=self.LR)

        model2, optimizer2, dataloader2, controller2 = self._make_private(
            model=model2,
            optimizer=optimizer2,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",  # Fixed clipping
        )

        criterion = nn.CrossEntropyLoss()

        # Train both for several steps
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader1, dataloader2)):
            if i >= 5:
                break

            # AdaClip training
            optimizer1.zero_grad()
            output1 = model1(x1)
            loss1 = criterion(output1, y1)
            loss1.backward()
            optimizer1.step()

            # Fixed clipping training
            optimizer2.zero_grad()
            output2 = model2(x2)
            loss2 = criterion(output2, y2)
            loss2.backward()
            optimizer2.step()

        # After training, parameters should differ
        # (because AdaClip adjusts clipbound while fixed doesn't)
        params_differ = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2, atol=1e-5):
                params_differ = True
                break

        self.assertTrue(
            params_differ, "AdaClip and fixed clipping should produce different results"
        )

        # Cleanup both controllers if they exist
        if controller1:
            controller1.cleanup()
        if controller2:
            controller2.cleanup()
        # Mark as cleaned up so tearDown doesn't try again
        self.controller = None

    def test_adaclip_parameter_validation(self):
        """Test that AdaClip validates parameters correctly."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        # Test: max_clipbound <= min_clipbound should raise error
        with self.assertRaises(ValueError):
            self._make_private(
                model=model,
                optimizer=optimizer,
                noise_multiplier=0.05,  # < 2 * unclipped_num_std
                max_grad_norm=1.0,
                clipping="adaptive",
                target_unclipped_quantile=0.5,  # Required param
                clipbound_learning_rate=0.2,  # Required param
                max_clipbound=0.01,  # Less than min - should trigger error
                min_clipbound=0.1,
                unclipped_num_std=0.05,
            )

    def test_adaclip_with_nonzero_noise(self):
        """Test AdaClip works with noise (full DP training)."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        unclipped_num_std = 0.5
        model, optimizer, dataloader, self.controller = self._make_private(
            model=model,
            optimizer=optimizer,
            noise_multiplier=0.8,  # With noise, < 2 * unclipped_num_std
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        criterion = nn.CrossEntropyLoss()

        # Train one step with noise
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Verify training completed successfully
        for param in model.parameters():
            self.assertIsNotNone(param.grad)


class AdaClipStandardEngineTest(BaseAdaClipTest, unittest.TestCase):
    """Test AdaClipDPOptimizer with standard PrivacyEngine."""

    ENGINE_CLASS = PrivacyEngine


class AdaClipGradSampleControllerEngineTest(BaseAdaClipTest, unittest.TestCase):
    """Test AdaClipDPOptimizer with GradSampleController-based PrivacyEngine."""

    ENGINE_CLASS = PrivacyEngineGradSampleController


if __name__ == "__main__":
    unittest.main()
