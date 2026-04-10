import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from opacus import PrivacyEngine


def test_weighted_sampler_privacy_accounting():
    """
    Test that WeightedRandomSampler doesn't break privacy accounting.
    
    Regression test for issue where sample_rate was computed from
    sampler.num_samples instead of the true dataset size, causing
    privacy budget to burn 100x-1000x faster than expected.
    """
    # Dataset with 100,000 samples
    X = torch.randn(100_000, 10)
    y = torch.randint(0, 2, (100_000,))
    dataset = TensorDataset(X, y)
    
    # WeightedRandomSampler with only 128 samples per epoch
    weights = torch.ones(100_000)
    sampler = WeightedRandomSampler(weights, num_samples=128, replacement=True)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=1,
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )
    
    # Verify privacy accounting uses true dataset size
    expected_sample_rate = 16 / 100_000  # 0.00016
    actual_sample_rate = optimizer.expected_batch_size / len(loader.dataset)
    
    assert optimizer.expected_batch_size == 16, \
        f"expected_batch_size should be 16, got {optimizer.expected_batch_size}"
    
    assert abs(actual_sample_rate - expected_sample_rate) < 1e-6, \
        f"sample_rate should be {expected_sample_rate}, got {actual_sample_rate}"