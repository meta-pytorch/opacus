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
from opacus.data_loader import CollateFnWithEmpty, DPDataLoader, wrap_collate_with_empty
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate


class DPDataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_size = 10
        self.dimension = 7
        self.num_classes = 11

    def test_collate_classes(self) -> None:
        """Test that empty batches are handled correctly with classification data"""
        x = torch.randn(self.data_size, self.dimension)
        y = torch.randint(low=0, high=self.num_classes, size=(self.data_size,))

        dataset = TensorDataset(x, y)
        # Use seeded generator with low sample rate to produce empty batches deterministically
        # seed=0, sample_rate=0.1 produces non-empty first batch followed by empty batches
        generator = torch.Generator().manual_seed(0)
        data_loader = DPDataLoader(dataset, sample_rate=0.1, generator=generator)

        # Process batches - verify structure is preserved
        first_batch = next(iter(data_loader))
        x_b, y_b = first_batch

        # First batch must be non-empty (to learn structure)
        self.assertGreater(x_b.size(0), 0, "First batch must be non-empty")
        self.assertEqual(len(x_b.shape), 2)
        self.assertEqual(x_b.shape[1], self.dimension)

        # Process all batches and verify at least one is empty
        batch_count = 1
        empty_batch_found = False
        for batch in data_loader:
            x_b, y_b = batch
            batch_size = x_b.size(0)

            # Batch dimension should be 0 or positive
            self.assertGreaterEqual(batch_size, 0)
            self.assertGreaterEqual(y_b.size(0), 0)

            if batch_size == 0:
                empty_batch_found = True
                # Empty batch should still have correct feature dimension
                self.assertEqual(x_b.shape[1], self.dimension)
            else:
                # Non-empty batch should have correct dimensions
                self.assertEqual(x_b.shape[1], self.dimension)
            batch_count += 1

        # Verify we actually tested empty batch handling
        self.assertTrue(empty_batch_found, "No empty batches produced - test doesn't verify empty batch handling")
        self.assertGreater(batch_count, 1)

    def test_collate_tensor(self) -> None:
        """Test that empty batches are handled correctly with single tensor data"""
        x = torch.randn(self.data_size, self.dimension)

        dataset = TensorDataset(x)
        # Use seeded generator with low sample rate to produce empty batches deterministically
        # seed=0, sample_rate=0.1 produces non-empty first batch followed by empty batches
        generator = torch.Generator().manual_seed(0)
        data_loader = DPDataLoader(dataset, sample_rate=0.1, generator=generator)
        first_batch = next(iter(data_loader))
        (s,) = first_batch

        # First batch must be non-empty (to learn structure)
        self.assertGreater(s.size(0), 0, "First batch must be non-empty")
        self.assertEqual(s.shape[1], self.dimension)

        # Process all batches and verify at least one is empty
        batch_count = 1
        empty_batch_found = False
        for batch in data_loader:
            (s,) = batch
            batch_size = s.size(0)

            self.assertGreaterEqual(batch_size, 0)

            if batch_size == 0:
                empty_batch_found = True
                # Empty batch should still have correct feature dimension
                self.assertEqual(s.shape[1], self.dimension)
            else:
                # Non-empty batch should have correct dimensions
                self.assertEqual(s.shape[1], self.dimension)
            batch_count += 1

        # Verify we actually tested empty batch handling
        self.assertTrue(empty_batch_found, "No empty batches produced - test doesn't verify empty batch handling")
        self.assertGreater(batch_count, 1)

    def test_drop_last_true(self) -> None:
        x = torch.randn(self.data_size, self.dimension)

        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=True)
        _ = DPDataLoader.from_data_loader(data_loader)


class CollateFnWithEmptyTest(unittest.TestCase):
    """Tests for the CollateFnWithEmpty class"""

    def test_simple_tensor_non_empty(self) -> None:
        """Test that non-empty batches are handled correctly with simple tensors"""
        collate_fn = CollateFnWithEmpty(default_collate)
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = collate_fn(batch)

        self.assertTrue(torch.is_tensor(result))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1, 2], [3, 4]])))

    def test_simple_tensor_empty_batch(self) -> None:
        """Test that empty batches generate correct empty tensors"""
        collate_fn = CollateFnWithEmpty(default_collate)

        # First process a non-empty batch to learn structure
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        _ = collate_fn(batch)

        # Now process empty batch
        empty_result = collate_fn([])

        self.assertTrue(torch.is_tensor(empty_result))
        self.assertEqual(empty_result.shape[0], 0)  # Batch dimension should be 0
        self.assertEqual(empty_result.shape[1], 2)  # Other dimensions preserved

    def test_empty_batch_before_first_raises_error(self) -> None:
        """Test that processing empty batch first raises ValueError"""
        collate_fn = CollateFnWithEmpty(default_collate)

        with self.assertRaises(ValueError) as context:
            collate_fn([])

        self.assertIn("First sampled batch cannot be empty", str(context.exception))

    def test_dict_structure_preserved(self) -> None:
        """Test that dictionary structures are preserved in empty batches"""
        collate_fn = CollateFnWithEmpty(default_collate)

        # First batch with dict structure
        batch = [
            {"x": torch.tensor([1, 2]), "y": torch.tensor([5])},
            {"x": torch.tensor([3, 4]), "y": torch.tensor([6])}
        ]
        result = collate_fn(batch)

        self.assertIsInstance(result, dict)
        self.assertIn("x", result)
        self.assertIn("y", result)

        # Empty batch should preserve dict structure
        empty_result = collate_fn([])

        self.assertIsInstance(empty_result, dict)
        self.assertIn("x", empty_result)
        self.assertIn("y", empty_result)
        self.assertEqual(empty_result["x"].shape[0], 0)
        self.assertEqual(empty_result["y"].shape[0], 0)

    def test_nested_list_structure(self) -> None:
        """Test that nested list structures are preserved"""
        collate_fn = CollateFnWithEmpty(default_collate)

        # First batch with list of tensors
        batch = [
            [torch.tensor([1, 2]), torch.tensor([3])],
            [torch.tensor([4, 5]), torch.tensor([6])]
        ]
        result = collate_fn(batch)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        # Empty batch should preserve list structure
        empty_result = collate_fn([])

        self.assertIsInstance(empty_result, list)
        self.assertEqual(len(empty_result), 2)
        self.assertEqual(empty_result[0].shape[0], 0)
        self.assertEqual(empty_result[1].shape[0], 0)

    def test_rand_on_empty_true(self) -> None:
        """Test rand_on_empty=True generates random tensors with batch_size=1"""
        collate_fn = CollateFnWithEmpty(default_collate, rand_on_empty=True)

        # First process non-empty batch
        batch = [torch.tensor([1, 2, 3])]
        _ = collate_fn(batch)

        # Empty batch should have batch_size=1 with random values
        empty_result = collate_fn([])

        self.assertTrue(torch.is_tensor(empty_result))
        self.assertEqual(empty_result.shape[0], 1)  # Batch dimension should be 1
        self.assertEqual(empty_result.shape[1], 3)  # Other dimensions preserved
        # Values should be 0 or 1 (from torch.randint(0, 2, ...))
        self.assertTrue(torch.all((empty_result == 0) | (empty_result == 1)))

    def test_batch_first_false(self) -> None:
        """Test batch_first=False puts batch dimension at index 1"""
        collate_fn = CollateFnWithEmpty(default_collate, batch_first=False)

        # First process non-empty batch - shape will be [batch, features]
        batch = [torch.tensor([1, 2, 3])]
        result = collate_fn(batch)

        # For empty batch with batch_first=False, batch dim should be at index 1
        empty_result = collate_fn([])

        self.assertTrue(torch.is_tensor(empty_result))
        # With batch_first=False, shape should be [features, 0]
        self.assertEqual(empty_result.shape[1], 0)

    def test_no_collator_fn(self) -> None:
        """Test with collator_fn=None returns batch as-is"""
        collate_fn = CollateFnWithEmpty(None)

        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = collate_fn(batch)

        # Without collator, should return list as-is
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_wrap_collate_with_empty_function(self) -> None:
        """Test the wrap_collate_with_empty factory function"""
        collate_fn = wrap_collate_with_empty(collate_fn=default_collate)

        self.assertIsInstance(collate_fn, CollateFnWithEmpty)

        # Test it works correctly
        batch = [torch.tensor([1, 2])]
        result = collate_fn(batch)
        self.assertTrue(torch.is_tensor(result))

    def test_multiple_empty_batches(self) -> None:
        """Test that multiple empty batches can be processed"""
        collate_fn = CollateFnWithEmpty(default_collate)

        # First non-empty batch
        batch = [torch.tensor([1, 2, 3])]
        _ = collate_fn(batch)

        # Multiple empty batches should work
        for _ in range(3):
            empty_result = collate_fn([])
            self.assertTrue(torch.is_tensor(empty_result))
            self.assertEqual(empty_result.shape[0], 0)

    def test_tuple_preservation(self) -> None:
        """Test that tuple structures are preserved"""
        def tuple_collate(batch):
            # Custom collator that returns tuples
            x = default_collate([item[0] for item in batch])
            y = default_collate([item[1] for item in batch])
            return (x, y)

        collate_fn = CollateFnWithEmpty(tuple_collate)

        batch = [(torch.tensor([1, 2]), torch.tensor([5])),
                 (torch.tensor([3, 4]), torch.tensor([6]))]
        result = collate_fn(batch)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Empty batch should preserve tuple
        empty_result = collate_fn([])

        self.assertIsInstance(empty_result, tuple)
        self.assertEqual(len(empty_result), 2)
        self.assertEqual(empty_result[0].shape[0], 0)
        self.assertEqual(empty_result[1].shape[0], 0)

    def test_unsupported_type_raises_error(self) -> None:
        """Test that unsupported batch types raise TypeError to preserve DP guarantees"""
        def custom_collate(batch):
            # Custom collator that returns an unsupported type (e.g., string)
            if len(batch) > 0:
                return "unsupported_type"
            return ""

        collate_fn = CollateFnWithEmpty(custom_collate)

        # First process non-empty batch
        batch = [torch.tensor([1, 2])]
        _ = collate_fn(batch)

        # Empty batch should raise TypeError for unsupported type
        with self.assertRaises(TypeError) as context:
            collate_fn([])

        self.assertIn("Unsupported batch type", str(context.exception))
