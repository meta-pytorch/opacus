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

import math
from typing import List, Sequence, Union

import numpy as np
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, Sampler


def _to_optimizer_list(
    optimizer: Union[DPOptimizer, Optimizer, Sequence[Union[DPOptimizer, Optimizer]]],
) -> List[Optimizer]:
    """
    Normalizes the optimizer argument into a flat list of optimizers.

    Accepts a single optimizer (DP or standard) or an iterable of optimizers
    and validates that every element is a ``torch.optim.Optimizer``.

    Args:
        optimizer: single optimizer instance or a sequence of optimizers

    Returns:
        List of optimizer instances
    """
    if isinstance(optimizer, Optimizer):
        return [optimizer]

    try:
        optimizers = list(optimizer)
    except TypeError:
        raise TypeError(
            f"optimizer must be an optimizer instance or a sequence of "
            f"optimizers, got: {type(optimizer)}"
        )

    for opt in optimizers:
        if not isinstance(opt, Optimizer):
            raise TypeError(
                f"All elements must be optimizer instances, got: {type(opt)}"
            )

    return optimizers


class BatchSplittingSampler(Sampler[List[int]]):
    """
    Samples according to the underlying instance of ``Sampler``, but splits
    the index sequences into smaller chunks.

    Used to split large logical batches into physical batches of a smaller size,
    while coordinating with DPOptimizer when the logical batch has ended.
    """

    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        optimizer: Union[
            DPOptimizer, Optimizer, Sequence[Union[DPOptimizer, Optimizer]]
        ],
    ):
        """

        Args:
            sampler: Wrapped Sampler instance
            max_batch_size: Max size of emitted chunk of indices
            optimizer: optimizer instance (or a sequence of optimizer instances,
                e.g. for private GANs or actor-critic models) to notify when the
                logical batch is over. Optimizers without DP support (i.e.
                without ``signal_skip_step``) are silently ignored.
        """
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.optimizers: List[Optimizer] = _to_optimizer_list(optimizer)

    def _signal_skip_step(self, do_skip: bool):
        """
        Notifies all optimizers that support it whether the next step should be
        skipped (i.e. only perform clipping and per sample gradient accumulation).

        Standard PyTorch optimizers don't have ``signal_skip_step`` and are
        skipped, so they can safely be mixed with DPOptimizers.
        """
        for optimizer in self.optimizers:
            if hasattr(optimizer, "signal_skip_step") and callable(
                optimizer.signal_skip_step
            ):
                optimizer.signal_skip_step(do_skip=do_skip)

    def __iter__(self):
        for batch_idxs in self.sampler:
            if len(batch_idxs) == 0:
                self._signal_skip_step(do_skip=False)
                yield []
                continue

            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            split_idxs = [s.tolist() for s in split_idxs]
            for x in split_idxs[:-1]:
                self._signal_skip_step(do_skip=True)
                yield x
            self._signal_skip_step(do_skip=False)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, BatchSampler):
            return math.ceil(
                len(self.sampler) * (self.sampler.batch_size / self.max_batch_size)
            )
        elif isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return math.ceil(
                len(self.sampler) * (expected_batch_size / self.max_batch_size)
            )

        return len(self.sampler)


def wrap_data_loader(
    *,
    data_loader: DataLoader,
    max_batch_size: int,
    optimizer: Union[DPOptimizer, Optimizer, Sequence[Union[DPOptimizer, Optimizer]]],
):
    """
    Replaces batch_sampler in the input data loader with ``BatchSplittingSampler``

    Args:
        data_loader: Wrapper DataLoader
        max_batch_size: max physical batch size we want to emit
        optimizer: DPOptimizer instance (or a sequence of optimizers) used for training

    Returns:
        New DataLoader instance with batch_sampler wrapped in ``BatchSplittingSampler``
    """

    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=BatchSplittingSampler(
            sampler=data_loader.batch_sampler,
            max_batch_size=max_batch_size,
            optimizer=optimizer,
        ),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )


class BatchMemoryManager:
    """
    Context manager to manage memory consumption during training.

    Allows setting hard limit on the physical batch size as a just one line code change.
    Can be used both for simulating large logical batches with limited memory and for
    safeguarding against occasional large batches produced by
    :class:`~opacus.utils.uniform_sampler.UniformWithReplacementSampler`.

    Note that it doesn't modify the input DataLoader, you'd need to use new DataLoader
    returned by the context manager.

    BatchSplittingSampler will split large logical batches into smaller sub-batches with
    certain maximum size.
    On every step optimizer will check if the batch was the last physical batch comprising
    a logical one, and will change behaviour accordingly.

    If it was not the last, ``optimizer.step()`` will only clip per sample gradients and
    sum them into ``p.summed_grad`.` ``optimizer.zero_grad()`` will clear ``p.grad_sample``,
    but will leave ``p.grad`` and ``p.summed_grad``

    If the batch was the last one of the current logical batch, then
    ``optimizer.step()`` and ``optimizer.zero_grad()`` will behave normally.

    In addition to a single optimizer, it also accepts a sequence of optimizers
    (e.g. one per model, as in private GANs or actor-critic setups). The logical
    batch boundaries are then communicated to every optimizer that supports it;
    standard PyTorch optimizers without DP support are handled cleanly and are
    simply left to step on every physical batch.

    Example:
        >>> # Assuming you've initialized your objects and passed them to PrivacyEngine.
        >>> # For this example we assume data_loader is initialized with batch_size=4
        >>> model, optimizer, data_loader = _init_private_training()
        >>> criterion = nn.CrossEntropyLoss()
        >>> with BatchMemoryManager(
        ...     data_loader=data_loader, max_physical_batch_size=2, optimizer=optimizer
        ... ) as new_data_loader:
        ...     for data, label in new_data_loader:
        ...         assert len(data) <= 2 # physical batch is no more than 2
        ...         output = model(data)
        ...         loss = criterion(output, label)
        ...         loss.backward()
        ...         # optimizer won't actually make a step unless logical batch is over
        ...         optimizer.step()
        ...         # optimizer won't actually clear gradients unless logical batch is over
        ...         optimizer.zero_grad()
    """

    def __init__(
        self,
        *,
        data_loader: DataLoader,
        max_physical_batch_size: int,
        optimizer: Union[
            DPOptimizer, Optimizer, Sequence[Union[DPOptimizer, Optimizer]]
        ],
    ):
        self.data_loader = data_loader
        self.optimizers: List[Optimizer] = _to_optimizer_list(optimizer)
        self.max_physical_batch_size = max_physical_batch_size

    def __enter__(self):
        return wrap_data_loader(
            data_loader=self.data_loader,
            max_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizers,
        )

    def __exit__(self, type, value, traceback):
        pass
