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
import copy
import logging
from typing import Mapping, Optional

import torch
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t


logger = logging.getLogger(__name__)


class CollateFnWithEmpty:
    first_batch = None

    def __init__(self, collator_fn, batch_first=True, rand_on_empty=False):
        self.wrapped_colator_fn = collator_fn
        self.batch_first = batch_first
        self.rand_on_empty = rand_on_empty

    def __call__(self, batch):
        if len(batch) > 0:
            if not self.wrapped_colator_fn:
                output = batch
            else:
                output = self.wrapped_colator_fn(batch)
            if self.first_batch is None:
                self.first_batch = copy.deepcopy(output)
        else:
            if self.first_batch is None:
                raise ValueError(
                    "Jebiga... At least the first sampled batch shouldn't be empty..."
                )

            # materialize into empty with the same structure as list/dict
            output = self._make_empty_batch(self.first_batch)

        return output

    def _make_empty_batch(self, sample):
        if torch.is_tensor(sample):
            shape = list(sample.shape)
            # If it's at least 1D, set batch dim to 1; otherwise make a 0-length 1D tensor
            batch_dim = 0 if self.batch_first else 1
            shape[batch_dim] = 1 if self.rand_on_empty else 0
            if self.rand_on_empty:
                return torch.randint(
                    0, 2, shape, dtype=sample.dtype, device=sample.device
                )
            else:
                return torch.empty(shape, dtype=sample.dtype, device=sample.device)

        if isinstance(sample, Mapping):
            return {k: self._make_empty_batch(v) for k, v in sample.items()}

        if isinstance(sample, (list, tuple)):
            converted = [self._make_empty_batch(v) for v in sample]
            return type(sample)(converted)

        # base case
        return sample


def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    batch_first: bool = True,
    rand_on_empty: bool = False,
):
    """
    Wraps given collate function to handle empty batches.

    Args:
        collate_fn: collate function to wrap
        batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
        rand_on_empty: set ``True`` to return a batch containing random numbers when encountering
            empty batches rather than tensors with zero-length batch dimensions

    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs empty tensors with shapes from ``sample_empty_shapes`` if
        the input batch is of size 0
    """

    return CollateFnWithEmpty(
        collate_fn, batch_first=batch_first, rand_on_empty=rand_on_empty
    )


class DPDataLoader(DataLoader):
    """
    DataLoader subclass that always does Poisson sampling and supports empty batches
    by default.

    Typically instantiated via ``DPDataLoader.from_data_loader()`` method based
    on another DataLoader. DPDataLoader would preserve the behaviour of the original
    data loader, except for the two aspects.

    First, it switches ``batch_sampler`` to ``UniformWithReplacementSampler``, thus enabling
    Poisson sampling (i.e. each element in the dataset is selected to be in the
    next batch with a certain probability defined by ``sample_rate`` parameter).
    NB: this typically leads to a batches of variable size.
    NB2: By default, ``sample_rate`` is calculated based on the ``batch_size`` of the
    original data loader, so that the average batch size stays the same

    Second, it wraps collate function with support for empty batches.
    Most PyTorch modules will happily process tensors of shape ``(0, N, ...)``,
    but many collate functions will fail to produce such a batch. As with the
    Poisson sampling empty batches become a possibility, we need a DataLoader that
    can handle them.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        batch_first: bool = True,
        rand_on_empty: bool = False,
        **kwargs,
    ):
        """

        Args:
            dataset: See :class:`torch.utils.data.DataLoader`
            sample_rate: probability with which each element of the dataset is included
                in the next batch.
            num_workers: See :class:`torch.utils.data.DataLoader`
            collate_fn: See :class:`torch.utils.data.DataLoader`
            pin_memory: See :class:`torch.utils.data.DataLoader`
            drop_last: See :class:`torch.utils.data.DataLoader`
            timeout: See :class:`torch.utils.data.DataLoader`
            worker_init_fn: See :class:`torch.utils.data.DataLoader`
            multiprocessing_context: See :class:`torch.utils.data.DataLoader`
            generator: Random number generator used to sample elements
            prefetch_factor: See :class:`torch.utils.data.DataLoader`
            persistent_workers: See :class:`torch.utils.data.DataLoader`
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
                Selects between ``DistributedUniformWithReplacementSampler`` and
                ``UniformWithReplacementSampler`` sampler implementations
            rand_on_empty: set ``True`` to return a batch containing random numbers when encountering
                empty batches rather than tensors with zero-length batch dimensions
        """

        self.sample_rate = sample_rate
        self.distributed = distributed

        if distributed:
            batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        else:
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                batch_first=batch_first,
                rand_on_empty=rand_on_empty,
            ),
            generator=generator,
            **kwargs,
        )

    @classmethod
    def from_data_loader(
        cls,
        data_loader: DataLoader,
        *,
        distributed: bool = False,
        generator=None,
        batch_first: bool = True,
        rand_on_empty: bool = False,
    ):
        """
        Creates new ``DPDataLoader`` based on passed ``data_loader`` argument.

        Args:
            data_loader: Any DataLoader instance. Must not be over an ``IterableDataset``
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
            generator: Random number generator used to sample elements. Defaults to
                generator from the original data loader.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            rand_on_empty: set ``True`` to return a batch containing random numbers when encountering
                empty batches rather than tensors with zero-length batch dimensions



        Returns:
            New DPDataLoader instance, with all attributes and parameters inherited
            from the original data loader, except for sampling mechanism.

        Examples:
            >>> x, y = torch.randn(64, 5), torch.randint(0, 2, (64,))
            >>> dataset = TensorDataset(x,y)
            >>> data_loader = DataLoader(dataset, batch_size=4)
            >>> dp_data_loader = DPDataLoader.from_data_loader(data_loader)
        """

        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        return cls(
            dataset=data_loader.dataset,
            sample_rate=1 / len(data_loader),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
            batch_first=batch_first,
            rand_on_empty=rand_on_empty,
        )


def _is_supported_batch_sampler(sampler: Sampler):
    return (
        isinstance(sampler, BatchSampler)
        or isinstance(sampler, UniformWithReplacementSampler)
        or isinstance(sampler, DistributedUniformWithReplacementSampler)
    )


def switch_generator(*, data_loader: DataLoader, generator):
    """
    Creates new instance of a ``DataLoader``, with the exact same behaviour of the
    provided data loader, except for the source of randomness.

    Typically used to enhance a user-provided data loader object with cryptographically
    secure random number generator

    Args:
        data_loader: Any ``DataLoader`` object
        generator:  Random number generator object

    Returns:
        New ``DataLoader`` object with the exact same behaviour as the input data loader,
        except for the source of randomness.
    """
    batch_sampler = data_loader.batch_sampler

    if batch_sampler is None or not _is_supported_batch_sampler(batch_sampler):
        raise ValueError(
            "Non-batch processing is not supported: Opacus always assumes one of the input dimensions to be batch dimension."
        )

    if isinstance(batch_sampler, BatchSampler):
        if not hasattr(batch_sampler.sampler, "generator"):
            raise ValueError(
                "Target sampler doesn't have generator attribute: nothing to switch"
            )

        batch_sampler.sampler.generator = generator
    else:
        batch_sampler.generator = generator

    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )
