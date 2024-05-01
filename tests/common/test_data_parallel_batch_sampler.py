"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
from typing import List, TypeVar

import numpy as np
import pytest
from torch.utils.data import Dataset, DistributedSampler

from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    StatefulDistributedSampler,
    UnsupportedDatasetError,
)

DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_ATOMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

T_co = TypeVar("T_co", covariant=True)


@pytest.fixture
def valid_dataset():
    class _Dataset(Dataset[T_co]):
        def data_sizes(self, batch_idx: List[int]) -> np.ndarray:
            return np.array(SIZE_ATOMS)[batch_idx]

        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    yield _Dataset(DATA)


@pytest.fixture
def invalid_dataset():
    class _Dataset(Dataset):
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return _Dataset(DATA)


def test_invalid_mode(invalid_dataset) -> None:
    with pytest.raises(
        ValueError,
        match="Only mode='atoms' or mode=True is supported, got mode='natoms'.",
    ):
        _ = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="natoms",
            on_error="raise",
        )

    with pytest.raises(
        ValueError,
        match="Only mode='atoms' or mode=True is supported, got mode='neighbors'.",
    ):
        _ = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="neighbors",
            on_error="raise",
        )


def test_invalid_dataset(invalid_dataset) -> None:
    with pytest.raises(UnsupportedDatasetError):
        sampler = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            on_error="raise",
        )
        _ = sampler._data_sizes(list(range(len(SIZE_ATOMS))))


def test_valid_dataset(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="atoms",
        on_error="raise",
    )
    assert (
        sampler._data_sizes(list(range(len(SIZE_ATOMS))))
        == np.array(SIZE_ATOMS)
    ).all()


def test_disabled(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode=False,
        on_error="raise",
    )
    assert sampler.disabled or not sampler._dist_enabled()


def test_single_node(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=1,
        device=None,
        mode="atoms",
        on_error="raise",
    )
    assert sampler.disabled or not sampler._dist_enabled()


def test_stateful_distributed_sampler_noshuffle(valid_dataset) -> None:
    for batch_size in range(1, 4):
        sampler = StatefulDistributedSampler(
            dataset=valid_dataset,
            batch_size=batch_size,
            rank=0,
            num_replicas=1,
            seed=0,
            shuffle=False,
        )
        full_list = list(sampler)
        assert full_list == list(range(len(full_list)))


def test_stateful_distributed_sampler_vs_distributed_sampler(
    valid_dataset,
) -> None:
    for seed in [0, 100, 200]:
        for batch_size in range(1, 4):
            stateful_sampler = StatefulDistributedSampler(
                dataset=valid_dataset,
                batch_size=batch_size,
                rank=0,
                num_replicas=2,
                seed=seed,
                shuffle=True,
                drop_last=True,
            )
            sampler = DistributedSampler(
                dataset=valid_dataset,
                rank=0,
                num_replicas=2,
                seed=seed,
                shuffle=True,
                drop_last=True,
            )
            assert list(stateful_sampler) == list(sampler)


def test_stateful_distributed_sampler(valid_dataset) -> None:
    for batch_size in range(1, 4):
        sampler = StatefulDistributedSampler(
            dataset=valid_dataset,
            batch_size=batch_size,
            rank=0,
            num_replicas=1,
            seed=0,
        )
        original_order = list(sampler)

        offset_step = 2
        loaded_sampler = StatefulDistributedSampler(
            dataset=valid_dataset,
            batch_size=batch_size,
            rank=0,
            seed=0,
            num_replicas=1,
        )
        loaded_sampler.set_epoch_and_start_iteration(0, offset_step)
        assert (
            list(loaded_sampler) == original_order[offset_step * batch_size :]
        )

        diff_sampler = StatefulDistributedSampler(
            dataset=valid_dataset,
            batch_size=batch_size,
            rank=0,
            num_replicas=1,
            seed=1,
        )
        assert list(diff_sampler) != original_order


def test_stateful_distributed_sampler_numreplicas(valid_dataset) -> None:
    fullset = set(range(len(valid_dataset)))
    for drop_last in [True, False]:
        for num_replicas in range(1, 4):
            for batch_size in [1]:
                samplers = [
                    StatefulDistributedSampler(
                        dataset=valid_dataset,
                        batch_size=batch_size,
                        rank=rank,
                        seed=0,
                        drop_last=drop_last,
                        num_replicas=num_replicas,
                    )
                    for rank in range(num_replicas)
                ]

                # make sure each subset only differs by at most one element in size
                len_samplers = np.array(
                    [len(list(sampler)) for sampler in samplers]
                )
                assert ((len_samplers - len_samplers.min()) <= 1).all()

                concat_idxs = functools.reduce(
                    lambda x, y: x + y, [list(sampler) for sampler in samplers]
                )
                if drop_last:
                    # make sure each subset is mutually exclusive and union covers the fullset
                    assert len(concat_idxs) == len(np.unique(concat_idxs))
                else:
                    assert set(concat_idxs) == fullset


def test_stateful_distributed_sampler_numreplicas_drop_last(
    valid_dataset,
) -> None:
    fullset = set(range(len(valid_dataset)))
    for num_replicas in range(1, 4):
        for batch_size in range(1, 4):
            samplers = [
                StatefulDistributedSampler(
                    dataset=valid_dataset,
                    batch_size=batch_size,
                    rank=rank,
                    seed=0,
                    num_replicas=num_replicas,
                    drop_last=True,
                )
                for rank in range(num_replicas)
            ]

            # make sure each subset only differs by at most one element in size
            len_samplers = np.array(
                [len(list(sampler)) for sampler in samplers]
            )
            assert ((len_samplers - len_samplers.min()) <= 1).all()

            # make sure each subset is mutually exclusive and union covers the fullset
            concat_idxs = functools.reduce(
                lambda x, y: x + y, [list(sampler) for sampler in samplers]
            )
            assert len(concat_idxs) == len(np.unique(concat_idxs))
            assert (
                len(concat_idxs)
                == (len(fullset) // num_replicas) * num_replicas
            )