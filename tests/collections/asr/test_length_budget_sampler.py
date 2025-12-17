# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from torch.utils.data import ConcatDataset, IterableDataset

from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.parts.utils.asr_batching import get_length_budget_batch_sampler
from nemo.collections.asr.parts.utils.length_budget_sampler import (
    DistributedLengthBudgetBatchSampler,
    LengthBudgetBatchSampler,
)


class _DummyASRDataset(_AudioTextDataset):
    """Minimal _AudioTextDataset stand-in that only exposes duration info."""

    def __init__(self, durations):
        self.collection = [type("Sample", (), {"duration": float(d)})() for d in durations]

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        return idx

    @property
    def collate_fn(self):
        return lambda batch: batch


class _DummyModel:
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.global_rank = rank


@pytest.mark.unit
def test_length_budget_sampler_respects_budget_single_rank():
    lengths = [1.0, 2.5, 3.0, 10.0]
    sampler = LengthBudgetBatchSampler(lengths=lengths, length_budget=6.0, shuffle=False)
    batches = list(sampler)

    assert any(len(batch) == 1 and batch[0] == 3 for batch in batches)

    for batch in batches:
        lens = [lengths[i] for i in batch]
        cost = max(lens) * len(lens)
        if max(lens) <= 6.0:
            assert cost <= 6.0


@pytest.mark.unit
def test_length_budget_sampler_defaults_do_not_drop_or_shuffle():
    durations = [1.0, 1.0, 1.0, 1.0]
    sampler = get_length_budget_batch_sampler(_DummyModel(), _DummyASRDataset(durations), {"length_budget": 2.0})

    sampler.set_epoch(0)
    batches_epoch0 = list(sampler)

    sampler.set_epoch(1)
    batches_epoch1 = list(sampler)

    assert batches_epoch0 == batches_epoch1


@pytest.mark.unit
def test_distributed_length_budget_sampler_default_drop_last_padding_prevents_zero_steps():
    durations = [1.0, 1.0, 1.0]
    world_size = 4
    samplers = [
        get_length_budget_batch_sampler(
            _DummyModel(world_size=world_size, rank=rank),
            _DummyASRDataset(durations),
            {"length_budget": 10.0},
        )
        for rank in range(world_size)
    ]

    assert all(isinstance(sampler, DistributedLengthBudgetBatchSampler) for sampler in samplers)
    assert all(len(sampler) == 1 for sampler in samplers)
    assert all(list(sampler) for sampler in samplers)


@pytest.mark.unit
def test_distributed_length_budget_sampler_balances_steps():
    durations = [float(x) for x in range(1, 25)]
    world_size = 3
    sampler_per_rank = [
        DistributedLengthBudgetBatchSampler(
            lengths=durations,
            length_budget=20.0,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=123,
            drop_last=False,
        )
        for rank in range(world_size)
    ]

    for sampler in sampler_per_rank:
        sampler.set_epoch(1)

    steps = len(sampler_per_rank[0])
    assert steps > 0
    assert all(len(sampler) == steps for sampler in sampler_per_rank)

    costs_by_rank = []
    for sampler in sampler_per_rank:
        costs = []
        for batch in sampler:
            lens = [durations[i] for i in batch]
            costs.append(max(lens) * len(lens))
        costs_by_rank.append(costs)

    ratios = [min(step_costs) / max(step_costs) for step_costs in zip(*costs_by_rank)]
    assert min(ratios) > 0.0


@pytest.mark.unit
def test_length_budget_sampler_supports_concat_dataset():
    ds_a = _DummyASRDataset([1.0, 2.0])
    ds_b = _DummyASRDataset([3.0])
    concat = ConcatDataset([ds_a, ds_b])

    sampler = get_length_budget_batch_sampler(
        _DummyModel(world_size=2, rank=1),
        concat,
        {"length_budget": 6.0, "shuffle": False, "drop_last": False, "max_batch_size": 2},
    )

    assert isinstance(sampler, DistributedLengthBudgetBatchSampler)

    batches = list(sampler)
    assert len(sampler) == len(batches)

    flat_indices = [idx for batch in batches for idx in batch]
    assert set(flat_indices).issubset(set(range(len(concat))))


@pytest.mark.unit
def test_length_budget_sampler_rejects_iterable_dataset():
    class _DummyIterable(IterableDataset):
        def __iter__(self):
            yield 0

    with pytest.raises(ValueError):
        get_length_budget_batch_sampler(_DummyModel(), _DummyIterable(), {"length_budget": 5.0})
