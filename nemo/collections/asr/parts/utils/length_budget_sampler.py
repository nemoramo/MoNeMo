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

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional, Sequence

from torch.utils.data import Sampler

__all__ = ["LengthBudgetBatchSampler", "DistributedLengthBudgetBatchSampler", "_build_length_budget_batches"]


def _build_length_budget_batches(
    lengths: Sequence[float],
    length_budget: float,
    shuffle: bool,
    seed: int,
    epoch: int,
    max_batch_size: Optional[int] = None,
) -> List[List[int]]:
    """
    Build batches on a single rank by greedily grouping samples until the cost
    (max length in batch * batch size) exceeds the length budget.
    """
    if length_budget <= 0:
        raise ValueError(f"length_budget must be positive, got {length_budget}.")

    num_samples = len(lengths)
    indices = list(range(num_samples))

    if shuffle:
        generator = random.Random(seed + epoch)
        generator.shuffle(indices)

    batches: List[List[int]] = []
    current_batch: List[int] = []
    max_len_in_batch = 0.0

    for idx in indices:
        sample_len = max(float(lengths[idx]), 1e-6)

        if not current_batch:
            current_batch = [idx]
            max_len_in_batch = sample_len
            continue

        if max_batch_size is not None and len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = [idx]
            max_len_in_batch = sample_len
            continue

        updated_max = max(max_len_in_batch, sample_len)
        cost = updated_max * (len(current_batch) + 1)

        if cost <= length_budget:
            current_batch.append(idx)
            max_len_in_batch = updated_max
        else:
            batches.append(current_batch)
            current_batch = [idx]
            max_len_in_batch = sample_len

    if current_batch:
        batches.append(current_batch)

    return batches


class LengthBudgetBatchSampler(Sampler[List[int]]):
    """Single-rank length-budget batch sampler."""

    def __init__(
        self,
        lengths: Sequence[float],
        length_budget: float,
        shuffle: bool = False,
        seed: int = 0,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(None)
        self.lengths = [float(x) for x in lengths]
        self.length_budget = float(length_budget)
        self.shuffle = shuffle
        self.seed = seed
        self.max_batch_size = max_batch_size

        self._epoch = 0
        self._batches: Optional[List[List[int]]] = None
        # Lightning/Fabric expects to call `set_epoch()` on `dataloader.batch_sampler.sampler`.
        # When this object is passed as `batch_sampler`, expose `.sampler` to itself.
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        """Update epoch to change shuffle order."""
        self._epoch = int(epoch)
        self._batches = None

    def _ensure_batches(self) -> None:
        if self._batches is None:
            self._batches = _build_length_budget_batches(
                self.lengths,
                self.length_budget,
                self.shuffle,
                self.seed,
                self._epoch,
                self.max_batch_size,
            )

    def __iter__(self) -> Iterator[List[int]]:
        self._ensure_batches()
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        self._ensure_batches()
        return len(self._batches)


class DistributedLengthBudgetBatchSampler(Sampler[List[int]]):
    """
    Distributed length-budget sampler that builds a global batch list once, then
    partitions batches so each rank sees the same number of steps.
    """

    def __init__(
        self,
        lengths: Sequence[float],
        length_budget: float,
        world_size: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        balance_across_ranks: bool = True,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(None)
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}.")

        self.lengths = [float(x) for x in lengths]
        self.length_budget = float(length_budget)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.balance_across_ranks = balance_across_ranks
        self.max_batch_size = max_batch_size

        self._epoch = 0
        self._rank_batches: Optional[List[List[int]]] = None
        self._steps: Optional[int] = None
        # Lightning/Fabric expects to call `set_epoch()` on `dataloader.batch_sampler.sampler`.
        # When this object is passed as `batch_sampler`, expose `.sampler` to itself.
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        """Update epoch to change shuffle order."""
        self._epoch = int(epoch)
        self._rank_batches = None
        self._steps = None

    def _build_all_batches(self) -> None:
        batches = _build_length_budget_batches(
            self.lengths,
            self.length_budget,
            self.shuffle,
            self.seed,
            self._epoch,
            self.max_batch_size,
        )
        num_batches = len(batches)

        if num_batches == 0:
            self._rank_batches = []
            self._steps = 0
            return

        costs: List[float] = []
        for batch in batches:
            max_len = max(self.lengths[i] for i in batch)
            costs.append(max_len * len(batch))

        batch_indices: List[int] = list(range(num_batches))
        if self.balance_across_ranks:
            batch_indices.sort(key=lambda i: costs[i], reverse=True)

        if self.drop_last:
            usable = (num_batches // self.world_size) * self.world_size

            # Guard against silently dropping everything when drop_last=True
            if usable == 0:
                raise ValueError(
                    "DistributedLengthBudgetBatchSampler(drop_last=True) would drop all batches because "
                    f"num_batches({num_batches}) < world_size({self.world_size}). "
                    "Set drop_last=False (will pad), or increase dataset size / length_budget / max_batch_size."
                )

            batch_indices = batch_indices[:usable]
            steps = usable // self.world_size
        else:
            steps = math.ceil(num_batches / self.world_size)
            total_needed = steps * self.world_size
            if total_needed > num_batches:
                pad = total_needed - num_batches
                if self.balance_across_ranks:
                    # Prefer duplicating the cheapest batches when padding.
                    for idx in range(pad):
                        batch_indices.append(batch_indices[-1 - (idx % num_batches)])
                else:
                    for idx in range(pad):
                        batch_indices.append(batch_indices[idx % num_batches])

        rank_batches: List[List[int]] = []
        for step in range(steps):
            within_step_rank = (self.rank + step) % self.world_size if self.balance_across_ranks else self.rank
            batch_idx = batch_indices[step * self.world_size + within_step_rank]
            rank_batches.append(batches[batch_idx])

        self._rank_batches = rank_batches
        self._steps = steps

    def __iter__(self) -> Iterator[List[int]]:
        if self._rank_batches is None:
            self._build_all_batches()
        for batch in self._rank_batches:
            yield batch

    def __len__(self) -> int:
        if self._steps is None:
            self._build_all_batches()
        return self._steps


if __name__ == "__main__":
    import numpy as np

    random.seed(0)
    durations = [random.uniform(0.5, 30.0) for _ in range(1000)]
    world_size = 4
    length_budget = 120.0

    samplers = [
        DistributedLengthBudgetBatchSampler(
            lengths=durations,
            length_budget=length_budget,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=123,
            drop_last=True,
        )
        for rank in range(world_size)
    ]

    for epoch in range(2):
        print(f"===== EPOCH {epoch} =====")
        for sampler in samplers:
            sampler.set_epoch(epoch)

        all_costs = []
        for rank, sampler in enumerate(samplers):
            costs_rank = []
            total_samples = 0
            for batch in sampler:
                lens = [durations[i] for i in batch]
                cost = max(lens) * len(lens)
                costs_rank.append(cost)
                total_samples += len(batch)

            print(
                f"Rank {rank}: steps={len(sampler)}, samples={total_samples}, "
                f"cost_mean={np.mean(costs_rank):.2f}, "
                f"cost_min={np.min(costs_rank):.2f}, cost_max={np.max(costs_rank):.2f}"
            )
            all_costs.append(costs_rank)

        steps = len(samplers[0])
        ratios = []
        for step in range(steps):
            step_costs = [all_costs[r][step] for r in range(world_size)]
            ratios.append(min(step_costs) / max(step_costs))

        print(
            f"Per-step cost ratio: min={np.min(ratios):.3f}, "
            f"avg={np.mean(ratios):.3f}, max={np.max(ratios):.3f}"
        )
