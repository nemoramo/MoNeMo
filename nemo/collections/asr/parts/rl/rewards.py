# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Reward interfaces for ASR post-training (GSPO / GRPO, etc).

Design goals:
- Minimal runtime dependencies and easy to extend.
- Text-in/text-out rewards for RLVR-style post-training (verifiable reward, no reward model).
- Support composing multiple reward terms with weights.

TODO:
- Add optional LM reward shaping (KenLM / neural LM rescoring).
- Add length normalization and coverage rewards to mitigate short-hypothesis bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from nemo.collections.asr.metrics.wer import word_error_rate


@dataclass
class TextRewardComponent:
    """
    Base class for a single scalar reward term computed from hypothesis/ref text.

    Each component contributes: `weight * score(hyp, ref)`.
    """

    name: str
    weight: float = 1.0

    def score(self, hyp_text: str, ref_text: str) -> float:
        raise NotImplementedError

    def __call__(self, hyp_text: str, ref_text: str) -> float:
        return float(self.weight) * float(self.score(hyp_text=hyp_text, ref_text=ref_text))


class TextErrorRateReward(TextRewardComponent):
    """
    Verifiable reward based on WER/CER.

    - If `one_minus_error=True`: reward = 1 - (WER/CER)
    - Else: reward = -(WER/CER)
    """

    def __init__(
        self,
        name: str = "wer",
        weight: float = 1.0,
        use_cer: bool = False,
        one_minus_error: bool = True,
    ):
        super().__init__(name=name, weight=weight)
        self.use_cer = bool(use_cer)
        self.one_minus_error = bool(one_minus_error)

    def score(self, hyp_text: str, ref_text: str) -> float:
        err = float(word_error_rate([hyp_text], [ref_text], use_cer=self.use_cer))
        return (1.0 - err) if self.one_minus_error else (-err)


class WeightedSumTextReward:
    """
    Weighted sum of multiple text rewards.

    This is the recommended interface for GSPO post-training so that users can
    incrementally add reward shaping terms without touching trainer logic.
    """

    def __init__(self, components: List[TextRewardComponent]):
        if len(components) == 0:
            raise ValueError("`components` must be non-empty")
        self.components = components

    def compute(self, hyp_text: str, ref_text: str) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
            total_reward: weighted sum reward
            parts: dict of weighted contributions per component name
        """
        parts: Dict[str, float] = {}
        for component in self.components:
            parts[component.name] = float(component(hyp_text=hyp_text, ref_text=ref_text))
        total = float(sum(parts.values()))
        return total, parts

    def __call__(self, hyp_text: str, ref_text: str) -> float:
        total, _ = self.compute(hyp_text=hyp_text, ref_text=ref_text)
        return total

