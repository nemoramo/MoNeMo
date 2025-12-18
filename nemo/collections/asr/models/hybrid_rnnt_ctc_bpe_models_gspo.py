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
Memory-first GSPO post-training for Hybrid TDT+CTC (FastConformer-Hybrid-TDT-CTC-BPE).

This module is intentionally written to be "slow but small":
- Prefer `batch_size=1` and gradient accumulation.
- Freeze encoder + preprocessor and cache encoder outputs.
- Generate a small n-best group using built-in RNNT/TDT beam decoding.
- Compute sequence-level logp via TDT NLL (marginal over alignments).

References (high level):
- PPO (clipped policy gradient): https://arxiv.org/abs/1707.06347
- TDT (Token-and-Duration Transducer): https://arxiv.org/abs/2304.06795
- GSPO / sequence-level importance ratios discussion: https://arxiv.org/pdf/2507.18071

TODO:
- Add optional LM reward shaping (KenLM / neural rescoring).
- Add an optional frozen reference model for KL regularization.
- Consider length-normalized rewards / penalties to mitigate short-hypothesis bias.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.rl.rewards import TextErrorRateReward, TextRewardComponent, WeightedSumTextReward
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@dataclass
class GSPOConfig:
    """
    GSPO config for ASR post-training.

    Note: This is a lightweight config container. The model reads from `cfg.gspo`,
    but you can also construct this dataclass and merge it into the model config.
    """

    # PPO-style clipping epsilon.
    clip_eps: float = 0.2

    # Beam/n-best group size.
    group_size: int = 4

    # Reward uses CER if True, else WER.
    use_cer: bool = False

    # Reward definition: `reward = 1 - WER/CER` if True, else `reward = -(WER/CER)`.
    reward_is_one_minus_error: bool = True
    # Unified reward interface (preferred).
    # If `reward` is provided, `use_cer` / `reward_is_one_minus_error` act only as defaults.
    reward: Optional[dict] = None

    # Group normalize rewards into advantages (r-mean)/std.
    normalize_advantage: bool = True
    advantage_eps: float = 1e-8

    # Optional supervised anchor (ground-truth NLL) to reduce drift.
    sft_weight: float = 0.0

    # Memory-first knobs.
    freeze_preprocessor: bool = True
    # If True, encoder parameters will not be updated and encoder forward will run under `torch.no_grad()`.
    # Default is False to match standard fine-tuning expectations; enable it to save memory.
    freeze_encoder: bool = False
    # If True, encoder forward runs under `torch.no_grad()` (even if `freeze_encoder=False`).
    # This is a memory-saving option but it will prevent encoder updates.
    encoder_no_grad: bool = False

    # Safety: Lightning may call `model.train()` and re-enable `.training=True` on all submodules,
    # even if they are frozen (`requires_grad=False`). This can re-enable dropout in the encoder,
    # causing non-deterministic rollouts / logp and destabilizing GSPO.
    enforce_eval_on_frozen_modules: bool = True

    # Disable dropout for rollout + logp computations.
    disable_decoder_dropout: bool = True
    disable_joint_dropout: bool = True


def _as_tensor_cfg(cfg: Optional[DictConfig], schema: GSPOConfig) -> DictConfig:
    if cfg is None:
        return OmegaConf.structured(schema)
    # Merge user cfg over defaults (user takes precedence).
    return OmegaConf.merge(OmegaConf.structured(schema), cfg)


def _group_normalize(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Args:
        rewards: Tensor[K]
    Returns:
        advantages: Tensor[K]
    """
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def gspo_clipped_loss_seq(
    logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, clip_eps: float
) -> torch.Tensor:
    """
    Sequence-level clipped objective (PPO-style).

    Args:
        logp_new: Tensor[K], requires grad.
        logp_old: Tensor[K], detached baseline.
        advantages: Tensor[K], detached.
    Returns:
        Scalar loss (to minimize).
    """
    ratio = torch.exp(logp_new - logp_old)
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    obj1 = ratio * advantages
    obj2 = ratio_clipped * advantages
    return (-torch.minimum(obj1, obj2)).mean()


def _freeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


class EncDecHybridRNNTCTCBPEModelGSPO(EncDecHybridRNNTCTCBPEModel):
    """
    EncDecHybridRNNTCTCBPEModel variant that overrides `training_step()` with
    a memory-first GSPO post-training objective for the RNNT/TDT branch.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer | None = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # Resolve GSPO config.
        self.gspo_cfg: DictConfig = _as_tensor_cfg(self.cfg.get("gspo", None), GSPOConfig())

        # Ensure decoding is configured for n-best rollouts (beam + return_best_hypothesis=False).
        self._configure_gspo_decoding()

        # Create a per-sample (reduction=None) transducer NLL module for logp computations.
        self._gspo_nll = self._build_per_sample_rnnt_loss()

        # Reward function (supports weighted sum of multiple components).
        self._reward_fn = self._build_reward_fn()

        # Memory-first defaults: freeze modules that we never want to train.
        if self.gspo_cfg.get("freeze_preprocessor", True):
            _freeze_module(self.preprocessor)
            self.preprocessor.eval()

        if self.gspo_cfg.get("freeze_encoder", False):
            _freeze_module(self.encoder)
            self.encoder.eval()

        # Encoder forward should avoid graph creation when the encoder is not being trained.
        self._train_encoder = not bool(self.gspo_cfg.get("freeze_encoder", False)) and not bool(
            self.gspo_cfg.get("encoder_no_grad", False)
        )
        self._encoder_no_grad = (not self._train_encoder) or bool(self.gspo_cfg.get("encoder_no_grad", False))

    def _build_reward_fn(self):
        """
        Build a unified reward function from `model.gspo.reward`.

        Supported config styles:

        1) Simple built-in registry (recommended for YAML-only users):

            model:
              gspo:
                reward:
                  components:
                    - name: wer
                      weight: 1.0
                      one_minus_error: true

        2) Hydra instantiate (for custom reward classes):

            model:
              gspo:
                reward:
                  _target_: nemo.collections.asr.parts.rl.rewards.WeightedSumTextReward
                  components:
                    - _target_: nemo.collections.asr.parts.rl.rewards.TextErrorRateReward
                      name: wer
                      weight: 1.0
                      use_cer: false
                      one_minus_error: true
        """
        reward_cfg = self.gspo_cfg.get("reward", None)
        if reward_cfg is None:
            use_cer = bool(self.gspo_cfg.get("use_cer", False))
            return TextErrorRateReward(
                name="cer" if use_cer else "wer",
                weight=1.0,
                use_cer=use_cer,
                one_minus_error=bool(self.gspo_cfg.get("reward_is_one_minus_error", True)),
            )

        # Hydra-style config with `_target_`.
        if isinstance(reward_cfg, DictConfig) and reward_cfg.get("_target_", None) is not None:
            try:
                from hydra.utils import instantiate
            except Exception as e:
                raise RuntimeError("Hydra is required to instantiate `model.gspo.reward` with `_target_`.") from e
            return instantiate(reward_cfg)

        # Simple registry: `reward.components = [{name, weight, ...}, ...]`.
        components_cfg = reward_cfg.get("components", None) if isinstance(reward_cfg, (dict, DictConfig)) else None
        if not components_cfg:
            raise ValueError("`model.gspo.reward` must define either `_target_` or a non-empty `components` list.")

        components: List[TextRewardComponent] = []
        for comp_cfg in components_cfg:
            # Allow Hydra instantiate at the component level.
            if isinstance(comp_cfg, DictConfig) and comp_cfg.get("_target_", None) is not None:
                from hydra.utils import instantiate

                component = instantiate(comp_cfg)
                components.append(component)
                continue

            name = str(comp_cfg.get("name", "wer"))
            weight = float(comp_cfg.get("weight", 1.0))

            if name in {"wer", "cer"}:
                use_cer = bool(comp_cfg.get("use_cer", name == "cer"))
                one_minus_error = bool(comp_cfg.get("one_minus_error", True))
                components.append(
                    TextErrorRateReward(
                        name=name,
                        weight=weight,
                        use_cer=use_cer,
                        one_minus_error=one_minus_error,
                    )
                )
                continue

            raise ValueError(
                f"Unknown reward component name `{name}`. "
                "Provide a `_target_` to use a custom reward class, or use one of: {wer, cer}."
            )

        return WeightedSumTextReward(components=components)

    def _configure_gspo_decoding(self) -> None:
        decoding_cfg = copy.deepcopy(self.cfg.decoding)
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "beam"
            decoding_cfg.beam.return_best_hypothesis = False
            decoding_cfg.beam.beam_size = int(self.gspo_cfg.get("group_size", 4))

        # This updates `self.decoding` + `self.wer` + internal configs.
        self.change_decoding_strategy(decoding_cfg=decoding_cfg, decoder_type="rnnt", verbose=False)

    def _build_per_sample_rnnt_loss(self) -> RNNTLoss:
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))
        num_classes = self.joint.num_classes_with_blank - 1
        if loss_name == "tdt":
            num_classes = num_classes - self.joint.num_extra_outputs

        return RNNTLoss(
            num_classes=num_classes,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            reduction=None,  # critical for sequence-level logp
        )

    def _strip_special_token_ids(self, token_ids: List[int]) -> List[int]:
        pad_id = getattr(self.tokenizer, "pad_id", None)
        bos_id = getattr(self.tokenizer, "bos_id", None)
        eos_id = getattr(self.tokenizer, "eos_id", None)
        special_ids = {x for x in (pad_id, bos_id, eos_id) if isinstance(x, int) and x >= 0}
        return [t for t in token_ids if t not in special_ids]

    def _tokens_to_text(self, token_ids: List[int]) -> str:
        token_ids = self._strip_special_token_ids(token_ids)
        # NOTE: Use the same reference decoding path as NeMo's `WER` metric
        # (`WER.update()` calls `self.decoding.decode_ids_to_str(target)`).
        if hasattr(self, "decoding") and hasattr(self.decoding, "decode_ids_to_str"):
            try:
                return self.decoding.decode_ids_to_str(token_ids).strip()
            except Exception:
                # Fall back to tokenizer decoding if decoding object is unavailable.
                return self.tokenizer.ids_to_text(token_ids).strip()
        return self.tokenizer.ids_to_text(token_ids).strip()

    def _hypothesis_to_token_ids(self, hyp: Hypothesis) -> List[int]:
        token_ids = hyp.y_sequence.tolist() if isinstance(hyp.y_sequence, torch.Tensor) else list(hyp.y_sequence)
        blank_id = getattr(self.decoding, "blank_id", None)
        if blank_id is None:
            blank_id = self.tokenizer.tokenizer.vocab_size

        # Align with `RNNTBPEDecoding.decode_hypothesis()` filtering logic:
        # - TDT: drop blank + duration outputs (>= blank_id)
        # - RNNT: drop blank id (== blank_id)
        is_tdt = getattr(self.decoding, "_is_tdt", False)
        if is_tdt:
            token_ids = [t for t in token_ids if t < blank_id]
        else:
            token_ids = [t for t in token_ids if t != blank_id]
        return token_ids

    def _encode_one(self, signal: torch.Tensor, signal_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single sample (or a small batch) without building an autograd graph.
        """
        if self._encoder_no_grad:
            with torch.no_grad():
                processed_signal, processed_signal_len = self.preprocessor(input_signal=signal, length=signal_len)
                encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        else:
            processed_signal, processed_signal_len = self.preprocessor(input_signal=signal, length=signal_len)
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        return encoded, encoded_len

    def _compute_logp_from_encoder(self, encoded: torch.Tensor, encoded_len: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
        """
        Compute log p(y|x) for a single hypothesis token sequence under the current model.

        Returns:
            Scalar tensor (logp).
        """
        # TODO: Decide how to handle empty hypotheses (rare, but possible for very short/noisy audio).
        if len(token_ids) == 0:
            return torch.tensor(0.0, device=encoded.device, dtype=torch.float32)

        targets = torch.tensor(token_ids, device=encoded.device, dtype=torch.long).unsqueeze(0)
        target_lens = torch.tensor([len(token_ids)], device=encoded.device, dtype=torch.long)

        dec, dec_lens, _ = self.decoder(targets=targets, target_length=target_lens)

        if self.joint.fuse_loss_wer:
            # Use fused Joint+Loss path to avoid materializing `joint` outside this scope.
            # NOTE: We temporarily set the loss reduction to None so that the fused joint returns per-sample NLL.
            loss_reduction = self.loss.reduction
            self.loss.reduction = None
            nll, _, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=dec,
                encoder_lengths=encoded_len,
                transcripts=targets,
                transcript_lengths=target_lens,
                compute_wer=False,
            )
            self.loss.reduction = loss_reduction
        else:
            # Align target length semantics with NeMo's standard RNNT/TDT training path:
            # - Decoder returns `dec_lens` (typically equal to the passed-in `target_lens`)
            # - RNNTLoss expects transcript lengths *without* the implicit SOS added inside the decoder.
            if not torch.equal(dec_lens, target_lens):
                raise RuntimeError(
                    "RNNTDecoder returned a target_length different from the provided `target_lens`. "
                    "Please verify target length semantics to avoid off-by-one errors."
                )
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=dec)
            nll = self._gspo_nll(
                log_probs=joint, targets=targets, input_lengths=encoded_len, target_lengths=target_lens
            )

        # `reduction=None` => Tensor[B]; B==1 here.
        return -nll.squeeze(0)

    def _ensure_policy_eval_mode(self) -> None:
        """
        Disable dropout for rollout/logp computations for better GSPO stability.

        Lightning will set the full model to train() for you. We selectively set
        submodules back to eval() each step.
        """
        if self.gspo_cfg.get("enforce_eval_on_frozen_modules", True):
            if self.gspo_cfg.get("freeze_preprocessor", True):
                self.preprocessor.eval()
            # Encoder dropout must be disabled when the encoder is not being trained (frozen or no-grad),
            # otherwise GSPO rollouts/logp become non-deterministic.
            if not self._train_encoder:
                self.encoder.eval()

        if self.gspo_cfg.get("disable_decoder_dropout", True):
            self.decoder.eval()
        if self.gspo_cfg.get("disable_joint_dropout", True):
            self.joint.eval()

    def _reward_from_text(self, hyp_text: str, ref_text: str) -> float:
        # Keep this method as a stable hook for trainer logic.
        # Prefer configuring rewards via `model.gspo.reward`.
        return float(self._reward_fn(hyp_text=hyp_text, ref_text=ref_text))

    def _gspo_one_sample(
        self, signal: torch.Tensor, signal_len: torch.Tensor, transcript: torch.Tensor, transcript_len: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        GSPO objective for a single sample (batch size 1).
        """
        device = signal.device

        ref_token_ids = transcript[0, : int(transcript_len.item())].tolist()
        ref_text = self._tokens_to_text(ref_token_ids)

        # 1) Encode once.
        encoded, encoded_len = self._encode_one(signal, signal_len)
        # Detach encoder outputs if encoder is not being trained (saves memory and avoids accidental grads).
        if not self._train_encoder:
            encoded = encoded.detach()

        # 2) Rollout: beam decode n-best hypotheses from cached encoder outputs.
        with torch.no_grad():
            hyps_batch = self.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)

        # `rnnt_decoder_predictions_tensor()` returns:
        # - List[List[Hypothesis]] for n-best (beam, return_best_hypothesis=False)
        # - List[Hypothesis] for best-only decoding
        if len(hyps_batch) == 0:
            raise RuntimeError("Decoding returned an empty hypothesis list; check decoding config and inputs.")

        if isinstance(hyps_batch[0], list):
            hyps = hyps_batch[0]
        else:
            # Best-only decoding is not sufficient for GSPO.
            raise RuntimeError(
                "Decoding returned best-only hypotheses, but GSPO requires n-best. "
                "Set `model.decoding.strategy=beam` and `model.decoding.beam.return_best_hypothesis=False`."
            )
        if len(hyps) == 0:
            raise RuntimeError("Decoding returned an empty hypothesis list; check decoding config and inputs.")

        # 3) Compute rewards and advantages on CPU-ish (tiny tensors).
        hyp_texts: List[str] = [h.text for h in hyps]
        rewards = torch.tensor([self._reward_from_text(t, ref_text) for t in hyp_texts], device=device, dtype=torch.float32)
        if self.gspo_cfg.get("normalize_advantage", True):
            advantages = _group_normalize(rewards, eps=float(self.gspo_cfg.get("advantage_eps", 1e-8)))
        else:
            advantages = rewards - rewards.mean()

        # 4) Compute logp_old (no_grad baseline) and logp_new (with grad) one-by-one to reduce peak memory.
        logp_old_list: List[torch.Tensor] = []
        logp_new_list: List[torch.Tensor] = []

        with torch.no_grad():
            for hyp in hyps:
                token_ids = self._hypothesis_to_token_ids(hyp)
                logp_old_list.append(self._compute_logp_from_encoder(encoded, encoded_len, token_ids).detach())

        for hyp in hyps:
            token_ids = self._hypothesis_to_token_ids(hyp)
            logp_new_list.append(self._compute_logp_from_encoder(encoded, encoded_len, token_ids))

        logp_old = torch.stack(logp_old_list, dim=0)
        logp_new = torch.stack(logp_new_list, dim=0)

        rl_loss = gspo_clipped_loss_seq(
            logp_new=logp_new, logp_old=logp_old, advantages=advantages.detach(), clip_eps=float(self.gspo_cfg.clip_eps)
        )

        # 5) Optional supervised anchor (ground truth NLL).
        total_loss = rl_loss
        sft_loss = None
        sft_weight = float(self.gspo_cfg.get("sft_weight", 0.0))
        if sft_weight > 0.0:
            gt_ids = transcript[0, : int(transcript_len.item())].tolist()
            gt_ids = self._strip_special_token_ids(gt_ids)
            gt_logp = self._compute_logp_from_encoder(encoded, encoded_len, gt_ids)
            sft_loss = -gt_logp  # minimize NLL
            total_loss = total_loss + sft_weight * sft_loss

        metrics: Dict[str, torch.Tensor] = {
            "gspo_rl_loss": rl_loss.detach(),
            "gspo_reward_mean": rewards.mean().detach(),
            "gspo_reward_std": rewards.std(unbiased=False).detach(),
        }
        if sft_loss is not None:
            metrics["gspo_sft_loss"] = sft_loss.detach()

        return total_loss, metrics

    # PTL override
    def training_step(self, batch, batch_nb):
        # NOTE: This post-training step intentionally ignores the auxiliary CTC loss.
        # TODO: Consider keeping a small CTC anchor if it improves stability for your dataset.

        signal, signal_len, transcript, transcript_len = batch

        # Keep rollout/logp deterministic (dropout off).
        self._ensure_policy_eval_mode()

        # Memory-first implementation assumes small batches and loops over samples.
        batch_size = int(signal.shape[0])
        losses: List[torch.Tensor] = []
        reward_means: List[torch.Tensor] = []
        reward_stds: List[torch.Tensor] = []

        for i in range(batch_size):
            loss_i, metrics_i = self._gspo_one_sample(
                signal[i : i + 1], signal_len[i : i + 1], transcript[i : i + 1], transcript_len[i : i + 1]
            )
            losses.append(loss_i)
            reward_means.append(metrics_i["gspo_reward_mean"])
            reward_stds.append(metrics_i["gspo_reward_std"])

        loss = torch.stack(losses).mean()

        # Logging (cheap scalars only).
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_gspo_reward_mean", torch.stack(reward_means).mean(), prog_bar=False, sync_dist=True)
        self.log("train_gspo_reward_std", torch.stack(reward_stds).mean(), prog_bar=False, sync_dist=True)

        return loss
