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

Important note (PPO semantics):
- Standard PPO uses samples from a behavior policy `pi_old` and optimizes a (potentially updated) current policy `pi_new`.
- This implementation is intentionally minimal and computes `logp_old` and `logp_new` using the same parameters within
  a single `training_step()`. As a result, the importance ratio is typically ~1 and the clipped objective behaves like a
  sequence-level on-policy policy gradient / MWER-style risk minimization objective, wrapped in a PPO-shaped loss.
- The PPO/GSPO "shell" is kept to make it easy to extend toward stricter PPO semantics later (see TODO roadmap below).

References (high level):
- PPO (clipped policy gradient): https://arxiv.org/abs/1707.06347
- TDT (Token-and-Duration Transducer): https://arxiv.org/abs/2304.06795
- GSPO / sequence-level importance ratios discussion: https://arxiv.org/pdf/2507.18071

TODO roadmap (future work):
- True PPO-style sample reuse (PPO epochs): rollout once, cache `logp_old`, do multiple optimizer steps by recomputing
  `logp_new` against the cached group. Add PPO diagnostics (clip fraction, approx KL, ratio stats).
- Old-policy snapshot: maintain a lagged copy of decoder/joint as `pi_old` (sync every N steps), use it for rollout and
  `logp_old` while optimizing the current policy. Prefer copying decoder/joint only (keep encoder frozen) for memory.
- KL regularization: add optional KL-to-reference (either snapshot or a separate frozen reference) to control drift.
- Reward shaping: add optional LM score / length normalization / other verifiable rewards via the unified reward interface.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.rl.rewards import TextErrorRateReward, TextRewardComponent, WeightedSumTextReward
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging


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

    # Diagnostics / logging.
    # When enabled, logs PPO-style stats (ratio, clip fraction, approx KL) as "canaries" for silent bugs.
    log_diagnostics: bool = True
    # Reduce logging overhead by emitting diagnostics every N optimizer steps.
    log_diagnostics_every_n_steps: int = 50
    # Additionally print a compact diagnostics line to the Python logger every N optimizer steps (0 disables).
    log_text_every_n_steps: int = 0
    # Optional: record coarse-grained encode/decode/logp timings (adds small CPU overhead).
    log_timings: bool = False
    # If True and running on CUDA, synchronize around timed blocks for more accurate timings (slower).
    timing_sync_cuda: bool = False
    # Optional: log gradient norms of decoder/joint (and encoder if trained).
    log_grad_norm: bool = False
    grad_norm_every_n_steps: int = 50

    # Ensure expensive diagnostics/printing run at most once per optimizer step (global_step),
    # which is important under gradient accumulation.
    diagnostics_once_per_global_step: bool = True

    # Distributed logging controls (useful when running DDP; keep defaults to preserve current behavior).
    sync_dist_core_metrics: bool = True
    sync_dist_extra_metrics: bool = True

    # One-time hard validation (optional):
    # Verify that the fused joint+loss path returns per-sample NLL consistent with a reference computation
    # that materializes the joint tensor and calls the loss directly.
    #
    # WARNING: This can be expensive and may OOM for long T/U since it materializes [B, T, U, V] tensors.
    validate_fused_logp_once: bool = False
    # Approximate upper bound on joint output elements (B * T * U * V) allowed for validation.
    validate_fused_logp_max_joint_elements: int = 50_000_000

    # Roadmap: PPO epochs (sample reuse).
    # NOTE: `ppo_epochs>1` is not implemented yet in the automatic-optimization Lightning path.
    ppo_epochs: int = 1

    # Optimization: Batch N-best hypotheses for logp computation.
    # Faster but uses more memory.
    batch_hypotheses: bool = False

    # Penalty logp for empty hypotheses (e.g. -100.0).
    empty_hyp_logp: float = -100.0


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


def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Small-sample safe Pearson correlation coefficient for diagnostics.

    Returns 0 if either vector has near-zero variance.
    """
    if x.numel() < 2 or y.numel() < 2:
        return torch.zeros((), device=x.device, dtype=torch.float32)

    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    return (x * y).mean() / (denom + eps)


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
    # Numerical stability (bf16/fp16 friendly):
    # - Cast BEFORE subtraction to preserve small deltas (and reduce quantization).
    # - Compute exp in fp32.
    # - Clamp delta to avoid exp overflow in fp32 (overflow starts around ~88).
    logp_new_f = logp_new.to(dtype=torch.float32)
    logp_old_f = logp_old.to(dtype=torch.float32)
    adv_f = advantages.to(dtype=torch.float32)

    delta = logp_new_f - logp_old_f
    delta = delta.clamp(min=-80.0, max=80.0)

    ratio = torch.exp(delta)
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    obj1 = ratio * adv_f
    obj2 = ratio_clipped * adv_f
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
        ppo_epochs = int(self.gspo_cfg.get("ppo_epochs", 1))
        if ppo_epochs != 1:
            raise NotImplementedError(
                "`model.gspo.ppo_epochs>1` requires PPO-style sample reuse (multiple optimizer steps per rollout), "
                "which is not implemented yet in this Lightning automatic-optimization GSPO trainer. "
                "Keep `model.gspo.ppo_epochs=1` for now; see the TODO roadmap in this module."
            )

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
        self._gspo_fused_logp_validation_ran = False
        self._gspo_blank_id: Optional[int] = None
        self._gspo_blank_id_source: Optional[str] = None
        self._gspo_last_diagnostics_global_step: int = -1

        blank_id, blank_id_source = self._resolve_blank_id()
        self._gspo_blank_id = blank_id
        self._gspo_blank_id_source = blank_id_source
        if trainer is None or getattr(trainer, "is_global_zero", True):
            logging.info(f"GSPO blank_id resolved: {blank_id} (source={blank_id_source})")

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

    def _resolve_tokenizer_vocab_size(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Best-effort tokenizer vocab size resolver.

        Returns:
            vocab_size, source_str
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return None, None

        # Common NeMo tokenizers expose `vocab_size` directly (e.g., SentencePieceTokenizer sets an int attribute).
        if hasattr(tokenizer, "vocab_size"):
            try:
                vocab_size = getattr(tokenizer, "vocab_size")
                vocab_size = vocab_size() if callable(vocab_size) else vocab_size
                return int(vocab_size), "tokenizer.vocab_size"
            except Exception:
                pass

        # Some wrappers expose the underlying tokenizer instance as `tokenizer.tokenizer`.
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None:
            if hasattr(inner, "vocab_size"):
                try:
                    vocab_size = getattr(inner, "vocab_size")
                    vocab_size = vocab_size() if callable(vocab_size) else vocab_size
                    return int(vocab_size), "tokenizer.tokenizer.vocab_size"
                except Exception:
                    pass

            # SentencePieceProcessor typically provides `get_piece_size()`.
            if hasattr(inner, "get_piece_size"):
                try:
                    return int(inner.get_piece_size()), "tokenizer.tokenizer.get_piece_size()"
                except Exception:
                    pass

        return None, None

    def _resolve_blank_id(self) -> Tuple[int, str]:
        """
        Resolve the blank token id (or "blank threshold" for TDT) for hypothesis filtering.

        Prefer `decoding.blank_id` when available; otherwise fall back to model/loss/tokenizer metadata.
        """
        blank_id = getattr(self.decoding, "blank_id", None)
        if isinstance(blank_id, int):
            return int(blank_id), "decoding.blank_id"

        # RNNTLoss stores the blank index as a private attribute `_blank` (set from `num_classes`).
        blank_id = getattr(self._gspo_nll, "_blank", None)
        if isinstance(blank_id, int):
            return int(blank_id), "_gspo_nll._blank"

        # Joint knows the full output dimension (including blank and extra outputs).
        if hasattr(self, "joint") and hasattr(self.joint, "num_classes_with_blank"):
            is_tdt = bool(getattr(self.decoding, "_is_tdt", False))
            if is_tdt and hasattr(self.joint, "num_extra_outputs"):
                # For TDT, treat the "blank id" as a threshold that filters out duration outputs as well.
                # `num_classes_with_blank = V + 1 + extra`, so vocab size V is:
                #   V = num_classes_with_blank - 1 - num_extra_outputs
                return (
                    int(self.joint.num_classes_with_blank - 1 - self.joint.num_extra_outputs),
                    "joint.num_classes_with_blank - 1 - joint.num_extra_outputs",
                )
            return int(self.joint.num_classes_with_blank - 1), "joint.num_classes_with_blank - 1"

        vocab_size, vocab_source = self._resolve_tokenizer_vocab_size()
        if vocab_size is not None:
            return int(vocab_size), f"{vocab_source} (fallback)"

        raise RuntimeError(
            "Could not resolve blank_id for hypothesis filtering.\n"
            "Expected `self.decoding.blank_id`, `self._gspo_nll._blank`, or a tokenizer vocab size.\n"
            f"decoding={type(self.decoding).__name__} tokenizer={type(getattr(self, 'tokenizer', None)).__name__}"
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
            blank_id = self._gspo_blank_id
        if blank_id is None:
            blank_id, blank_id_source = self._resolve_blank_id()
            self._gspo_blank_id = blank_id
            self._gspo_blank_id_source = blank_id_source
            logging.info(f"GSPO blank_id resolved lazily: {blank_id} (source={blank_id_source})")

        # Align with `RNNTBPEDecoding.decode_hypothesis()` filtering logic:
        # - TDT: drop blank + duration outputs (>= blank_id)
        # - RNNT: drop blank id (== blank_id)
        is_tdt = getattr(self.decoding, "_is_tdt", False)
        if is_tdt:
            token_ids = [t for t in token_ids if t < blank_id]
        else:
            token_ids = [t for t in token_ids if t != blank_id]
        # Strip special tokens (BOS/EOS/PAD) to match the target tokenization semantics used in training.
        return self._strip_special_token_ids(token_ids)

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
            return torch.tensor(float(self.gspo_cfg.get("empty_hyp_logp", -100.0)), device=encoded.device, dtype=torch.float32)

        targets = torch.tensor(token_ids, device=encoded.device, dtype=torch.long).unsqueeze(0)
        target_lens = torch.tensor([len(token_ids)], device=encoded.device, dtype=torch.long)

        dec, dec_lens, _ = self.decoder(targets=targets, target_length=target_lens)

        if self.joint.fuse_loss_wer:
            # Use fused Joint+Loss path to avoid materializing `joint` outside this scope.
            # NOTE: We temporarily set the loss reduction to None so that the fused joint returns per-sample NLL.
            # This mutates shared module state, so it must be restored even if an exception is raised.
            joint_loss = getattr(self.joint, "loss", None)
            if joint_loss is None:
                raise RuntimeError("`joint.fuse_loss_wer=True` but `joint.loss` is None; cannot compute per-sample logp.")

            loss_reduction = joint_loss.reduction
            try:
                joint_loss.reduction = None
                nll, _, _, _ = self.joint(
                    encoder_outputs=encoded,
                    decoder_outputs=dec,
                    encoder_lengths=encoded_len,
                    transcripts=targets,
                    transcript_lengths=target_lens,
                    compute_wer=False,
                )
            finally:
                joint_loss.reduction = loss_reduction
        else:
            # Align target length semantics with NeMo's standard RNNT/TDT training path:
            # - Decoder returns `dec_lens` (typically equal to the passed-in `target_lens`)
            # - RNNTLoss expects transcript lengths *without* the implicit SOS added inside the decoder.
            # NOTE: Some decoder implementations may return lengths with different dtype (int32 vs int64) or
            # different semantics (e.g., U+1 when including SOS). Keep this as fail-fast, but make it diagnostic.
            dec_lens_long = dec_lens.to(dtype=torch.long)
            target_lens_long = target_lens.to(dtype=torch.long)
            if not torch.equal(dec_lens_long, target_lens_long):
                dec_list = dec_lens_long.detach().cpu().tolist()
                tgt_list = target_lens_long.detach().cpu().tolist()
                delta = [int(d - t) for d, t in zip(dec_list, tgt_list)]

                loss_name, _ = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))
                decoding_model_type = getattr(getattr(self.cfg, "decoding", None), "model_type", None)
                is_tdt = bool(getattr(self.decoding, "_is_tdt", False))

                raise RuntimeError(
                    "RNNTDecoder returned a target_length different from the provided `target_lens`. "
                    "Please verify target length semantics to avoid off-by-one errors.\n"
                    f"dec_lens={dec_list} target_lens={tgt_list} delta(dec-target)={delta}\n"
                    f"dec_lens_dtype={dec_lens.dtype} target_lens_dtype={target_lens.dtype}\n"
                    f"loss_name={loss_name} decoding_model_type={decoding_model_type} is_tdt={is_tdt}\n"
                    f"decoder={type(self.decoder).__name__} decoding={type(self.decoding).__name__}"
                )
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=dec)
            nll = self._gspo_nll(
                log_probs=joint, targets=targets, input_lengths=encoded_len, target_lengths=target_lens
            )

        # `reduction=None` => Tensor[B]; B==1 here.
        return -nll.squeeze(0)

    def _compute_logp_batch(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        token_ids_list: List[List[int]],
        *,
        compute_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Compute log p(y|x) for a batch of hypotheses.
        Returns: Tensor[K]
        """
        # Handle empty hypotheses
        non_empty_indices = []
        empty_indices = []
        for i, ids in enumerate(token_ids_list):
            if len(ids) > 0:
                non_empty_indices.append(i)
            else:
                empty_indices.append(i)

        empty_hyp_logp = float(self.gspo_cfg.get("empty_hyp_logp", -100.0))
        if not non_empty_indices:
            return torch.full((len(token_ids_list),), empty_hyp_logp, device=encoded.device, dtype=torch.float32)

        # Prepare batch for non-empty
        batch_ids = [token_ids_list[i] for i in non_empty_indices]
        max_u = max(len(ids) for ids in batch_ids)
        k = len(batch_ids)

        targets = torch.zeros((k, max_u), device=encoded.device, dtype=torch.long)
        target_lens = torch.tensor([len(ids) for ids in batch_ids], device=encoded.device, dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            targets[i, : len(ids)] = torch.tensor(ids, device=encoded.device, dtype=torch.long)

        # Expand encoder outputs: [1, T, D] -> [K, T, D]
        encoded_batch = encoded.expand(k, -1, -1)
        encoded_len_batch = encoded_len.expand(k)

        dec, dec_lens, _ = self.decoder(targets=targets, target_length=target_lens)
        if compute_diagnostics and bool(self.gspo_cfg.get("debug_check_batch_lens", True)):
            # Diagnostics-only semantic checks to catch silent off-by-one / length-definition bugs.
            # - In most NeMo RNNT/TDT decoders, `dec_lens` equals `target_lens` (no SOS counted in lengths),
            #   while the decoder output time dimension is `max(target_lens)+1` (SOS is prepended in the output).
            # - If your decoder differs, these checks should fail fast so we don't silently compute wrong logp.
            expected_u = int(target_lens.max().item()) + 1
            got_u = int(dec.shape[-1])

            dec_lens_long = dec_lens.to(dtype=torch.long) if dec_lens is not None else None
            target_lens_long = target_lens.to(dtype=torch.long)

            if got_u != expected_u:
                raise RuntimeError(
                    "Batched logp: decoder output U dimension mismatch. "
                    "Expected U == max(target_lens)+1 (SOS in decoder outputs).\n"
                    f"got_U={got_u} expected_U={expected_u} target_lens={target_lens_long.detach().cpu().tolist()}\n"
                    f"decoder={type(self.decoder).__name__}"
                )

            if dec_lens_long is not None and not torch.equal(dec_lens_long, target_lens_long):
                dec_list = dec_lens_long.detach().cpu().tolist()
                tgt_list = target_lens_long.detach().cpu().tolist()
                delta = [int(d - t) for d, t in zip(dec_list, tgt_list)]

                loss_name, _ = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))
                decoding_model_type = getattr(getattr(self.cfg, "decoding", None), "model_type", None)
                is_tdt = bool(getattr(self.decoding, "_is_tdt", False))

                raise RuntimeError(
                    "Batched logp: RNNTDecoder returned a target_length different from provided `target_lens`. "
                    "Please verify target length semantics to avoid off-by-one errors.\n"
                    f"dec_lens={dec_list} target_lens={tgt_list} delta(dec-target)={delta}\n"
                    f"dec_lens_dtype={dec_lens.dtype} target_lens_dtype={target_lens.dtype}\n"
                    f"loss_name={loss_name} decoding_model_type={decoding_model_type} is_tdt={is_tdt}\n"
                    f"decoder={type(self.decoder).__name__} decoding={type(self.decoding).__name__}"
                )

        if self.joint.fuse_loss_wer:
            # Use fused Joint+Loss path to avoid materializing `joint` outside this scope.
            joint_loss = getattr(self.joint, "loss", None)
            if joint_loss is None:
                raise RuntimeError(
                    "`joint.fuse_loss_wer=True` but `joint.loss` is None; cannot compute per-sample logp."
                )

            loss_reduction = joint_loss.reduction
            try:
                joint_loss.reduction = None
                nll, _, _, _ = self.joint(
                    encoder_outputs=encoded_batch,
                    decoder_outputs=dec,
                    encoder_lengths=encoded_len_batch,
                    transcripts=targets,
                    transcript_lengths=target_lens,
                    compute_wer=False,
                )
            finally:
                joint_loss.reduction = loss_reduction
        else:
            # Standard path
            joint = self.joint(encoder_outputs=encoded_batch, decoder_outputs=dec)
            nll = self._gspo_nll(
                log_probs=joint, targets=targets, input_lengths=encoded_len_batch, target_lengths=target_lens
            )

        logp_non_empty = -nll  # `reduction=None` => Tensor[K_non_empty]

        # Reconstruct full K-sized list while preserving gradients for non-empty hypotheses.
        out: List[torch.Tensor] = []
        j = 0
        for ids in token_ids_list:
            if len(ids) == 0:
                out.append(torch.tensor(empty_hyp_logp, device=encoded.device, dtype=torch.float32))
            else:
                out.append(logp_non_empty[j])
                j += 1
        return torch.stack(out, dim=0)

    def _maybe_validate_fused_logp(
        self, encoded: torch.Tensor, encoded_len: torch.Tensor, token_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        One-time "hard validation" for fused logp semantics.

        Compares per-sample fused NLL returned by `self.joint(... fuse_loss_wer=True)` (with reduction=None)
        against a reference computed by explicitly materializing the joint tensor and calling the RNNT/TDT loss.

        This is intended to catch silent shape/reduction mismatches in the fused path.
        """
        if self._gspo_fused_logp_validation_ran:
            return {}
        if not bool(self.gspo_cfg.get("validate_fused_logp_once", False)):
            return {}
        if not bool(getattr(self.joint, "fuse_loss_wer", False)):
            return {}

        # Only run on global rank 0 to avoid redundant expensive work.
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return {}

        device = encoded.device
        metrics: Dict[str, torch.Tensor] = {}

        # Mark ran regardless of outcome to avoid repeated expensive work.
        self._gspo_fused_logp_validation_ran = True

        token_ids = self._strip_special_token_ids(token_ids)
        if len(token_ids) == 0:
            return {"gspo_logp_validate_skipped": torch.tensor(1.0, device=device, dtype=torch.float32)}

        t = int(encoded_len.max().item())
        u = int(len(token_ids) + 1)  # decoder output length includes SOS
        v = int(getattr(self.joint, "num_classes_with_blank", 0))
        approx_joint_numel = int(t * u * max(1, v))

        max_numel = int(self.gspo_cfg.get("validate_fused_logp_max_joint_elements", 50_000_000))
        metrics["gspo_logp_validate_joint_numel"] = torch.tensor(float(approx_joint_numel), device=device)
        if approx_joint_numel > max_numel:
            metrics["gspo_logp_validate_skipped"] = torch.tensor(1.0, device=device, dtype=torch.float32)
            return metrics

        # Detach to avoid holding onto any graph from the main training path.
        encoded = encoded.detach()

        try:
            with torch.no_grad():
                targets = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
                target_lens = torch.tensor([len(token_ids)], device=device, dtype=torch.long)
                dec, _, _ = self.decoder(targets=targets, target_length=target_lens)

                # Fused per-sample NLL (what GSPO uses when `fuse_loss_wer=True`).
                joint_loss = getattr(self.joint, "loss", None)
                if joint_loss is None:
                    raise RuntimeError(
                        "`joint.fuse_loss_wer=True` but `joint.loss` is None; cannot validate fused logp."
                    )
                loss_reduction = joint_loss.reduction
                try:
                    joint_loss.reduction = None
                    nll_fused, _, _, _ = self.joint(
                        encoder_outputs=encoded,
                        decoder_outputs=dec,
                        encoder_lengths=encoded_len,
                        transcripts=targets,
                        transcript_lengths=target_lens,
                        compute_wer=False,
                    )
                finally:
                    joint_loss.reduction = loss_reduction

                # Reference NLL from explicit joint tensor + loss.
                # NOTE: This materializes [B, T, U, V] tensors; keep it behind the size guard above.
                #
                # Important: On CUDA, NeMo's RNNT/TDT joint often returns logits (log_softmax is applied inside
                # the GPU RNNT/TDT loss). Therefore we do not try to "fix" normalization here. The purpose of this
                # validator is to compare fused vs non-fused semantics, not to validate logits-vs-log_probs.
                fuse_flag = bool(getattr(self.joint, "_fuse_loss_wer", False))
                try:
                    self.joint._fuse_loss_wer = False
                    joint_out = self.joint(encoder_outputs=encoded, decoder_outputs=dec)
                finally:
                    self.joint._fuse_loss_wer = fuse_flag

                # Log a tiny logsumexp slice for debugging only (do not gate behavior on it).
                try:
                    t_sl = min(int(joint_out.shape[1]), 2)
                    u_sl = min(int(joint_out.shape[2]), 2)
                    metrics["gspo_logp_validate_joint_lse_mean"] = torch.logsumexp(
                        joint_out[:, :t_sl, :u_sl, :].to(torch.float32), dim=-1
                    ).mean()
                except Exception:
                    pass

                nll_ref = self._gspo_nll(
                    log_probs=joint_out, targets=targets, input_lengths=encoded_len, target_lengths=target_lens
                )

            nll_fused = nll_fused.reshape(-1).to(torch.float32)
            nll_ref = nll_ref.reshape(-1).to(torch.float32)

            abs_err = (nll_fused - nll_ref).abs()
            rel_err = abs_err / (nll_ref.abs() + 1e-8)

            metrics["gspo_logp_validate_skipped"] = torch.tensor(0.0, device=device, dtype=torch.float32)
            metrics["gspo_logp_validate_failed"] = torch.tensor(0.0, device=device, dtype=torch.float32)
            metrics["gspo_logp_validate_abs_err"] = abs_err.mean()
            metrics["gspo_logp_validate_rel_err"] = rel_err.mean()
            metrics["gspo_logp_validate_fused_nll"] = nll_fused.mean()
            metrics["gspo_logp_validate_ref_nll"] = nll_ref.mean()
            return metrics
        except Exception:
            metrics["gspo_logp_validate_skipped"] = torch.tensor(0.0, device=device, dtype=torch.float32)
            metrics["gspo_logp_validate_failed"] = torch.tensor(1.0, device=device, dtype=torch.float32)
            return metrics

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
        self,
        signal: torch.Tensor,
        signal_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        *,
        compute_diagnostics: bool,
        compute_timings: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        GSPO objective for a single sample.

        Notes:
        - This implementation intentionally recomputes `logp_old` and `logp_new` in separate passes even though
          they are often numerically identical in the current on-policy single-step regime. This redundancy is a
          useful "canary" to detect unintended stochasticity (e.g., dropout not fully disabled) and logp path drift.
        - `ppo_epochs>1` is not implemented yet (see module docstring TODO roadmap).
        """
        ppo_epochs = int(self.gspo_cfg.get("ppo_epochs", 1))
        if ppo_epochs != 1:
            raise NotImplementedError(
                "GSPO PPO-epochs style sample reuse is not implemented yet in the automatic-optimization path. "
                "See the TODO roadmap in `hybrid_rnnt_ctc_bpe_models_gspo.py`."
            )

        device = signal.device
        timings: Dict[str, float] = {}

        ref_token_ids = transcript[0, : int(transcript_len.item())].tolist()
        ref_text = self._tokens_to_text(ref_token_ids)

        # 1) Encode once.
        t0 = time.perf_counter()
        encoded, encoded_len = self._encode_one(signal, signal_len)
        if not self._train_encoder:
            encoded = encoded.detach()
        if compute_timings:
            if bool(self.gspo_cfg.get("timing_sync_cuda", False)) and signal.is_cuda:
                torch.cuda.synchronize(signal.device)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2) Rollout: beam decode n-best hypotheses from cached encoder outputs.
        t1 = time.perf_counter()
        with torch.no_grad():
            hyps_batch = self.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
        if compute_timings:
            if bool(self.gspo_cfg.get("timing_sync_cuda", False)) and signal.is_cuda:
                torch.cuda.synchronize(signal.device)
            timings["decode_ms"] = (time.perf_counter() - t1) * 1000.0

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

        # 3) Compute rewards and advantages (tiny tensors).
        hyp_token_ids_list: List[List[int]] = [self._hypothesis_to_token_ids(hyp) for hyp in hyps]
        empty_hyp_count = sum(1 for ids in hyp_token_ids_list if len(ids) == 0)

        # Decode hypothesis text from token ids so reward and logp always correspond to the same sequence.
        hyp_texts: List[str] = [self._tokens_to_text(ids) for ids in hyp_token_ids_list]
        group_size = len(hyp_texts)
        uniq_hyp_count = len(set(hyp_texts))

        rewards = torch.tensor(
            [self._reward_from_text(t, ref_text) for t in hyp_texts], device=device, dtype=torch.float32
        )
        if self.gspo_cfg.get("normalize_advantage", True):
            advantages = _group_normalize(rewards, eps=float(self.gspo_cfg.get("advantage_eps", 1e-8)))
        else:
            advantages = rewards - rewards.mean()
        advantages_detached = advantages.detach()

        # 4) Compute logp_old (no_grad baseline) and logp_new (with grad).
        use_batch = bool(self.gspo_cfg.get("batch_hypotheses", False))

        t2 = time.perf_counter()
        with torch.no_grad():
            if use_batch:
                logp_old = self._compute_logp_batch(
                    encoded, encoded_len, hyp_token_ids_list, compute_diagnostics=compute_diagnostics
                ).detach()
            else:
                logp_old_list: List[torch.Tensor] = []
                for token_ids in hyp_token_ids_list:
                    logp_old_list.append(self._compute_logp_from_encoder(encoded, encoded_len, token_ids).detach())
                logp_old = torch.stack(logp_old_list, dim=0)
        if compute_timings:
            if bool(self.gspo_cfg.get("timing_sync_cuda", False)) and signal.is_cuda:
                torch.cuda.synchronize(signal.device)
            timings["logp_old_ms"] = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        if use_batch:
            logp_new = self._compute_logp_batch(
                encoded, encoded_len, hyp_token_ids_list, compute_diagnostics=compute_diagnostics
            )
        else:
            logp_new_list: List[torch.Tensor] = []
            for token_ids in hyp_token_ids_list:
                logp_new_list.append(self._compute_logp_from_encoder(encoded, encoded_len, token_ids))
            logp_new = torch.stack(logp_new_list, dim=0)
        if compute_timings:
            if bool(self.gspo_cfg.get("timing_sync_cuda", False)) and signal.is_cuda:
                torch.cuda.synchronize(signal.device)
            timings["logp_new_ms"] = (time.perf_counter() - t3) * 1000.0

        logp_old_detached = logp_old.detach()
        logp_new_detached = logp_new.detach()

        rl_loss = gspo_clipped_loss_seq(
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=advantages_detached,
            clip_eps=float(self.gspo_cfg.clip_eps),
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

        metrics: Dict[str, torch.Tensor] = {}

        # Always log core scalars.
        metrics["gspo_rl_loss"] = rl_loss.detach()
        metrics["gspo_reward_mean"] = rewards.mean().detach()
        metrics["gspo_reward_std"] = rewards.std(unbiased=False).detach()
        metrics["gspo_group_size"] = torch.tensor(float(group_size), device=device, dtype=torch.float32)
        metrics["gspo_uniq_hyp_count"] = torch.tensor(float(uniq_hyp_count), device=device, dtype=torch.float32)
        metrics["gspo_uniq_hyp_frac"] = torch.tensor(
            float(uniq_hyp_count) / float(max(1, group_size)), device=device, dtype=torch.float32
        )
        metrics["gspo_dup_frac"] = torch.tensor(
            1.0 - (float(uniq_hyp_count) / float(max(1, group_size))), device=device, dtype=torch.float32
        )
        metrics["gspo_empty_hyp_count"] = torch.tensor(float(empty_hyp_count), device=device, dtype=torch.float32)
        metrics["gspo_empty_hyp_frac"] = torch.tensor(
            float(empty_hyp_count) / float(max(1, len(hyp_token_ids_list))), device=device, dtype=torch.float32
        )

        # NOTE: In the current "on-policy single-step" mode, `train_loss` / `gspo_rl_loss` can be close to 0 because
        # advantages are mean-zero (and often normalized). This does NOT imply gradients are zero.
        # These statistics are the primary "is it updating / is it buggy?" canary set.
        metrics["gspo_adv_mean"] = advantages_detached.mean()
        metrics["gspo_adv_std"] = advantages_detached.std(unbiased=False)

        metrics["gspo_logp_old_mean"] = logp_old_detached.mean()
        metrics["gspo_logp_old_std"] = logp_old_detached.std(unbiased=False)
        metrics["gspo_logp_new_mean"] = logp_new_detached.mean()
        metrics["gspo_logp_new_std"] = logp_new_detached.std(unbiased=False)

        logp_old_f = logp_old_detached.to(dtype=torch.float32)
        logp_new_f = logp_new_detached.to(dtype=torch.float32)
        delta_raw = logp_new_f - logp_old_f
        delta_clamp_min, delta_clamp_max = -80.0, 80.0
        clamp_hit = ((delta_raw < delta_clamp_min) | (delta_raw > delta_clamp_max)).to(torch.float32).mean()
        log_ratio = delta_raw.clamp(min=delta_clamp_min, max=delta_clamp_max)
        ratio = torch.exp(log_ratio)
        clip_eps = float(self.gspo_cfg.clip_eps)
        metrics["gspo_ratio_mean"] = ratio.mean()
        metrics["gspo_ratio_std"] = ratio.std(unbiased=False)
        metrics["gspo_ratio_max"] = ratio.max()
        metrics["gspo_clip_frac"] = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()
        metrics["gspo_approx_kl"] = (logp_old_f - logp_new_f).mean()
        metrics["gspo_logp_span"] = (logp_old_f.max() - logp_old_f.min())
        metrics["gspo_best_reward_minus_mean"] = (rewards.max() - rewards.mean()).detach()
        metrics["gspo_log_ratio_raw_max"] = delta_raw.max().detach()
        metrics["gspo_log_ratio_raw_min"] = delta_raw.min().detach()
        metrics["gspo_log_ratio_clamp_hit_frac"] = clamp_hit.detach()

        if compute_diagnostics:
            metrics.update(self._maybe_validate_fused_logp(encoded=encoded, encoded_len=encoded_len, token_ids=ref_token_ids))

            # N-best group diagnostics.
            hyp_lens = torch.tensor([len(ids) for ids in hyp_token_ids_list], device=device, dtype=torch.float32)
            metrics["gspo_hyp_len_mean"] = hyp_lens.mean()
            metrics["gspo_hyp_len_std"] = hyp_lens.std(unbiased=False)
            metrics["gspo_hyp_len_min"] = hyp_lens.min()
            metrics["gspo_hyp_len_max"] = hyp_lens.max()

            metrics["gspo_corr_reward_hyp_len"] = _safe_corrcoef(rewards, hyp_lens)
            metrics["gspo_corr_adv_hyp_len"] = _safe_corrcoef(advantages_detached, hyp_lens)
            metrics["gspo_corr_reward_logp_old"] = _safe_corrcoef(rewards, logp_old_detached)

            if compute_timings:
                for name, value in timings.items():
                    metrics[f"gspo_time_{name}"] = torch.tensor(value, device=device, dtype=torch.float32)

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
        metrics_acc: Dict[str, List[torch.Tensor]] = {}

        log_diagnostics = bool(self.gspo_cfg.get("log_diagnostics", True))
        diag_every = int(self.gspo_cfg.get("log_diagnostics_every_n_steps", 50))
        log_text_every = int(self.gspo_cfg.get("log_text_every_n_steps", 0))

        do_log = log_diagnostics and (diag_every <= 1 or (self.global_step % diag_every == 0))
        do_print = log_text_every > 0 and (self.global_step % log_text_every == 0)

        wants_validation = bool(self.gspo_cfg.get("validate_fused_logp_once", False)) and not bool(
            getattr(self, "_gspo_fused_logp_validation_ran", False)
        )
        if wants_validation:
            do_log = True

        # ---- Diagnostics gating under grad accumulation ----
        # Under grad accumulation, `training_step()` can be called multiple times with the same `global_step`.
        # We want expensive diagnostics/printing/validation to run at most once per optimizer step.
        if bool(self.gspo_cfg.get("diagnostics_once_per_global_step", True)):
            if (do_log or do_print or wants_validation) and (int(self.global_step) == self._gspo_last_diagnostics_global_step):
                do_log = False
                do_print = False
                wants_validation = False
            elif do_log or do_print or wants_validation:
                self._gspo_last_diagnostics_global_step = int(self.global_step)

        compute_diagnostics = do_log or do_print or wants_validation

        compute_timings = compute_diagnostics and bool(self.gspo_cfg.get("log_timings", False))

        for i in range(batch_size):
            loss_i, metrics_i = self._gspo_one_sample(
                signal[i : i + 1],
                signal_len[i : i + 1],
                transcript[i : i + 1],
                transcript_len[i : i + 1],
                compute_diagnostics=compute_diagnostics,
                compute_timings=compute_timings,
            )
            losses.append(loss_i)
            for key, value in metrics_i.items():
                metrics_acc.setdefault(key, []).append(value)

        loss = torch.stack(losses).mean()

        # Logging.
        sync_core = bool(self.gspo_cfg.get("sync_dist_core_metrics", True))
        sync_extra = bool(self.gspo_cfg.get("sync_dist_extra_metrics", True))

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # Always log core "is it updating / is it buggy?" metrics so users don't over-interpret `train_loss≈0`.
        always_log_keys = {
            "gspo_rl_loss",
            "gspo_reward_mean",
            "gspo_reward_std",
            "gspo_best_reward_minus_mean",
            "gspo_group_size",
            "gspo_uniq_hyp_count",
            "gspo_uniq_hyp_frac",
            "gspo_dup_frac",
            "gspo_empty_hyp_count",
            "gspo_empty_hyp_frac",
            "gspo_adv_mean",
            "gspo_adv_std",
            "gspo_logp_old_mean",
            "gspo_logp_old_std",
            "gspo_logp_new_mean",
            "gspo_logp_new_std",
            "gspo_logp_span",
            "gspo_ratio_mean",
            "gspo_ratio_std",
            "gspo_ratio_max",
            "gspo_clip_frac",
            "gspo_approx_kl",
            "gspo_log_ratio_raw_max",
            "gspo_log_ratio_raw_min",
            "gspo_log_ratio_clamp_hit_frac",
        }
        for key in sorted(always_log_keys):
            if key in metrics_acc:
                self.log(f"train_{key}", torch.stack(metrics_acc[key]).mean(), prog_bar=False, sync_dist=sync_core)

        if do_log:
            for key, values in metrics_acc.items():
                if key in always_log_keys:
                    continue
                # Metrics are per-sample; aggregate across batch to one scalar.
                self.log(f"train_{key}", torch.stack(values).mean(), prog_bar=False, sync_dist=sync_extra)

        # Optional: compact text logs for quick debugging (global rank 0 only).
        if do_print and getattr(self.trainer, "is_global_zero", True):
            msg = (
                f"GSPO step={int(self.global_step)} "
                f"loss={loss.detach().float().item():.4f} "
                f"reward_mean={torch.stack(metrics_acc['gspo_reward_mean']).mean().item():.4f} "
                f"reward_std={torch.stack(metrics_acc['gspo_reward_std']).mean().item():.4f}"
            )
            if "gspo_ratio_mean" in metrics_acc:
                msg += (
                    f" ratio_mean={torch.stack(metrics_acc['gspo_ratio_mean']).mean().item():.4f} "
                    f"clip_frac={torch.stack(metrics_acc['gspo_clip_frac']).mean().item():.4f} "
                    f"approx_kl={torch.stack(metrics_acc['gspo_approx_kl']).mean().item():.4f}"
                )
            if "gspo_dup_frac" in metrics_acc:
                msg += f" dup_frac={torch.stack(metrics_acc['gspo_dup_frac']).mean().item():.4f}"
            logging.info(msg)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if not bool(self.gspo_cfg.get("log_grad_norm", False)):
            return

        every = int(self.gspo_cfg.get("grad_norm_every_n_steps", 50))
        if every > 1 and (self.global_step % every != 0):
            return

        def _module_grad_norm(module: torch.nn.Module) -> torch.Tensor:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads:
                return torch.zeros((), device=next(self.parameters()).device, dtype=torch.float32)
            return torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads))

        sync_extra = bool(self.gspo_cfg.get("sync_dist_extra_metrics", True))
        self.log("train_gspo_grad_norm_decoder", _module_grad_norm(self.decoder), prog_bar=False, sync_dist=sync_extra)
        self.log("train_gspo_grad_norm_joint", _module_grad_norm(self.joint), prog_bar=False, sync_dist=sync_extra)
        if self._train_encoder:
            self.log("train_gspo_grad_norm_encoder", _module_grad_norm(self.encoder), prog_bar=False, sync_dist=sync_extra)
