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
GSPO post-training for FastConformer-Hybrid-TDT-CTC-BPE (memory-first, slow OK).

This script runs a sequence-level PPO-style objective (GSPO) using:
- n-best rollouts from NeMo's built-in RNNT/TDT beam search
- sequence logp(y|x) from the transducer loss (TDT) as a proxy for log-likelihood
- WER/CER reward computed from decoded text

Config:
- `examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe_gspo.yaml`

References:
- PPO (clipped objective): https://arxiv.org/abs/1707.06347
- TDT (Token-and-Duration Transducer): https://arxiv.org/abs/2304.06795
- GSPO / sequence-level ratios discussion: https://arxiv.org/pdf/2507.18071

TODO:
- Add LM reward shaping / reranking (KenLM) for length bias mitigation.
- Add a frozen reference model for KL regularization (optional).
"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModelGSPO
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(
    config_path="../conf/fastconformer/hybrid_transducer_ctc/",
    config_name="fastconformer_hybrid_tdt_ctc_bpe_gspo.yaml",
)
def main(cfg):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecHybridRNNTCTCBPEModelGSPO(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter

