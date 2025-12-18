# Length-Budget Batch Sampler Examples

This directory contains example YAML configurations showing how to use the length-budget batch sampler for ASR training.

## What is Length-Budget Batch Sampler?

The length-budget batch sampler is a dynamic batching strategy that groups audio samples based on a computational budget defined as:

```
cost = batch_size × max_sample_length_in_batch
```

This allows more efficient GPU utilization by:
- Creating larger batches for short utterances
- Creating smaller batches for long utterances
- Keeping memory usage relatively constant across batches

## Key Configuration Parameters

- **`length_budget`** (required): Maximum cost allowed per batch (e.g., 300.0 means max batch size × max duration ≤ 300 seconds)
- **`max_batch_size`** (optional): Hard limit on batch size regardless of length budget
- **`shuffle`** (default: false): Whether to shuffle samples before batching
- **`drop_last`** (default: false): Whether to drop the last incomplete batch
- **`balance_across_ranks`** (default: true): For distributed training, balance workload across GPUs
- **`length_budget_seed`** or **`seed`** (default: 0): Random seed for shuffling

## Important Notes

1. **Cannot be combined with semi-sorted batching**: Length-budget and semi-sorted batching are mutually exclusive
2. **Only for map-style datasets**: Does not work with iterable datasets or tarred datasets
3. **Batch size is ignored**: When using length-budget sampler, the regular `batch_size` parameter is ignored
4. **Distributed training**: The sampler automatically handles multi-GPU training and ensures all ranks have the same number of steps

## Examples

See the example configuration files in this directory:
- `conformer_ctc_length_budget.yaml` - CTC model with length-budget sampler
- `conformer_transducer_length_budget.yaml` - Transducer model with length-budget sampler
