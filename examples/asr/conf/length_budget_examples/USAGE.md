# Length-Budget Batch Sampler Usage Guide

## Overview

The length-budget batch sampler enables dynamic batching for ASR training where batch sizes automatically adjust based on audio duration to maintain relatively constant memory usage.

## Quick Start

### Minimal Configuration

To enable length-budget sampling, add these lines to your train_ds configuration:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    length_budget: 300.0  # Enable length-budget sampler
    max_batch_size: 32    # Optional: limit maximum batch size
    shuffle: true         # Recommended for training
```

### Common Scenarios

#### Scenario 1: Training with Variable-Length Audio (Default Settings)

Best for general ASR training with mixed-length utterances:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    sample_rate: 16000
    
    # Length-budget configuration
    length_budget: 300.0      # Adjust based on GPU memory
    max_batch_size: 32        # Prevent extremely large batches
    shuffle: true             # Shuffle for training diversity
    drop_last: false          # Use all training data
    balance_across_ranks: true  # Balance across GPUs
    length_budget_seed: 42    # For reproducibility
    
    # Other settings
    num_workers: 8
    pin_memory: true
    max_duration: 20.0
    min_duration: 0.1
```

#### Scenario 2: Memory-Constrained Training (16GB GPU)

For smaller GPUs, use a lower budget:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    length_budget: 150.0      # Lower budget for less memory
    max_batch_size: 16        # Smaller max batch size
    shuffle: true
    drop_last: false
```

#### Scenario 3: Large GPU Memory (80GB A100)

Take advantage of more memory:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    length_budget: 600.0      # Higher budget for more memory
    max_batch_size: 64        # Larger max batch size
    shuffle: true
    drop_last: false
```

#### Scenario 4: Multi-GPU Distributed Training

The sampler automatically handles distributed training:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    length_budget: 300.0
    max_batch_size: 32
    shuffle: true
    balance_across_ranks: true  # Balances workload across GPUs (recommended)
    drop_last: false

trainer:
  devices: 8              # Number of GPUs
  num_nodes: 1           # Number of nodes
  strategy: ddp          # Distributed strategy
```

#### Scenario 5: Validation/Test with Length-Budget

While typically validation uses fixed batch sizes, you can also use length-budget:

```yaml
model:
  validation_ds:
    manifest_filepath: /path/to/val_manifest.json
    length_budget: 400.0      # Larger budget ok for validation
    shuffle: false            # Don't shuffle validation
    drop_last: false          # Evaluate on all samples
    num_workers: 8
```

Or use traditional fixed batching for validation:

```yaml
model:
  validation_ds:
    manifest_filepath: /path/to/val_manifest.json
    batch_size: 16      # Fixed batch size
    shuffle: false
    num_workers: 8
```

## Parameter Reference

### Required Parameters

- **`length_budget`**: (float) Maximum cost per batch. Cost = batch_size × max_duration_in_batch
  - Example: `300.0` means a batch of 10 samples with max 30s duration, or 20 samples with max 15s duration

### Optional Parameters

- **`max_batch_size`**: (int, optional) Hard limit on batch size
  - Prevents very large batches when all utterances are short
  - Example: `32` limits batch to maximum 32 samples regardless of their length

- **`shuffle`**: (bool, default: false) Shuffle samples before batching
  - Set to `true` for training to increase diversity
  - Set to `false` for validation/testing for reproducibility

- **`drop_last`**: (bool, default: false) Drop the last incomplete batch
  - Usually `false` for training to use all data
  - May be `true` in distributed settings to ensure equal steps across ranks

- **`balance_across_ranks`**: (bool, default: true) Balance workload in distributed training
  - `true`: Sorts batches by cost and distributes them to balance GPU utilization
  - `false`: Simple round-robin distribution
  - Recommended: `true` for better GPU efficiency

- **`length_budget_seed`** or **`seed`**: (int, default: 0) Random seed for shuffling
  - Used when `shuffle=true` for reproducible batch order
  - Changes each epoch automatically

### Standard DataLoader Parameters

These work normally with length-budget sampler:

```yaml
num_workers: 8          # Number of data loading workers
pin_memory: true        # Pin memory for faster GPU transfer
max_duration: 20.0      # Filter samples longer than this (applied before batching)
min_duration: 0.1       # Filter samples shorter than this (applied before batching)
```

## Choosing the Right `length_budget` Value

The optimal `length_budget` depends on:
1. GPU memory available
2. Model size
3. Precision (fp32, fp16, bf16)
4. Audio feature dimensions

### Rule of Thumb

Start with these values and adjust based on OOM errors:

| GPU Memory | Precision | Suggested Budget | Max Batch Size |
|------------|-----------|------------------|----------------|
| 16GB       | fp32      | 100-150          | 16             |
| 16GB       | fp16/bf16 | 150-250          | 24             |
| 32GB       | fp32      | 200-300          | 24             |
| 32GB       | fp16/bf16 | 300-450          | 32             |
| 80GB       | fp32      | 400-600          | 48             |
| 80GB       | fp16/bf16 | 600-900          | 64             |

Adjust based on:
- **OOM (Out of Memory) errors**: Decrease `length_budget` and `max_batch_size`
- **Low GPU utilization**: Increase `length_budget` and `max_batch_size`

## Important Limitations

1. **Cannot combine with semi-sorted batching**:
   ```yaml
   # This will cause an error:
   train_ds:
     length_budget: 300.0
     use_semi_sorted_batching: true  # ERROR: mutually exclusive
   ```

2. **Only for map-style datasets**:
   ```yaml
   # Length-budget does NOT work with tarred datasets:
   train_ds:
     length_budget: 300.0
     is_tarred: true  # ERROR: not supported
   ```

3. **`batch_size` parameter is ignored**:
   ```yaml
   # When using length-budget, batch_size is ignored:
   train_ds:
     length_budget: 300.0
     batch_size: 16  # This value is ignored (but can be kept for reference)
   ```

## Advanced Usage

### Combining with Duration Filtering

Pre-filter samples by duration to avoid very long/short utterances:

```yaml
model:
  train_ds:
    # Filter first
    max_duration: 20.0  # Remove samples > 20s
    min_duration: 0.5   # Remove samples < 0.5s
    
    # Then apply length-budget batching
    length_budget: 300.0
    max_batch_size: 32
```

### Different Budgets for Train/Val

Use aggressive batching for training, conservative for validation:

```yaml
model:
  train_ds:
    length_budget: 300.0
    shuffle: true
    
  validation_ds:
    # Option 1: Use larger budget for faster validation
    length_budget: 500.0
    shuffle: false
    
    # Option 2: Use fixed batch size for consistent validation
    # batch_size: 16  # Comment out length_budget to use this
```

### Reproducible Training

For reproducible batch ordering across runs:

```yaml
model:
  train_ds:
    length_budget: 300.0
    shuffle: true
    length_budget_seed: 12345  # Fixed seed for reproducibility
```

## Troubleshooting

### Problem: Out of Memory Errors

**Solution**: Decrease `length_budget` or `max_batch_size`:

```yaml
train_ds:
  length_budget: 200.0  # Reduce from 300.0
  max_batch_size: 16    # Reduce from 32
```

### Problem: Training is Slower Than Expected

**Solution**: The batches might be too small. Increase `length_budget`:

```yaml
train_ds:
  length_budget: 400.0  # Increase from 300.0
```

### Problem: All Batches Being Dropped (Error)

This happens when `drop_last=true` in distributed training with very few batches.

**Solution**: Set `drop_last=false` or increase dataset/budget:

```yaml
train_ds:
  length_budget: 300.0
  drop_last: false  # Don't drop batches
```

### Problem: Unbalanced GPU Usage in Multi-GPU

**Solution**: Enable `balance_across_ranks`:

```yaml
train_ds:
  length_budget: 300.0
  balance_across_ranks: true  # Balance workload
```

## Complete Example Command

```bash
python examples/asr/asr_ctc/speech_to_text_ctc.py \
  --config-path=conf/length_budget_examples \
  --config-name=conformer_ctc_length_budget \
  model.train_ds.manifest_filepath=/path/to/train.json \
  model.validation_ds.manifest_filepath=/path/to/dev.json \
  model.tokenizer.dir=/path/to/tokenizer \
  trainer.devices=4 \
  trainer.max_epochs=100
```

## Migration from Fixed Batch Size

If you have an existing config with fixed `batch_size`:

**Before (fixed batching)**:
```yaml
train_ds:
  batch_size: 16
  shuffle: true
  drop_last: false
```

**After (length-budget batching)**:
```yaml
train_ds:
  length_budget: 300.0     # Add this
  max_batch_size: 32       # Add this (optional)
  batch_size: 16           # Ignored, but can keep for reference
  shuffle: true            # Keep as is
  drop_last: false         # Keep as is
```
