# Quick Reference: Length-Budget Batch Sampler

## Basic Example

Add to your `train_ds` configuration:

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/train.json
    
    # Enable length-budget batching
    length_budget: 300.0        # Required: max cost (batch_size × max_duration)
    max_batch_size: 32          # Optional: hard limit on batch size
    shuffle: true               # Recommended for training
    drop_last: false            # Use all data
    balance_across_ranks: true  # Balance GPUs (distributed training)
    
    # Standard settings
    num_workers: 8
    pin_memory: true
```

## What Does This Do?

Instead of fixed batch sizes, the sampler creates batches where:
- **Short utterances** → larger batches (e.g., 30 samples × 10 seconds = 300)
- **Long utterances** → smaller batches (e.g., 10 samples × 30 seconds = 300)
- **Mixed lengths** → dynamically sized batches keeping cost ≤ `length_budget`

## Complete Minimal Config

```yaml
name: "ASR-with-LengthBudget"

model:
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    sample_rate: 16000
    
    # Length-budget sampler - just add these 3 lines!
    length_budget: 300.0
    max_batch_size: 32
    shuffle: true
    
    num_workers: 8
    pin_memory: true
    max_duration: 20.0
    min_duration: 0.1
    
  validation_ds:
    manifest_filepath: /path/to/val_manifest.json
    batch_size: 16  # Use fixed batch for validation
    shuffle: false
    num_workers: 8
    
  # ... rest of model config ...
  
trainer:
  devices: 4        # Multi-GPU supported automatically
  strategy: ddp
  # ... rest of trainer config ...
```

## Quick Tuning Guide

| Your GPU | Your Precision | Set `length_budget` | Set `max_batch_size` |
|----------|----------------|---------------------|----------------------|
| 16GB     | fp32           | 150                 | 16                   |
| 16GB     | fp16/bf16      | 250                 | 24                   |
| 32GB     | fp32           | 300                 | 24                   |
| 32GB     | fp16/bf16      | 450                 | 32                   |
| 80GB     | fp32           | 600                 | 48                   |
| 80GB     | fp16/bf16      | 900                 | 64                   |

**Got OOM error?** → Decrease both values by 30%
**GPU underutilized?** → Increase both values by 30%

## Important Notes

✅ **Works with:**
- Map-style datasets (AudioToCharDataset, AudioToBPEDataset)
- Distributed training (automatically balanced)
- Standard data augmentation

❌ **Does NOT work with:**
- Tarred datasets (`is_tarred: true`)
- Semi-sorted batching (`use_semi_sorted_batching: true`)
- Iterable datasets

⚠️ **Remember:**
- The `batch_size` parameter is **ignored** when `length_budget` is set
- Each batch can have a different size (that's the point!)
- Set `shuffle: true` for training, `false` for validation

## See More

- `conformer_ctc_length_budget.yaml` - Full CTC example
- `conformer_transducer_length_budget.yaml` - Full Transducer example  
- `USAGE.md` - Detailed usage guide with all scenarios
- `README.md` - Technical overview
