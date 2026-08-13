# GPU training pipeline optimization

## Bottleneck

The v3 grid is 15.1GB in float16 and fits in system RAM. The original `Dataset.__getitem__` pipeline performed three mmap accesses, three float32 conversions, normalization and tensor copies for every individual sample. GPU utilization was commonly low because input preparation was serialized in Python.

## New path

`train_multistream_grid.py` supports `RAM_BATCHED=1`:

1. copy the contiguous float16 grid from mmap into RAM once;
2. shuffle only integer row indices;
3. gather full NumPy batches in a background prefetch thread;
4. transfer float16 batches to GPU;
5. normalize, clip, clean and transpose on GPU;
6. train with physical batch256 and four-step accumulation.

Additional settings:

- `torch.backends.cudnn.benchmark=True`;
- TF32 for CUDA matmul and cuDNN;
- fused AdamW when available;
- correct accumulation scaling and final partial-step handling;
- effective batch logging;
- full-data mode and resumable train state.

## Benchmark

End-to-end model forward + backward + optimizer, v3 400/120, d_model64:

| Path | Physical batch | Samples/s |
|---|---:|---:|
| Per-sample mmap | 128 | ~1,700 |
| RAM/GPU batch | 128 | ~2,710–2,860 |
| RAM/GPU batch | 256 | **~4,036** |
| RAM/GPU batch | 384 | ~4,041 |
| RAM/GPU batch | 512 | ~3,991 |

Batch256 is selected because it reaches the throughput plateau with lower memory risk. With accumulation4, the effective batch remains 1024.
