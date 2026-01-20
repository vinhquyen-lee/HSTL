# HSTL: Hierarchical Spatio-Temporal Representation Learning for Gait Recognition

Implementation and improvements of the ICCV 2023 paper, adapted for single-GPU Kaggle training.

## Project Overview

This codebase implements HSTL (Hierarchical Spatio-Temporal Representation Learning) for gait recognition on the CASIA-B dataset. The project has been adapted from the original multi-GPU distributed training setup to run efficiently on Kaggle's single GPU environment (P100/T4).

**Original Paper:** Wang et al., "Hierarchical Spatio-Temporal Representation Learning for Gait Recognition", ICCV 2023

**Key Adaptations:**
- Single-GPU compatible (no distributed training required)
- Kaggle-optimized paths and environment detection
- Multiple backbone variants (BasicConv3d, P3D)
- Multiple loss functions (TripletLoss, CircleLoss)
- Batch testing tool for evaluating multiple checkpoints

---

## System Requirements

### Hardware
- GPU: NVIDIA P100 (16GB) or T4 (16GB) recommended
- RAM: 16GB+ for dataset preprocessing
- Storage: 20GB+ for dataset and checkpoints

### Software
- Python 3.11+
- PyTorch 1.10+
- CUDA 11.0+

### Dependencies
```
torch>=1.10
torchvision
pytorch-metric-learning
pyyaml
opencv-python
tqdm
numpy
```

---

## Dataset Preparation

### CASIA-B Dataset

**Structure:**
- 124 subjects (IDs: 001-124)
- 3 walking conditions: NM (normal), BG (bag), CL (coat)
- 11 view angles: 0, 18, 36, ..., 180 degrees

**Standard Split (Large-sample Training):**
- Training: 74 subjects (001-074)
- Testing: 50 subjects (075-124)

### Preprocessing Raw Dataset

Convert PNG silhouettes to preprocessed pickle files:

```bash
python datasets/pretreatment.py \
  --input_path "D:/CASIA-B-raw" \
  --output_path "D:/CASIA-B-pkl" \
  -n 1 \
  -r 64 \
  -d CASIAB
```

**Parameters:**
- `-i, --input_path`: Raw dataset directory
- `-o, --output_path`: Output directory for pickle files
- `-n, --n_workers`: Number of workers (use 1 for low-memory systems)
- `-r, --img_size`: Output image size (default: 64)
- `-d, --dataset`: Dataset name (default: CASIAB)

**Partial Processing (for missing subjects):**
```bash
python datasets/pretreatment.py \
  -i "D:/CASIA-B-raw" \
  -o "D:/CASIA-B-pkl" \
  -n 1 \
  --start_id 092 \
  --end_id 124
```

**Expected Output:**
- Processing time: ~30-40 minutes (9,868 sequences)
- Output size: ~15-20 GB
- Format: `{subject}/{condition}/{view}/{view}.pkl`
- Each pickle contains: numpy array [N, 64, 64]

**Common Warnings:**
- "has no data": Corrupt frames (automatically skipped)
- "less than 5 valid data": Very short sequences (still saved)
- Subject 005: Known to have ~130 corrupt frames (normal)

---

## Model Variants

This project implements 4 model variants:

| Variant | Backbone | Loss Function | save_name | Description |
|---------|----------|---------------|-----------|-------------|
| **HSTL** | BasicConv3d | TripletLoss | HSTL | Original architecture |
| **kaggle** | BasicConv3d | TripletLoss | kaggle | Original + optimizations |
| **p3d** | P3D (2+1D) | TripletLoss | p3d | P3D backbone |
| **med-circleloss** | BasicConv3d | CircleLoss | med-circleloss | CircleLoss variant |
| **p3d-circleloss** | P3D (2+1D) | CircleLoss | p3d-circleloss | P3D + CircleLoss |

**Backbone Types:**
- `conv3d`: Standard 3D convolution (BasicConv3d)
- `p3d`: Pseudo-3D (spatial conv 1x3x3 + temporal conv 3x1x1)

**Loss Functions:**
- `triplet`: Triplet loss with margin 0.2
- `circle`: Circle loss with m=0.25, gamma=128

---

## Configuration

### Main Config File: `config/hstl.yaml`

**Key Sections:**

```yaml
# Model architecture control
model_cfg:
  model: HSTL
  channels: [32, 64, 128]
  class_num: 74
  backbone_type: p3d      # conv3d | p3d
  loss_type: circle       # triplet | circle

# Loss configuration (must match loss_type)
loss_cfg:
  - type: CircleLoss      # TripletLoss | CircleLoss
    log_prefix: circle    # triplet | circle
    # ... loss parameters

# Training control
trainer_cfg:
  save_name: p3d-circleloss
  restore_hint: 0         # 0 = start from scratch, N = resume from iteration N
  optimizer_reset: false  # false = load optimizer state when resuming
  scheduler_reset: false  # false = load scheduler state when resuming
  save_iter: 2500         # Save checkpoint every N iterations
  total_iter: 100000      # Total training iterations
  enable_float16: true    # FP16 mixed precision
  with_test: true         # Test at every checkpoint save
```

**Important:** When switching variants, ensure these match:
- `model_cfg.loss_type` matches `loss_cfg[0].log_prefix`
- `trainer_cfg.save_name` describes the variant
- `evaluator_cfg.save_name` matches `trainer_cfg.save_name`

### Partition Files

Located in `misc/partitions/`:
- `CASIA-B.json`: Excludes subject 005 (73 training subjects)
- `CASIA-B_include_005.json`: Includes subject 005 (74 training subjects)

Use `include_005` for standard benchmarks.

---

## Local Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n hstl python=3.11
conda activate hstl

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-metric-learning pyyaml opencv-python tqdm
```

### 2. Training Locally

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python lib/main.py \
  --cfgs config/hstl.yaml \
  --phase train \
  --log_to_file
```

**Resume from checkpoint:**
```bash
# Set in config/hstl.yaml:
trainer_cfg:
  restore_hint: 50000
  optimizer_reset: false
  scheduler_reset: false
```

### 3. Testing Locally

**Single checkpoint:**
```bash
python lib/main.py \
  --cfgs config/hstl.yaml \
  --phase test \
  --iter 80000
```

**Batch testing (multiple checkpoints):**
```bash
# 1. Configure models in batch_test_config.yaml
# 2. Run batch test
python batch_test.py
```

---

## Kaggle Deployment

### Initial Setup

**1. Install Kaggle API:**
```bash
pip install kaggle
```

**2. Authentication:**
- Visit Kaggle -> Settings -> Create New API Token
- Move `kaggle.json` to:
  - Linux/Mac: `~/.kaggle/`
  - Windows: `C:\Users\<username>\.kaggle\`

**3. Verify:**
```bash
kaggle competitions list
```

### Dataset Upload

**Initialize dataset:**
```bash
# Edit dataset-metadata.json with your Kaggle username
kaggle datasets create -p . --dir-mode zip
```

**Update dataset:**
```bash
kaggle datasets version -p . --dir-mode zip -m "Updated dataset"
```

**Required Kaggle datasets:**
- Dataset 1: `casiab-complete` (preprocessed CASIA-B with 124 subjects)
- Dataset 2: `project-hstl` (source code, partition files, checkpoints)

### Kernel Setup

**Initialize kernel:**
```bash
# Edit kernel-metadata.json
kaggle kernels init -p .
```

**Example `kernel-metadata.json`:**
```json
{
  "id": "yourusername/hstl-training",
  "title": "HSTL Training",
  "code_file": "kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "yourusername/casiab-complete",
    "yourusername/project-hstl"
  ]
}
```

**Push to Kaggle:**
```bash
kaggle kernels push -p .
```

**Check status:**
```bash
kaggle kernels status yourusername/hstl-training
```

**Download output:**
```bash
kaggle kernels output yourusername/hstl-training -p ./kaggle_output
```

### Training on Kaggle

**Method 1: Direct training (`kaggle.py`)**

```python
# kaggle.ipynb
%cd /kaggle/input/project-hstl
!pip install -q pytorch-metric-learning
!python kaggle.py
```

**Method 2: Batch testing (`batch_test_kaggle.py`)**

```python
# kaggle_test.ipynb
%cd /kaggle/input/project-hstl
!pip install -q pytorch-metric-learning pyyaml
!python batch_test_kaggle.py
```

### Kaggle Environment

**Paths:**
- Dataset input: `/kaggle/input/<dataset-name>/`
- Working directory: `/kaggle/working/`
- Output: `/kaggle/working/output/`

**Automatic detection:**
Code automatically detects Kaggle environment via:
```python
if os.path.exists('/kaggle/input'):
    # Use Kaggle paths
    # Disable TensorBoard (protobuf conflicts)
```

**GPU Types:**
- P100: 16GB VRAM (recommended batch_size: [8, 8])
- T4: 16GB VRAM (recommended batch_size: [8, 8])

**Time Limits:**
- Free tier: 12 hours per session
- Estimated iterations: 20,000-25,000 per session

---

## Training Workflow

### Single Training Session

**1. Configure variant** in `config/hstl.yaml`:
```yaml
model_cfg:
  backbone_type: p3d
  loss_type: circle

trainer_cfg:
  save_name: p3d-circleloss
  restore_hint: 0
  total_iter: 100000
  save_iter: 2500

loss_cfg:
  - type: CircleLoss
    log_prefix: circle
```

**2. Run training:**
```bash
python lib/main.py --cfgs config/hstl.yaml --phase train --log_to_file
```

**3. Monitor progress:**
- Logs: `output/{dataset}/{model}/{save_name}/logs/{timestamp}.txt`
- Checkpoints: `output/{dataset}/{model}/{save_name}/checkpoints/{save_name}-{iter:05d}.pt`

### Resume Training

**Local:**
```yaml
trainer_cfg:
  restore_hint: 50000
  optimizer_reset: false
  scheduler_reset: false
```

**Kaggle multi-session:**
1. Download checkpoint from previous session
2. Upload to project-hstl dataset
3. Update dataset version
4. Set `restore_hint` in config
5. Run new session

---

## Testing and Evaluation

### Single Checkpoint Test

```bash
python lib/main.py \
  --cfgs config/hstl.yaml \
  --phase test \
  --iter 80000
```

### Batch Testing (Multiple Checkpoints)

**1. Configure `batch_test_config.yaml`:**
```yaml
dataset_root: D:/biometrics/gait-reg/dataset/CASIA-B-pkl
dataset_partition: ./misc/partitions/CASIA-B_include_005.json

models:
  p3d-circleloss:
    iterations: [2500, 5000, 7500, ..., 100000]
    save_name: p3d-circleloss
    checkpoint_path: ./output/CASIA-B/HSTL/p3d-circleloss/checkpoints
    backbone_type: p3d
    loss_type: circle
```

**2. Run batch test:**
```bash
# Local
python batch_test.py

# Kaggle
python batch_test_kaggle.py
```

**3. View results:**
- Summary table: `batch_test_results/summary.txt`
- Detailed results: `batch_test_results/detailed_results.txt`
- Individual logs: `batch_test_results/logs/{model}/{model}_{iter}.log`

### Result Format

```
===Rank-1 (Include identical-view cases)===
NM: 98.5,	BG: 92.9,	CL: 93.7

===Rank-1 (Exclude identical-view cases)===
NM: 98.4,	BG: 92.5,	CL: 93.0

===Rank-1 of each angle (Exclude identical-view cases)===
NM: [ 96.25  99.69 100.00  98.75  97.50  97.19 ...]
BG: [90.91 94.55 93.03 91.56 92.12 92.73 ...]
CL: [87.50 93.75 99.06 97.81 92.50 92.81 ...]
```

**Metrics:**
- NM: Normal walking accuracy
- BG: Carrying bag accuracy
- CL: Wearing coat accuracy
- Include: Same-view probe-gallery pairs included
- Exclude: Same-view pairs excluded (standard evaluation)

---

## File Structure

```
HSTL/
├── config/
│   ├── hstl.yaml              # Main configuration file
│   ├── default.yaml           # Default parameters
│   └── env-gpu.yaml           # Conda environment
├── lib/
│   ├── main.py                # Entry point
│   ├── data/
│   │   ├── dataset.py         # CASIA-B dataset loader
│   │   ├── sampler.py         # TripletSampler, InferenceSampler
│   │   ├── collate_fn.py      # Batch collation
│   │   └── transform.py       # Data transforms
│   ├── modeling/
│   │   ├── base_model.py      # Base training/testing framework
│   │   ├── models/
│   │   │   └── HSTL-CB.py     # HSTL model for CASIA-B
│   │   ├── modules.py         # Building blocks (ARME, FTA, ASTP)
│   │   ├── losses/
│   │   │   ├── triplet.py     # Triplet loss
│   │   │   ├── circle.py      # Circle loss
│   │   │   └── softmax.py     # Cross-entropy loss
│   │   └── loss_aggregator.py # Multi-loss combination
│   └── utils/
│       ├── common.py          # Utility functions
│       ├── msg_manager.py     # Logging system
│       └── evaluation.py      # Accuracy metrics
├── datasets/
│   └── pretreatment.py        # Dataset preprocessing
├── misc/
│   └── partitions/
│       └── CASIA-B_include_005.json  # Train/test split
├── batch_test.py              # Local batch testing
├── batch_test_kaggle.py       # Kaggle batch testing
├── batch_test_config.yaml     # Batch test configuration
├── kaggle.py                  # Kaggle training script
├── kaggle.ipynb               # Kaggle training notebook
└── kaggle_test.ipynb          # Kaggle testing notebook
```

---

## Checkpoint Format

**Naming:** `{save_name}-{iteration:05d}.pt`

**Examples:**
- `HSTL-80000.pt`
- `p3d-circleloss-37500.pt`
- `kaggle-100000.pt`

**Contents:**
```python
{
    'model': model.state_dict(),      # Model weights
    'optimizer': optimizer.state_dict(),  # Optimizer state
    'scheduler': scheduler.state_dict(),  # LR scheduler state
    'iteration': iteration_number         # Current iteration
}
```

**Path Structure:**
- Save: `output/{dataset}/{model}/{save_name}/checkpoints/`
- Load (Kaggle): `/kaggle/input/project-hstl/output/{dataset}/{model}/{save_name}/checkpoints/`

---

## Architecture Details

### HSTL Model Components

**1. ARME (Adaptive Region-based Motion Extractor)**
- ARME1: 1->32 channels, full 64x64 spatial
- ARME2: 32->64 channels, split [40, 24] regions
- ARME3: 64->128 channels, split [8, 32, 16, 8] regions

**2. FTA (Frame-level Temporal Aggregation)**
- Multi-scale temporal pooling (kernels: 3x1x1, 5x1x1)
- Attention-weighted aggregation

**3. ASTP (Adaptive Spatio-Temporal Pooling)**
- Per-region GeM pooling
- Total: 73 parts (1 + 2 + 2 + 4 + 64)

**4. SeFC (Separable Fully Connected)**
- Part-wise FC layers: 73 parts x 128 channels

**Output:** 73-part embeddings for metric learning

### Backbone Comparison

| Aspect | BasicConv3d | P3D (2+1D) |
|--------|-------------|------------|
| Parameters | Lower | Slightly higher |
| Computation | Standard | More efficient |
| Accuracy | Baseline | Typically better |
| Training speed | Faster | Comparable |

### Loss Function Comparison

| Aspect | TripletLoss | CircleLoss |
|--------|-------------|------------|
| Optimization | Harder (local optima) | More stable |
| Convergence | Slower | Faster |
| Hyperparameters | margin=0.2 | m=0.25, gamma=128 |
| Performance | Baseline | Typically better |

---

## Training Parameters

### Recommended Settings (Kaggle T4/P100)

```yaml
# Optimizer
optimizer_cfg:
  solver: Adam
  lr: 1.0e-4
  weight_decay: 5.0e-4

# Scheduler
scheduler_cfg:
  scheduler: MultiStepLR
  milestones: [70000]
  gamma: 0.1

# Training
trainer_cfg:
  batch_size: [8, 8]        # 8 persons x 8 samples = 64 total
  frames_num_fixed: 30       # Frames per sequence
  enable_float16: true       # FP16 for memory efficiency
  save_iter: 2500            # Checkpoint frequency
  log_iter: 300              # Logging frequency
  total_iter: 100000         # Total iterations
```

**Memory Usage (FP16):**
- Batch [8,8] x 30 frames x 64x64: ~8-10GB VRAM
- Safe for T4 (16GB) and P100 (16GB)

---

## Batch Testing Tool

### Purpose

Test multiple checkpoints across different model variants efficiently.

### Configuration

**Edit `batch_test_config.yaml`:**
```yaml
dataset_root: D:/CASIA-B-pkl
dataset_partition: ./misc/partitions/CASIA-B_include_005.json

models:
  p3d-circleloss:
    iterations: [2500, 5000, 7500, 10000, ..., 100000]
    save_name: p3d-circleloss
    checkpoint_path: ./output/CASIA-B/HSTL/p3d-circleloss/checkpoints
    backbone_type: p3d
    loss_type: circle
```

**Critical:** `backbone_type` and `loss_type` must match how the checkpoint was trained.

### Usage

**Local:**
```bash
python batch_test.py
```

**Kaggle:**
```python
# In Kaggle notebook
%cd /kaggle/input/project-hstl
!pip install -q pytorch-metric-learning pyyaml
!python batch_test_kaggle.py
```

### Output

**Summary table** (`batch_test_results/summary.txt`):
```
Model: p3d-circleloss
Iteration    NM (Incl)    BG (Incl)    CL (Incl)    NM (Excl)    BG (Excl)    CL (Excl)
2500         92.3         85.1         79.8         91.5         84.2         78.1
5000         94.8         87.2         82.4         94.1         86.3         81.0
...
```

**Detailed results** (`batch_test_results/detailed_results.txt`):
- Full angle-wise accuracy for each checkpoint
- NM, BG, CL results for all 11 view angles

**Individual logs** (`batch_test_results/logs/{model}/{model}_{iter}.log`):
- Complete test output for each checkpoint
- Includes all metrics and debugging info

---

## Common Issues and Solutions

### Issue 1: State Dict Mismatch

**Error:**
```
Missing key(s): "arme2.0.conv3d.0.spatial_conv.weight"
Unexpected key(s): "arme2.0.conv3d.0.conv3d.weight"
```

**Cause:** Checkpoint architecture doesn't match current code.

**Solution:** Ensure `backbone_type` in config matches checkpoint:
- Checkpoint has `conv3d.weight` -> use `backbone_type: conv3d`
- Checkpoint has `spatial_conv/temporal_conv` -> use `backbone_type: p3d`

### Issue 2: Loss Type Mismatch

**Error:**
```
KeyError: 'triplet' (or 'circle')
```

**Cause:** Model builds with wrong loss_type.

**Solution:** Ensure `model_cfg.loss_type` matches `loss_cfg[0].log_prefix`.

### Issue 3: Memory Allocation Failure (Preprocessing)

**Error:**
```
ImportError: DLL load failed while importing cv2: The paging file is too small
```

**Solution:** Use single worker:
```bash
python datasets/pretreatment.py -i <input> -o <output> -n 1
```

### Issue 4: Incomplete Test Results

**Issue:** Testing shows only 17 subjects (075-091) instead of 50 (075-124).

**Cause:** Dataset missing subjects 092-124.

**Solution:** 
1. Download complete CASIA-B dataset
2. Preprocess missing subjects: `--start_id 092 --end_id 124`
3. Re-run tests

### Issue 5: Batch Size Error

**Error:**
```
TypeError: 'int' object is not subscriptable
```

**Cause:** TripletSampler requires `batch_size` as list [P, K].

**Solution:**
```yaml
sampler:
  batch_size:
    - 8  # P: number of persons
    - 8  # K: samples per person
```

---

## Performance Benchmarks

### p3d-circleloss (100k iterations)

**Test Set: CASIA-B subjects 075-124 (50 subjects)**

| Condition | Rank-1 Accuracy (Exclude identical-view) |
|-----------|------------------------------------------|
| NM | 98.0% |
| BG | 90.3% |
| CL | 87.7% |

**Training time:** ~12 hours per 25k iterations on Kaggle P100

### Comparison Across Variants

| Variant | NM | BG | CL | Notes |
|---------|----|----|----| ------|
| HSTL (conv3d + triplet) | 99.0% | 93.1% | 94.3% | Baseline |
| p3d (p3d + triplet) | TBD | TBD | TBD | Needs testing |
| med-circleloss (conv3d + circle) | TBD | TBD | TBD | Needs testing |
| p3d-circleloss (p3d + circle) | 98.0% | 90.3% | 87.7% | Best convergence |

**Note:** Results vary based on training iterations and random seed.

---

## Advanced Usage

### Custom Backbone

Add new backbone in `lib/modeling/models/HSTL-CB.py`:

```python
class ARME_Custom(nn.Module):
    def __init__(self, in_channels, out_channels, split_param, m, ...):
        # Your implementation
        pass

# In build_network():
if backbone_type == 'custom':
    self.arme2 = nn.Sequential(ARME_Custom(...))
```

### Custom Loss Function

1. Create `lib/modeling/losses/custom.py`
2. Inherit from `BaseLoss`
3. Add to `loss_cfg` with matching `log_prefix`

### Hyperparameter Tuning

**Learning rate schedule:**
```yaml
scheduler_cfg:
  milestones: [10000, 20000, 30000]  # LR drops at these iterations
  gamma: 0.1                          # LR *= 0.1 at each milestone
```

**Batch size (memory vs speed):**
- `[16, 4]`: 64 samples, more diverse
- `[8, 8]`: 64 samples, better triplets
- `[8, 4]`: 32 samples, lower memory

---

## Model Variants Configuration Reference

### Variant 1: HSTL (Original)
```yaml
model_cfg:
  backbone_type: conv3d
  loss_type: triplet
trainer_cfg:
  save_name: HSTL
loss_cfg:
  - type: TripletLoss
    margin: 0.2
    log_prefix: triplet
```

### Variant 2: P3D Backbone
```yaml
model_cfg:
  backbone_type: p3d
  loss_type: triplet
trainer_cfg:
  save_name: p3d
loss_cfg:
  - type: TripletLoss
    margin: 0.2
    log_prefix: triplet
```

### Variant 3: CircleLoss
```yaml
model_cfg:
  backbone_type: conv3d
  loss_type: circle
trainer_cfg:
  save_name: med-circleloss
loss_cfg:
  - type: CircleLoss
    m: 0.25
    gamma: 128
    log_prefix: circle
```

### Variant 4: P3D + CircleLoss
```yaml
model_cfg:
  backbone_type: p3d
  loss_type: circle
trainer_cfg:
  save_name: p3d-circleloss
loss_cfg:
  - type: CircleLoss
    m: 0.25
    gamma: 128
    log_prefix: circle
```

---

## Kaggle API Workflow

### Complete Training Cycle

**1. Prepare locally:**
```bash
# Edit config/hstl.yaml with desired variant
# Ensure backbone_type, loss_type, save_name match
```

**2. Push to Kaggle:**
```bash
kaggle datasets version -p . --dir-mode zip -m "Training session N"
kaggle kernels push -p .
```

**3. Monitor:**
```bash
kaggle kernels status yourusername/hstl-training
```

**4. Download results:**
```bash
kaggle kernels output yourusername/hstl-training -p ./session_N_output
```

**5. Extract checkpoint:**
```bash
# Checkpoints saved to /kaggle/working/output/
# Download and add to project-hstl dataset for next session
```

### Multi-Session Training

For training beyond 12 hours:

**Session 1:**
```yaml
trainer_cfg:
  restore_hint: 0
  total_iter: 100000
```
Runs 0->25000, saves checkpoint-25000.pt

**Session 2:**
```yaml
trainer_cfg:
  restore_hint: 25000
  optimizer_reset: false
  scheduler_reset: false
  total_iter: 100000
```
Continues 25000->50000, saves checkpoint-50000.pt

**Between sessions:**
1. Download checkpoint from /kaggle/working/
2. Upload to project-hstl dataset
3. Update dataset version
4. Start new session

---

## Troubleshooting

### Distributed Training Errors

**Error:** "Default process group has not been initialized"

**Solution:** Already handled via safe wrappers in `lib/utils/common.py`. Ensure:
```python
from utils import safe_get_rank, safe_get_world_size
```

### TensorBoard Import Error (Kaggle)

**Error:** "cannot import name 'notf' from tensorboard"

**Solution:** Already handled. TensorBoard auto-disabled on Kaggle.

### Python 3.11 Compatibility

**Error:** "module 'inspect' has no attribute 'getargspec'"

**Solution:** Already fixed. Code uses `getfullargspec()`.

### Read-Only File System (Kaggle)

**Error:** "Read-only file system: /kaggle/input/..."

**Solution:** Already handled. Code uses:
- Write to: `/kaggle/working/output/`
- Read from: `/kaggle/input/project-hstl/output/`

---

## Development Notes

### Code Modifications for Kaggle

**1. Single-GPU compatibility** (`lib/utils/common.py`):
- `safe_get_rank()`: Returns 0 if not distributed
- `safe_get_world_size()`: Returns 1 if not distributed
- `is_distributed()`: Checks if distributed initialized

**2. Kaggle path detection** (`lib/main.py`, `lib/modeling/base_model.py`):
```python
if os.path.exists('/kaggle/input'):
    # Use Kaggle-specific paths
```

**3. TensorBoard conditional** (`lib/utils/msg_manager.py`):
```python
if os.path.exists('/kaggle/input'):
    SummaryWriter = None  # Disable on Kaggle
```

### Testing with Complete Dataset

After preprocessing subjects 092-124:
- Update `dataset_root` to complete dataset path
- Re-run batch tests for accurate benchmarks
- Results comparable to published papers

---

## Citation

```bibtex
@InProceedings{Wang_2023_ICCV,
    author    = {Wang, Lei and Liu, Bo and Liang, Fangfang and Wang, Bincheng},
    title     = {Hierarchical Spatio-Temporal Representation Learning for Gait Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19639-19649}
}
```

## Acknowledgments

- Original codebase based on [OpenGait](https://github.com/ShiqiYu/OpenGait)
- HSTL implementation from [gudaochangsheng/HSTL](https://github.com/gudaochangsheng/HSTL)
- PyTorch Metric Learning library for CircleLoss

---

## License

This project is for research purposes. Please refer to original paper and codebase for licensing terms.

---

## Contact

For issues, improvements, or questions about this implementation, please refer to the original HSTL repository or open an issue.


