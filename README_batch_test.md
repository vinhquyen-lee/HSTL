# Batch Testing Tool for HSTL Models

Simple tool to test multiple model checkpoints at once.

## Files

- `batch_test_config.yaml` - Configuration file (define models and iterations)
- `batch_test.py` - Local testing script
- `batch_test_kaggle.py` - Kaggle-optimized script

## Usage

### Local Testing

1. Edit `batch_test_config.yaml` to configure:
   - Dataset paths
   - Models to test
   - Checkpoint iterations

2. Run:
```bash
python batch_test.py
```

### Kaggle Testing

1. Upload checkpoints to Kaggle dataset
2. Edit paths in `batch_test_kaggle.py` (KAGGLE_CONFIG section)
3. Run in Kaggle notebook:
```python
!python batch_test_kaggle.py
```

## Output

Results saved to `batch_test_results/`:
- `summary.txt` - Table of all results
- `detailed_results.txt` - Full results with angle-wise accuracy
- `logs/[model]/[model]_[iter].log` - Individual test logs

## Result Format

Preserves exact format from training logs:
```
===Rank-1 (Include identical-view cases)===
NM: 98.5,	BG: 92.9,	CL: 93.7
===Rank-1 (Exclude identical-view cases)===
NM: 98.4,	BG: 92.5,	CL: 93.0
===Rank-1 of each angle (Exclude identical-view cases)===
NM: [ 96.25  99.69 100.00 ...]
BG: [90.91 94.55 93.03 ...]
CL: [87.50 93.75 99.06 ...]
```

