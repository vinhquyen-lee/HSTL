import os
import subprocess

# Kaggle = 1 GPU → force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [
    "python",
    "lib/main.py",
    "--cfgs", "./config/hstl.yaml",
    "--phase", "train",
    "--log_to_file"
]

subprocess.run(cmd, check=True)
