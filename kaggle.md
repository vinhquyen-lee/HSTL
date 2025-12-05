source: https://deepwiki.com/Kaggle/kaggle-api/2-installation-and-setup?utm_source=chatgpt.com

pip install kaggle

create API key

move kaggle.json to C:\Users\Admin\.kaggle

verify installation and authentication with: kaggle competitions list
--> list of active kaggle competitions without any authentication errors --> success

Custom API Endpoint

default endpoint: https://www.kaggle.com/api/v1

specify a custom endpoint by setting the "KAGGLE_API_ENDPOINT" env var:


<!-- add workflow with kaggle API -->

## 1. Easier way to make a full multi-file project fit Kaggle’s abstraction

There are two core seperate abstractions:

- Data -> Kaggle Dataset
- Source code -> Kaggle Kernel folder

### 1. Data: use Kaggle Datasets + API (no manual zip)

1. One-time init
    > kaggle datasets init -p ./data_dir   # creates dataset-metadata.json

- edit `dataset-metadata.json`

    ```json
    {
    "title": "HSTL Project",
    "id": "huypt94/project-hstl",
    "licenses": [
        {
        "name": "CC0-1.0"
        }
    ]
    }
    ```

2. First upload
    > kaggle datasets create -p ./data_dir --dir-mode zip

3. Update on any change via API (no manual zip like using web-based version)

    > kaggle datasets version -p ./data_dir --dir-mode zip -m "update data"

### 2. Code: use “script kernel + folder push”, not “code as dataset”

For a project, you should treat it as a **kernel workspace**, not as a dataset.

> kaggle kernels init -p /path/to/kernel
-> create `kernel-metadata.json`
-> one-time init

- edit `kernel-metadata.json`
    ```json
    {
    "id": "yourname/my-project-kernel",
    "title": "My Project on Kaggle",
    "code_file": "main.py or main.ipynb",
    "language": "python",
    "kernel_type": "script or notebook",
    "enable_gpu": true,
    "enable_internet": false,
    "dataset_sources": ["yourname/your-dataset-id"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
    }
    ```

notebook is used to fit with kaggle workspace, in notebook, we call commands to run executed files

> kaggle kernels push -p /path/to/kernel
-> upload the whole folder and run it
-> every time modifying files in project

> kaggle kernels status your_kaggle_username/my-project
-> poll execution status

### 3. Output: Download all output to local device

> kaggle kernels output your_kaggle_username/my-project -p outputs_local
-> download everything from `kaggle/working`

<!-- download log with different encoding by using cmd -->
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001

kaggle kernels output huypt94/improved-hstl -p logs\kaggle --force
