# How to use free resource from Kaggle ?

## 1. Kaggle enviroment

- Python 3.11.x -> modify current source code (from Python 3.10)

- **Dataset** (load from `/kaggle/input`) and **Kernel** (load from `kaggle/working`)

- Using Kaggle API is the best option

## 2. Why Kaggle API ?

In common uses, people use Kaggle on web, they need to upload dataset, source code. Users have to compressed them as `.zip` files and upload as datasets.

But, what happens when source code need some changes ? Web-based Kaggle does not let users apply changes on uploaded items. The only solution is removing the current source code on web-based Kaggle and upload a new compressed one. **What a mess!!!**

There is a better option, users can work with Kaggle by using API, transfer working items from local device. Users can also create sessions locally. That is **Kaggle API!!!**

It takes much time to read more and deeply understand Kaggle API. Now, we will jump into the core commands for this project only.

## Setup

```bash
1. Install          - pip install kaggle
2. Create API Key   - Visit Kaggle -> Settings -> Account -> Legacy API Credentials -> Create Legacy API Key
3. Config path      - Move auto-downloaded file "kaggle.json" to "~/.kaggle/" (for Linux) or "C:\Users\Admin\.kaggle" (for Windows)
4. Verify           - Run "kaggle competitions list" in terminal, the installation and authentication succeeded if list of active kaggle competitions is shown
5. DONE             - (leave advanced setup later)
```

## How to use Kaggle API ? (a.k.a Workflow)

### 1. Easier way to make a full multi-file project fit Kaggle's abstraction

There are two core seperate abstractions:

- Data -> Kaggle Dataset
- Source code -> Kaggle Kernel folder

#### 1. Data: use Kaggle Datasets + API (no manual zip)

1. One-time init
    ```bash
    kaggle datasets init -p ./data_dir   # creates "dataset-metadata.json" in "./data_dir"
    ```
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

    ```bash
    kaggle datasets create -p ./data_dir --dir-mode zip
    ```

3. Update on any change via API (no manual zip like using web-based version)

    ```bash
    kaggle datasets version -p ./data_dir --dir-mode zip -m "update data"
    ```
#### 2. Code: use "script kernel + folder push", not "code as dataset"

For a project, you should treat it as a **kernel workspace**, not as a dataset.

```bash
kaggle kernels init -p /path/to/kernel
# create `kernel-metadata.json`
# one-time init
```
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

Notebook is used to fit with kaggle workspace, in notebook, we call commands to run executed files
```bash
kaggle kernels push -p /path/to/kernel
# upload the whole folder and run it
# use every time modifying files in project
```

```bash
kaggle kernels status your_kaggle_username/my-project # poll execution status
```
#### 3. Output: Download all output to local device

```bash
kaggle kernels output your_kaggle_username/my-project -p outputs_local # download everything from `kaggle/working`
```

**Note: download log with different encoding by using cmd (if using Windows)**

- Paste these lines in cmd (enable for current cmd only)
    ```bash
    set PYTHONIOENCODING=utf-8
    set PYTHONUTF8=1
    chcp 65001
    ```
- Then try again with this command

    ```bash
    kaggle kernels output huypt94/improved-hstl -p logs\kaggle --force # kaggle kernels output vnhquynlquang/hstl-gait-recognition -p ./output_local --force
    ```
---
## Reference

- [Kaggle from Deepwiki](https://deepwiki.com/Kaggle/kaggle-api/2-installation-and-setup) (main source)
