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
    kaggle kernels output huypt94/improved-hstl -p logs\kaggle --force
    ```
---
## Reference

- [Kaggle from Deepwiki](https://deepwiki.com/Kaggle/kaggle-api/2-installation-and-setup) (main source)

---
# BUG

> from ducmanh device


(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> kaggle datasets create -p . --dir-mode zip
Starting upload for file .gitignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:01<00:00, 92.8B/s]
Upload successful: .gitignore (132B)
Starting upload for file .kaggleignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77.0/77.0 [00:01<00:00, 54.4B/s]
Upload successful: .kaggleignore (77B)
Starting upload for file colab.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.1k/15.1k [00:01<00:00, 11.6kB/s]
Upload successful: colab.ipynb (15KB)
Starting upload for file config.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.03k/4.03k [00:01<00:00, 3.06kB/s]
Upload successful: config.zip (4KB)
Starting upload for file datasets.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.68k/3.68k [00:01<00:00, 2.80kB/s]
Upload successful: datasets.zip (4KB)
Starting upload for file kaggle.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 589/589 [00:01<00:00, 443B/s]
Upload successful: kaggle.ipynb (589B)
Starting upload for file kaggle.md
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.09k/4.09k [00:01<00:00, 3.15kB/s]
Upload successful: kaggle.md (4KB)
Starting upload for file kaggle.py
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:01<00:00, 194B/s]
Upload successful: kaggle.py (280B)
Starting upload for file lib.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32.4k/32.4k [00:01<00:00, 24.8kB/s]
Upload successful: lib.zip (32KB)
[Errno 22] Invalid argument
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> kaggle datasets create -p . --dir-mode zip
Starting upload for file .gitignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:01<00:00, 101B/s]
Upload successful: .gitignore (132B)
Starting upload for file .kaggleignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77.0/77.0 [00:01<00:00, 61.7B/s]
Upload successful: .kaggleignore (77B)
Starting upload for file colab.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.1k/15.1k [00:02<00:00, 6.92kB/s]
Upload successful: colab.ipynb (15KB)
Starting upload for file config.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.03k/4.03k [00:01<00:00, 3.12kB/s]
Upload successful: config.zip (4KB)
Starting upload for file datasets.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.68k/3.68k [00:01<00:00, 3.13kB/s]
Upload successful: datasets.zip (4KB)
Starting upload for file kaggle.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 589/589 [00:01<00:00, 481B/s]
Upload successful: kaggle.ipynb (589B)
Starting upload for file kaggle.md
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.09k/4.09k [00:01<00:00, 3.39kB/s]
Upload successful: kaggle.md (4KB)
Starting upload for file kaggle.py
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:01<00:00, 211B/s]
Upload successful: kaggle.py (280B)
Starting upload for file lib.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32.4k/32.4k [00:01<00:00, 22.9kB/s]
Upload successful: lib.zip (32KB)
[Errno 22] Invalid argument
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> kaggle datasets create -p . --dir-mode zip
Starting upload for file .gitignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:01<00:00, 107B/s]
Upload successful: .gitignore (132B)
Starting upload for file .kaggleignore
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77.0/77.0 [00:01<00:00, 58.0B/s]
Upload successful: .kaggleignore (77B)
Starting upload for file colab.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.1k/15.1k [00:01<00:00, 11.8kB/s]
Upload successful: colab.ipynb (15KB)
Starting upload for file config.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.03k/4.03k [00:01<00:00, 3.07kB/s]
Upload successful: config.zip (4KB)
Starting upload for file datasets.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.68k/3.68k [00:01<00:00, 2.84kB/s]
Upload successful: datasets.zip (4KB)
Starting upload for file kaggle.ipynb
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 589/589 [00:01<00:00, 481B/s]
Upload successful: kaggle.ipynb (589B)
Starting upload for file kaggle.md
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.09k/4.09k [00:01<00:00, 3.10kB/s]
Upload successful: kaggle.md (4KB)
Starting upload for file kaggle.py
Error while trying to load upload info: KaggleObject.from_dict() got an unexpected keyword argument 'token'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:01<00:00, 228B/s]
Upload successful: kaggle.py (280B)
Starting upload for file lib.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32.4k/32.4k [00:01<00:00, 24.7kB/s]
Upload successful: lib.zip (32KB)
[Errno 22] Invalid argument
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> kaggle --version
Kaggle API 1.8.3
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> where kaggle
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> python -m pip install --upgrade --force-reinstall kaggle
Collecting kaggle
  Using cached kaggle-1.8.3-py3-none-any.whl.metadata (16 kB)
Collecting black>=24.10.0 (from kaggle)
  Using cached black-25.12.0-cp313-cp313-win_amd64.whl.metadata (86 kB)
Collecting bleach (from kaggle)
  Using cached bleach-6.3.0-py3-none-any.whl.metadata (31 kB)
Collecting kagglesdk<1.0,>=0.1.14 (from kaggle)
  Using cached kagglesdk-0.1.14-py3-none-any.whl.metadata (13 kB)
Collecting mypy>=1.15.0 (from kaggle)
  Using cached mypy-1.19.1-cp313-cp313-win_amd64.whl.metadata (2.3 kB)
Collecting packaging (from kaggle)
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting protobuf (from kaggle)
  Using cached protobuf-6.33.2-cp310-abi3-win_amd64.whl.metadata (593 bytes)
Collecting python-dateutil (from kaggle)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting python-slugify (from kaggle)
  Using cached python_slugify-8.0.4-py2.py3-none-any.whl.metadata (8.5 kB)
Collecting requests (from kaggle)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting six>=1.10 (from kaggle)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting tqdm (from kaggle)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting types-requests (from kaggle)
  Using cached types_requests-2.32.4.20250913-py3-none-any.whl.metadata (2.0 kB)
Collecting types-tqdm (from kaggle)
  Using cached types_tqdm-4.67.0.20250809-py3-none-any.whl.metadata (1.7 kB)
Collecting urllib3>=1.15.1 (from kaggle)
  Using cached urllib3-2.6.2-py3-none-any.whl.metadata (6.6 kB)
Collecting click>=8.0.0 (from black>=24.10.0->kaggle)
  Using cached click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
Collecting mypy-extensions>=0.4.3 (from black>=24.10.0->kaggle)
  Using cached mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
Collecting pathspec>=0.9.0 (from black>=24.10.0->kaggle)
  Using cached pathspec-0.12.1-py3-none-any.whl.metadata (21 kB)
Collecting platformdirs>=2 (from black>=24.10.0->kaggle)
  Downloading platformdirs-4.5.1-py3-none-any.whl.metadata (12 kB)
Collecting pytokens>=0.3.0 (from black>=24.10.0->kaggle)
  Using cached pytokens-0.3.0-py3-none-any.whl.metadata (2.0 kB)
Collecting colorama (from click>=8.0.0->black>=24.10.0->kaggle)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting typing_extensions>=4.6.0 (from mypy>=1.15.0->kaggle)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting librt>=0.6.2 (from mypy>=1.15.0->kaggle)
  Using cached librt-0.7.4-cp313-cp313-win_amd64.whl.metadata (1.4 kB)
Collecting webencodings (from bleach->kaggle)
  Using cached webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
Collecting text-unidecode>=1.3 (from python-slugify->kaggle)
  Using cached text_unidecode-1.3-py2.py3-none-any.whl.metadata (2.4 kB)
Collecting charset_normalizer<4,>=2 (from requests->kaggle)
  Using cached charset_normalizer-3.4.4-cp313-cp313-win_amd64.whl.metadata (38 kB)
Collecting idna<4,>=2.5 (from requests->kaggle)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting certifi>=2017.4.17 (from requests->kaggle)
  Using cached certifi-2025.11.12-py3-none-any.whl.metadata (2.5 kB)
Using cached kaggle-1.8.3-py3-none-any.whl (102 kB)
Using cached kagglesdk-0.1.14-py3-none-any.whl (159 kB)
Using cached black-25.12.0-cp313-cp313-win_amd64.whl (1.4 MB)
Using cached click-8.3.1-py3-none-any.whl (108 kB)
Using cached mypy-1.19.1-cp313-cp313-win_amd64.whl (10.1 MB)
Using cached librt-0.7.4-cp313-cp313-win_amd64.whl (49 kB)
Using cached mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Using cached pathspec-0.12.1-py3-none-any.whl (31 kB)
Downloading platformdirs-4.5.1-py3-none-any.whl (18 kB)
Using cached pytokens-0.3.0-py3-none-any.whl (12 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached urllib3-2.6.2-py3-none-any.whl (131 kB)
Using cached bleach-6.3.0-py3-none-any.whl (164 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached protobuf-6.33.2-cp310-abi3-win_amd64.whl (436 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached python_slugify-8.0.4-py2.py3-none-any.whl (10 kB)
Using cached text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Using cached charset_normalizer-3.4.4-cp313-cp313-win_amd64.whl (107 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached certifi-2025.11.12-py3-none-any.whl (159 kB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached types_requests-2.32.4.20250913-py3-none-any.whl (20 kB)
Using cached types_tqdm-4.67.0.20250809-py3-none-any.whl (24 kB)
Using cached webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
Installing collected packages: webencodings, text-unidecode, urllib3, typing_extensions, six, pytokens, python-slugify, protobuf, platformdirs, pathspec, packaging, mypy-extensions, librt, idna, colorama, charset_normalizer, certifi, bleach, types-requests, tqdm, requests, python-dateutil, mypy, click, types-tqdm, kagglesdk, black, kaggle
  Attempting uninstall: webencodings
    Found existing installation: webencodings 0.5.1
    Uninstalling webencodings-0.5.1:
      Successfully uninstalled webencodings-0.5.1
  Attempting uninstall: text-unidecode
    Found existing installation: text-unidecode 1.3
    Uninstalling text-unidecode-1.3:
      Successfully uninstalled text-unidecode-1.3
  Attempting uninstall: urllib3
    Found existing installation: urllib3 2.5.0
    Uninstalling urllib3-2.5.0:
      Successfully uninstalled urllib3-2.5.0
  Attempting uninstall: typing_extensions
    Found existing installation: typing_extensions 4.15.0
    Uninstalling typing_extensions-4.15.0:
      Successfully uninstalled typing_extensions-4.15.0
  Attempting uninstall: six
    Found existing installation: six 1.17.0
    Uninstalling six-1.17.0:
      Successfully uninstalled six-1.17.0
  Attempting uninstall: pytokens
    Found existing installation: pytokens 0.3.0
    Uninstalling pytokens-0.3.0:
      Successfully uninstalled pytokens-0.3.0
  Attempting uninstall: python-slugify
    Found existing installation: python-slugify 8.0.4
    Uninstalling python-slugify-8.0.4:
      Successfully uninstalled python-slugify-8.0.4
  Attempting uninstall: protobuf
    Found existing installation: protobuf 6.30.2
    Uninstalling protobuf-6.30.2:
      Successfully uninstalled protobuf-6.30.2
  Attempting uninstall: platformdirs
    Found existing installation: platformdirs 4.3.8
    Uninstalling platformdirs-4.3.8:
      Successfully uninstalled platformdirs-4.3.8
  Attempting uninstall: pathspec
    Found existing installation: pathspec 0.12.1
    Uninstalling pathspec-0.12.1:
      Successfully uninstalled pathspec-0.12.1
  Attempting uninstall: packaging
    Found existing installation: packaging 24.2
    Uninstalling packaging-24.2:
      Successfully uninstalled packaging-24.2
  Attempting uninstall: mypy-extensions
    Found existing installation: mypy_extensions 1.1.0
    Uninstalling mypy_extensions-1.1.0:
      Successfully uninstalled mypy_extensions-1.1.0
  Attempting uninstall: librt
    Found existing installation: librt 0.7.4
    Uninstalling librt-0.7.4:
      Successfully uninstalled librt-0.7.4
  Attempting uninstall: idna
    Found existing installation: idna 3.11
    Uninstalling idna-3.11:
      Successfully uninstalled idna-3.11
  Attempting uninstall: colorama
    Found existing installation: colorama 0.4.6
    Uninstalling colorama-0.4.6:
      Successfully uninstalled colorama-0.4.6
  Attempting uninstall: charset_normalizer
    Found existing installation: charset-normalizer 3.4.4
    Uninstalling charset-normalizer-3.4.4:
      Successfully uninstalled charset-normalizer-3.4.4
  Attempting uninstall: certifi
    Found existing installation: certifi 2025.11.12
    Uninstalling certifi-2025.11.12:
      Successfully uninstalled certifi-2025.11.12
  Attempting uninstall: bleach
    Found existing installation: bleach 6.3.0
    Uninstalling bleach-6.3.0:
      Successfully uninstalled bleach-6.3.0
  Attempting uninstall: types-requests
    Found existing installation: types-requests 2.32.4.20250913
    Uninstalling types-requests-2.32.4.20250913:
      Successfully uninstalled types-requests-2.32.4.20250913
  Attempting uninstall: tqdm
    Found existing installation: tqdm 4.67.1
    Uninstalling tqdm-4.67.1:
      Successfully uninstalled tqdm-4.67.1
  Attempting uninstall: requests
    Found existing installation: requests 2.32.5
    Uninstalling requests-2.32.5:
      Successfully uninstalled requests-2.32.5
  Attempting uninstall: python-dateutil
    Found existing installation: python-dateutil 2.9.0.post0
    Uninstalling python-dateutil-2.9.0.post0:
      Successfully uninstalled python-dateutil-2.9.0.post0
  Attempting uninstall: mypy
    Found existing installation: mypy 1.19.1
    Uninstalling mypy-1.19.1:
      Successfully uninstalled mypy-1.19.1
  Attempting uninstall: click
    Found existing installation: click 8.1.8
    Uninstalling click-8.1.8:
      Successfully uninstalled click-8.1.8
  Attempting uninstall: types-tqdm
    Found existing installation: types-tqdm 4.67.0.20250809
    Uninstalling types-tqdm-4.67.0.20250809:
      Successfully uninstalled types-tqdm-4.67.0.20250809
  Attempting uninstall: kagglesdk
    Found existing installation: kagglesdk 0.1.14
    Uninstalling kagglesdk-0.1.14:
      Successfully uninstalled kagglesdk-0.1.14
  Attempting uninstall: black
    Found existing installation: black 25.12.0
    Uninstalling black-25.12.0:
      Successfully uninstalled black-25.12.0
  Attempting uninstall: kaggle
    Found existing installation: kaggle 1.8.3
    Uninstalling kaggle-1.8.3:
      Successfully uninstalled kaggle-1.8.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
anaconda-cli-base 0.6.0 requires click<8.2, but you have click 8.3.1 which is incompatible.
Successfully installed black-25.12.0 bleach-6.3.0 certifi-2025.10.5 charset_normalizer-3.4.4 click-8.1.8 colorama-0.4.6 idna-3.11 kaggle-1.8.3 kagglesdk-0.1.14 librt-0.7.4 mypy-1.19.1 mypy-extensions-1.1.0 packaging-25.0 pathspec-0.12.1 platformdirs-4.5.0 protobuf-6.33.2 python-dateutil-2.9.0.post0 python-slugify-8.0.4 pytokens-0.3.0 requests-2.32.5 six-1.17.0 text-unidecode-1.3 tqdm-4.67.1 types-requests-2.32.4.20250913 types-tqdm-4.67.0.20250809 typing_extensions-4.15.0 urllib3-2.5.0 webencodings-0.5.1
(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> conda env list

# conda environments:
#
# *  -> active
# + -> frozen
                         C:\Users\ADMIN\anaconda3\envs\3DLocalCNN
base                 *   C:\Users\ADMIN\miniconda3
3DLocalCNN               C:\Users\ADMIN\miniconda3\envs\3DLocalCNN
d2l                      C:\Users\ADMIN\miniconda3\envs\d2l
deepface310              C:\Users\ADMIN\miniconda3\envs\deepface310
hstl_cpu                 C:\Users\ADMIN\miniconda3\envs\hstl_cpu

(base) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> conda activate hstl_cpu
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> kaggle competitions list
kaggle : The term 'kaggle' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was 
included, verify that the path is correct and try again.
At line:1 char:1
+ kaggle competitions list
+ ~~~~~~
    + CategoryInfo          : ObjectNotFound: (kaggle:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException


Suggestion [3,General]: The command kaggle was not found, but does exist in the current location. Windows PowerShell does not load commands from the current location by default. If you trust this command, instead type: ".\kaggle". See "get-help about_Command_Precedence" for more details.
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> python -m pip install --upgrade --force-reinstall kaggle
Collecting kaggle
  Downloading kaggle-1.7.4.5-py3-none-any.whl.metadata (16 kB)
Collecting bleach (from kaggle)
  Using cached bleach-6.3.0-py3-none-any.whl.metadata (31 kB)
Collecting certifi>=14.05.14 (from kaggle)
  Using cached certifi-2025.11.12-py3-none-any.whl.metadata (2.5 kB)
Collecting charset-normalizer (from kaggle)
  Using cached charset_normalizer-3.4.4-cp310-cp310-win_amd64.whl.metadata (38 kB)
Collecting idna (from kaggle)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting protobuf (from kaggle)
  Using cached protobuf-6.33.2-cp310-abi3-win_amd64.whl.metadata (593 bytes)
Collecting python-dateutil>=2.5.3 (from kaggle)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting python-slugify (from kaggle)
  Using cached python_slugify-8.0.4-py2.py3-none-any.whl.metadata (8.5 kB)
Collecting requests (from kaggle)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting setuptools>=21.0.0 (from kaggle)
  Using cached setuptools-80.9.0-py3-none-any.whl.metadata (6.6 kB)
Collecting six>=1.10 (from kaggle)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting text-unidecode (from kaggle)
  Using cached text_unidecode-1.3-py2.py3-none-any.whl.metadata (2.4 kB)
Collecting tqdm (from kaggle)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting urllib3>=1.15.1 (from kaggle)
  Using cached urllib3-2.6.2-py3-none-any.whl.metadata (6.6 kB)
Collecting webencodings (from kaggle)
  Using cached webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
Collecting colorama (from tqdm->kaggle)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Downloading kaggle-1.7.4.5-py3-none-any.whl (181 kB)
Using cached certifi-2025.11.12-py3-none-any.whl (159 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached setuptools-80.9.0-py3-none-any.whl (1.2 MB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached urllib3-2.6.2-py3-none-any.whl (131 kB)
Using cached bleach-6.3.0-py3-none-any.whl (164 kB)
Using cached charset_normalizer-3.4.4-cp310-cp310-win_amd64.whl (107 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached protobuf-6.33.2-cp310-abi3-win_amd64.whl (436 kB)
Using cached python_slugify-8.0.4-py2.py3-none-any.whl (10 kB)
Using cached text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
Installing collected packages: webencodings, text-unidecode, urllib3, six, setuptools, python-slugify, protobuf, idna, colorama, charset-normalizer, certifi, bleach, tqdm, requests, python-dateutil, kaggle
  Attempting uninstall: urllib3
    Found existing installation: urllib3 2.6.0
    Uninstalling urllib3-2.6.0:
      Successfully uninstalled urllib3-2.6.0
  Attempting uninstall: setuptools
    Found existing installation: setuptools 80.9.0
    Uninstalling setuptools-80.9.0:
      Successfully uninstalled setuptools-80.9.0
  Attempting uninstall: protobuf
    Found existing installation: protobuf 6.33.2
    Uninstalling protobuf-6.33.2:
      Successfully uninstalled protobuf-6.33.2
  Attempting uninstall: idna
    Found existing installation: idna 3.11
    Uninstalling idna-3.11:
      Successfully uninstalled idna-3.11
  Attempting uninstall: colorama
    Found existing installation: colorama 0.4.6
    Uninstalling colorama-0.4.6:
      Successfully uninstalled colorama-0.4.6
  Attempting uninstall: charset-normalizer
    Found existing installation: charset-normalizer 3.4.4
    Uninstalling charset-normalizer-3.4.4:
      Successfully uninstalled charset-normalizer-3.4.4
  Attempting uninstall: certifi
    Found existing installation: certifi 2025.11.12
    Uninstalling certifi-2025.11.12:
      Successfully uninstalled certifi-2025.11.12
  Attempting uninstall: tqdm
    Found existing installation: tqdm 4.67.1
    Uninstalling tqdm-4.67.1:
      Successfully uninstalled tqdm-4.67.1
  Attempting uninstall: requests
    Found existing installation: requests 2.32.5
    Uninstalling requests-2.32.5:
      Successfully uninstalled requests-2.32.5
Successfully installed bleach-6.3.0 certifi-2025.11.12 charset-normalizer-3.4.4 colorama-0.4.6 idna-3.11 kaggle-1.7.4.5 protobuf-6.33.2 python-dateutil-2.9.0.post0 python-slugify-8.0.4 requests-2.32.5 setuptools-80.9.0 six-1.17.0 text-unidecode-1.3 tqdm-4.67.1 urllib3-2.6.2 webencodings-0.5.1
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> python -c "import kaggle,sys; print(kaggle.__version__, kaggle.__file__)"

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\main.py", line 5, in <module>
    from modeling import models
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\models\__init__.py", line 11, in <module>
    module = import_module(f"{__name__}.{module_name}")
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\models\HSTL-CB.py", line 5, in <module>
    from ..base_model import BaseModel
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\base_model.py", line 16, in <module>
    from . import backbones
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\backbones\__init__.py", line 11, in <module>
    module = import_module(f"{__name__}.{module_name}")
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\backbones\plain.py", line 2, in <module>
    from ..modules import BasicConv2d
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\modules.py", line 5, in <module>
    from utils import clones, is_list_or_tuple
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\utils\__init__.py", line 11, in <module>
    from .msg_manager import get_msg_mgr
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\utils\msg_manager.py", line 5, in <module>
    import torchvision.utils as vutils
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\__init__.py", line 5, in <module>
    from torchvision import datasets, io, models, ops, transforms, utils
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\__init__.py", line 17, in <module>
    from . import detection, optical_flow, quantization, segmentation, video
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\detection\__init__.py", line 1, in <module>
    from .faster_rcnn import *
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\detection\faster_rcnn.py", line 16, in <module>
    from .anchor_utils import AnchorGenerator
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\detection\anchor_utils.py", line 10, in <module>
    class AnchorGenerator(nn.Module):
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\detection\anchor_utils.py", line 63, in AnchorGenerator
    device: torch.device = torch.device("cpu"),
C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\site-packages\torchvision\models\detection\anchor_utils.py:63: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:77.)
  device: torch.device = torch.device("cpu"),
Traceback (most recent call last):
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\main.py", line 5, in <module>
    from modeling import models
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\models\__init__.py", line 11, in <module>
    module = import_module(f"{__name__}.{module_name}")
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\models\HSTL-CB.py", line 5, in <module>
    from ..base_model import BaseModel
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\base_model.py", line 17, in <module>
    from .loss_aggregator import LossAggregator
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\loss_aggregator.py", line 2, in <module>
    from . import losses
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\losses\__init__.py", line 11, in <module>
    module = import_module(f"{__name__}.{module_name}")
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\lib\modeling\losses\circle.py", line 3, in <module>
    from pytorch_metric_learning.losses import CircleLoss as PMLCircleLoss
ModuleNotFoundError: No module named 'pytorch_metric_learning'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL\kaggle.py", line 15, in <module>
    subprocess.run(cmd, check=True)
  File "C:\Users\ADMIN\miniconda3\envs\hstl_cpu\lib\subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['python', 'lib/main.py', '--cfgs', './config/hstl.yaml', '--phase', 'train', '--log_to_file']' returned non-zero exit status 1.
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> python -c "import sysconfig;print(sysconfig.get_path('scripts'))"
C:\Users\ADMIN\miniconda3\envs\hstl_cpu\Scripts
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> C:\Users\ADMIN\miniconda3\envs\hstl_cpu\Scripts\kaggle.exe --version
Kaggle API 1.7.4.5
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> setx PATH "%PATH%;C:\Users\ADMIN\miniconda3\envs\hstl_cpu\Scripts"

SUCCESS: Specified value was saved.
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> where kaggle
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL>     kaggle datasets create -p ./data_dir --dir-mode zip
Invalid folder: ./data_dir
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL>     kaggle datasets create -p . --dir-mode zip         
Starting upload for file .gitignore
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:01<00:00, 118B/s]
Upload successful: .gitignore (132B)
Starting upload for file .kaggleignore
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77.0/77.0 [00:01<00:00, 75.7B/s]
Upload successful: .kaggleignore (77B)
Starting upload for file colab.ipynb
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.1k/15.1k [00:01<00:00, 8.90kB/s]
Upload successful: colab.ipynb (15KB)
Starting upload for file config.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.03k/4.03k [00:01<00:00, 3.96kB/s]
Upload successful: config.zip (4KB)
Starting upload for file datasets.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.68k/3.68k [00:01<00:00, 3.08kB/s]
Upload successful: datasets.zip (4KB)
Starting upload for file kaggle.ipynb
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 589/589 [00:01<00:00, 570B/s]
Upload successful: kaggle.ipynb (589B)
Starting upload for file kaggle.md
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.09k/4.09k [00:01<00:00, 4.04kB/s]
Upload successful: kaggle.md (4KB)
Starting upload for file kaggle.py
Error while trying to load upload info: ApiStartBlobUploadRequest.__init__() got an unexpected keyword argument 'type'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:01<00:00, 171B/s]
Upload successful: kaggle.py (280B)
Starting upload for file lib.zip
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60.1k/60.1k [00:01<00:00, 43.0kB/s]
Upload successful: lib.zip (60KB)
[Errno 22] Invalid argument
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> git status
On branch update/medium
Your branch is ahead of 'origin/update/medium' by 1 commit.
  (use "git push" to publish your local commits)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        logsP3D/
        output/CASIA-B/HSTL/kaggle/

nothing added to commit but untracked files present (use "git add" to track)
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> git status
On branch update/medium
Your branch is ahead of 'origin/update/medium' by 1 commit.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   .gitignore

no changes added to commit (use "git add" and/or "git commit -a")
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> git add .
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> git commit -m "unresolved bug with kaggle api"
[update/medium d4244b0] unresolved bug with kaggle api
 1 file changed, 2 insertions(+), 1 deletion(-)
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> git status
On branch update/medium
Your branch is ahead of 'origin/update/medium' by 2 commits.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
(hstl_cpu) PS C:\Users\ADMIN\OneDrive - VNU-HCMUS\CNTT4-HK1\STH\HSTL> 