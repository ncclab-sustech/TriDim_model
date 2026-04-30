# eeg_basis_mixer_v2_nobasis (release)

Single-model release of the **nobasis** TriAxis mixer baseline used in our
EEG cross-subject classification experiments. This package is intentionally
minimal: only one model (`eeg_basis_mixer_v2_nobasis`) and one data loader,
covering 19 public EEG datasets through a unified subject-wise h5 layout.

## 1. Layout

```
TeCh_nobasis_release/
├── run.py                          # entry point (argparse + yaml override)
├── requirements.txt
├── configs/
│   ├── datasets/<dataset>.yaml     # one file per dataset, sets root_path + hparams
│   └── electrodes/<dataset>.csv    # 3D electrode coordinates (used by channel adapter)
├── data_provider/
│   ├── data_factory.py             # routes every dataset to ADHDLoader
│   ├── data_loader.py              # subject-wise h5 loader, 3 split modes
│   └── uea.py                      # collate_fn / normalization
├── exp/
│   ├── exp_basic.py                # device + model registry
│   └── exp_classification.py       # train / val / test loop
├── layers/Augmentation.py          # jitter / scale / flip / mask augmentations
├── models/eeg_basis_mixer_v2_nobasis.py
├── utils/tools.py                  # cosine LR, EarlyStopping
└── scripts/
│   └── bsub/run_tridim_physionet.lsf 
│   ├── run_faced_nobasis_xsub.sh


```

## 2. Install

```bash
conda create -n nobasis python=3.10 -y
conda activate nobasis
pip install -r requirements.txt
```

The pinned versions correspond to PyTorch 2.4.1 + CUDA 12. Adjust the torch
build to match your CUDA driver if needed.

## 3. Data format

All 19 supported datasets share one h5 schema. Each subject is one file under
`<root_path>/`. The loader auto-detects the naming convention; the following
patterns all work without code changes:

| Style                                  | Examples                                | Datasets                          |
|----------------------------------------|-----------------------------------------|-----------------------------------|
| `sub_<int>.h5`                         | `sub_001.h5`, `sub_42.h5`               | most datasets                     |
| `sub-<int>.h5`                         | `sub-1.h5`                              | older preprocessing               |
| `sub-<gender><int>.h5`                 | `sub-f1.h5`, `sub-m2.h5`                | FACED-style gendered IDs          |
| `sub-<int>_task-...eeg.h5`             | `sub-001_task-eyesclosed_eeg.h5`        | AD65                              |
| `S<int>.h5`                            | `S001.h5`                               | Physionet_MI                      |
| `sub<int>.h5` (no separator)           | `sub000.h5`                             | FACED_new                         |
| `sub_SC<int><tail>.h5`                 | `sub_SC4001E0.h5`                       | sleep-cassette-200hz              |
| `A<int>[TE].h5`                        | `A01T.h5`, `A01E.h5`                    | BCIC2A (T/E treated as separate)  |
| `<H\|MDD> S<int> <EC\|EO\|TASK>.h5`    | `MDD S1 EC.h5`, `H S15 EO.h5`           | MDD                               |

Inside each h5:

```
sub_001.h5
└── trial_<i>           (h5.Group)
    └── segment_<j>     (h5.Group, attrs may carry "label")
        ├── eeg         (h5.Dataset, shape (C, T) or (T, C); attrs may carry "label")
        └── label       (optional dataset, used if attrs absent)
```

Channel order in `eeg` must match the rows of
`configs/electrodes/<dataset>.csv` (columns: `name, x, y, z` in head-coordinate
metres). Update the CSV if you preprocess with a different montage.

## 4. Run

The 5-seed cross-subject (4:3:3) sweep is the default configuration. CLI args
override yaml values. Minimal invocation（不要使用augmentations）:

```bash
python run.py \
    --model eeg_basis_mixer_v2_nobasis \
    --data FACED_new \
    --dataset_paths_yaml ./configs/datasets/FACED_new.yaml \
    --root_path /your/path/to/FACED_new \
    --gpu 0 --gpu_idx 0 --num_workers 4 \
    --itr 5 --seed_start 42 \
    --split_mode label_order --train_ratio 0.4 --val_ratio 0.3 \
    --augmentations none --select_metric F1
```

There is also `scripts/run_faced_nobasis_xsub.sh` you can copy as a template
for other datasets — change `--data`, `--dataset_paths_yaml`, and `--root_path`.

For deterministic cuBLAS:

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

(`run.py` already sets this if not present in the environment.)

## 5. Split modes

这里严格使用跨被试的，然后数据划分方式是4:3:3

| `--split_mode`               | Behavior                                               |
|------------------------------|--------------------------------------------------------|
| `label_order` *(default)*    | Deterministic subject split; ordered by class then ID. Used in the paper. |
| `stratified_random`          | Per-class random subject split, seeded by `--seed_start + itr_idx`. |
| `segment_stratified_random`  | Splits **segments** instead of subjects (NOT cross-subject; for ablations only). |

Defaults: `train_ratio=0.4`, `val_ratio=0.3`, test gets the rest.

## 6. Output

After all `--itr` seeds finish, `run.py` prints mean/std over six metrics:
Accuracy, Precision, Recall, F1, AUROC, AUPRC. Per-seed checkpoints are
written under `./checkpoints/<setting>/` and removed after testing. Per-seed
metric jsons land under `./results/`.

## 7. Notes for porting

* `--downstream_root /shared/dir` lets you point the loader at a single
  parent directory containing `<dataset>/` subfolders, instead of setting
  `--root_path` per run.
* `--use_channel_adapter` enables the RBF geometric prior over electrodes.
  It assumes both the input montage CSV and the canonical montage CSV are
  set (yaml does this for you).
* `--v_layer` is accepted for yaml compatibility but unused by the nobasis
  model.


## Qiming 服务器使用方法

请不要在登录节点直接运行训练任务，登录节点只用于代码编辑、环境配置、提交任务和查看日志。正式训练需要通过 `bsub` 提交到 GPU 队列。

### 1. 连接服务器

校外访问需要先连接学校 VPN。连接后，可使用VSCode Remote-SSH。建议在本地 `~/.ssh/config` 中加入：

```sshconfig
Host qiming
  HostName 172.18.6.10
  Port 18188
  User <your_username>
  ServerAliveInterval 60
  ServerAliveCountMax 120
```

然后在 VSCode 中选择：

```text
Remote-SSH: Connect to Host... -> qiming
```

进入服务器后，打开项目目录，例如：

```bash
cd ~/xinke/TriDim_small
```

请在home文件夹 ~/ 下建一个以自己名字命名的文件夹，比如 

```bash
cd ~/
mkdir junjie
```

### 2. 加载环境

项目推荐使用 conda 环境。每次登录后运行：

```bash
module load python/anaconda3/2022.10
source activate
conda activate ~/conda_envs/eeg3dim
```

如果环境不存在，可以按上述requirements新建，注意需安装pyyaml包。


### 3. 提交 GPU 任务

示例脚本位于：

```text
scripts/bsub/run_physionet_onlyT.lsf
```

示例内容：

```bash
#!/bin/bash
#BSUB -J tridim_physionet_onlyT
#BSUB -q gpu-bme-liuqy
#BSUB -m b07u22g
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -R "span[ptile=4]"
#BSUB -e logs/%J.err
#BSUB -o logs/%J.out

set -e

echo "Job started at:"
date
echo "Job ID: ${LSB_JOBID}"
echo "Running on host:"
hostname
echo "Working directory:"
pwd

cd ~/TriDim_small
mkdir -p logs

module load python/anaconda3/2022.10
source activate
conda activate ~/conda_envs/eeg3dim

echo "Python:"
which python
python -V

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

python run.py \
  --model eeg_basis_mixer_v2_nobasis_onlyT \
  --data Physionet_MI \
  --dataset_paths_yaml ./configs/datasets/Physionet_MI.yaml \
  --gpu 0 \
  --gpu_idx 0 \
  > logs/${LSB_JOBID}.train.log 2>&1

echo "Job finished at:"
date
```

提交任务：

```bash
mkdir -p logs
bsub < scripts/bsub/run_physionet_onlyT.lsf
```

### 4. 查看任务和日志

查看当前任务：

```bash
bjobs
```

查看某个任务详细信息：

```bash
bjobs -l <JOBID>
```

查看训练日志：

```bash
tail -f logs/<JOBID>.train.log
```

查看 LSF 输出和错误文件：

```bash
cat logs/<JOBID>.out
cat logs/<JOBID>.err
```

终止任务：

```bash
bkill <JOBID>
```

如果 `.err` 文件为空但任务失败，请优先查看：

```bash
logs/<JOBID>.train.log
```

因为 Python 的标准输出和错误都被重定向到了这个文件。

### 5. GPU 编号说明

在 LSF 作业中，系统分配的 GPU 通常会通过 `CUDA_VISIBLE_DEVICES` 映射为当前进程中的 `cuda:0`。因此单卡任务一般使用：

```bash
--gpu 0 --gpu_idx 0
```

不要直接使用物理 GPU 编号，除非明确知道当前作业中的 `CUDA_VISIBLE_DEVICES` 设置。

---

## GitHub 操作说明

本项目使用 GitHub 进行代码协作。请不要直接把实验日志、checkpoint、大数据文件提交到仓库。

### 1. 克隆仓库

在服务器上运行：

```bash
cd ~
git clone https://github.com/ncclab-sustech/TriDim_model.git TriDim_small
cd TriDim_small
```

如果使用 SSH：

```bash
git clone git@github.com:ncclab-sustech/TriDim_model.git TriDim_small
cd TriDim_small
```

私有仓库需要确认自己的 GitHub 账号已经加入项目，并拥有访问权限。

### 2. 设置 Git 用户信息

第一次使用时设置用户名和邮箱：

```bash
git config user.name "Your Name"
git config user.email "your_email@sustech.edu.cn"
```

### 3. 分支说明

本项目采用如下分支策略：

```text
main    稳定版本，只放经过确认的代码
dev     日常开发分支，大家的修改先合并到这里
user/<name> 个人工作分支，适合初学者日常使用
```

简单 ablation 不需要每个都新建分支。推荐做法是：

* 代码实现稳定后，简单 ablation 通过 config 和 bsub 脚本控制；
* 每个组员维护一个长期个人分支，例如 `user/xiaoming`；
* 只有涉及较大模型结构修改时，才新建 `feature/...` 分支。

### 4. 第一次创建自己的分支

```bash
git checkout dev
git pull origin dev
git checkout -b user/<your_name>
git push -u origin user/<your_name>
```

例如：

```bash
git checkout dev
git pull origin dev
git checkout -b user/xiaoming
git push -u origin user/xiaoming
```

### 5. 日常更新代码

每次开始工作前，建议先同步最新 `dev`：

```bash
git checkout user/<your_name>
git fetch origin
git merge origin/dev
```

如果本地有未保存修改，先提交：

```bash
git status
git add .
git commit -m "wip: save current work"
```

然后再同步：

```bash
git fetch origin
git merge origin/dev
```

### 6. 提交自己的修改

查看修改：

```bash
git status
```

添加文件并提交：

```bash
git add <changed_files>
git commit -m "brief description of the change"
```

例如：

```bash
git add models/ configs/ scripts/bsub/
git commit -m "add onlyT ablation script"
```

推送到自己的远程分支：

```bash
git push origin user/<your_name>
```

### 7. 合并到 dev

完成一个功能或实验脚本后，在 GitHub 上创建 Pull Request：

```text
base: dev
compare: user/<your_name>
```

经检查后再合并到 `dev`。不要直接向 `main` 提交代码。

### 8. 常见冲突处理

如果同步时出现冲突，会看到类似：

```text
CONFLICT (content): Merge conflict in models/xxx.py
Automatic merge failed; fix conflicts and then commit the result.
```

处理步骤：

```bash
git status
```

打开冲突文件，找到类似内容：

```text
<<<<<<< HEAD
本地版本
=======
远程版本
>>>>>>> origin/dev
```

手动保留正确代码，并删除 `<<<<<<<`、`=======`、`>>>>>>>` 这些标记。然后执行：

```bash
git add .
git commit -m "resolve conflicts with dev"
git push origin user/<your_name>
```

如果不确定如何解决冲突，不要强行提交，先联系维护者。

### 9. 不要随意使用的命令

初学者不要随便执行以下命令，除非明确知道后果：

```bash
git reset --hard
git clean -fd
git push --force
```

这些命令可能删除本地修改或覆盖远程代码。

### 10. 常见 Git 报错

#### `src refspec main does not match any`

通常表示本地还没有 commit，或者本地没有 `main` 分支。

解决：

```bash
git add .
git commit -m "initial commit"
git branch -M main
git push -u origin main
```

#### `rejected: fetch first`

表示远程分支有本地没有的提交。先拉取再推送：

```bash
git fetch origin
git pull origin main --allow-unrelated-histories
git push origin main
```

日常开发中，推荐先在个人分支工作，再通过 Pull Request 合并到 `dev`。

### 11. 提交前检查

提交前请确认：

* 没有提交 `logs/`、`checkpoints/`、数据文件或缓存文件；
* 没有硬编码自己的绝对路径；
* 新模型或新实验有对应 config 或 bsub 脚本；
* 代码至少能正常 import；
* 重要实验记录了 Git commit、模型名、数据集、seed 和日志路径。
