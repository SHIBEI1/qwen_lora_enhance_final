# Qwen LoRA Enhance Final

本项目是本次图像增强实验的终版交付目录。唯一基础模型为 **Qwen-Image-Edit-2511**，只训练其 DiT 中的 LoRA。默认适配器固定为用户选定的 **epoch 3 / step 1920**，不是第 6 轮最后保存的权重。

## 直接使用

当前远程机器已经具备 `myenv` 环境。进入本项目目录后执行：

```bash
bash scripts/infer.sh \
  --input /path/to/low_quality.png \
  --output outputs/inference/enhanced.png
```

无需提供 `--lora` 或文本提示词。程序默认读取 `weights/qwen_image_enhance_lora-000003.safetensors`，并内部注入固定的图像质量增强提示词。输出与输入图尺寸一致；旁边的 `.png.json` 文件记录权重、种子、步数与推理参数。

默认推理参数为 25 步、seed 3407、guidance 4.0、16 个 CPU block swap。3090 上曾测得单张图约 12 分钟，时间受图像尺寸、模型加载与主机内存影响。

## 目录

```text
dataset/                       原样复制的成对图像、划分清单与数据说明
weights/                       唯一发布的 epoch-03 LoRA
models/hf/Qwen-Image-Edit-2511/  本次使用的完整基础模型快照
.cache/huggingface/             运行必需的 tokenizer / processor 离线资源
third_party/musubi-tuner/       固定版本的训练和推理后端，含缓存键修补
qwen_enhance/                  固定提示词、默认权重、路径和日志实现
configs/                       3090 训练及单 GPU 配置
scripts/                       环境、数据准备、缓存、训练、推理与完整性检查
environment/                   当前已验证环境的版本清单
patches/                       本次训练必需的缓存键补丁
docs/                          使用手册与架构说明
logs/                          每个流程的最新带时间戳日志
outputs/                       后续运行时生成的图像或新训练权重
artifacts/                     可重新生成的数据 metadata 与特征缓存
release.json                   发布权重、模型版本和数据集记录
release_checksums.sha256        发布文件的完整性清单
```

模型、数据集和离线资源均为项目内的真实文件，未使用符号链接或指向开发备份的硬链接。`outputs/` 中未来重新训练生成的权重与 `weights/` 中的终版权重分开保存。

## 迁移到其他 Linux 训练机器

请复制整个目录，包含隐藏目录 `.cache/`，不要仅复制可见文件。项目使用 Linux + NVIDIA CUDA；Windows 目录可作为存储和传输副本。

```bash
# 新机器需要联网安装 Python/CUDA 包；模型已经随项目携带，无需重下。
bash scripts/setup_environment.sh

# 数据 metadata 含绝对路径，移动目录后重新生成即可。
bash scripts/prepare_dataset.sh

# 单图推理
bash scripts/infer.sh --input /path/to/input.png --output outputs/inference/result.png

# 可选：校验搬运后的全部发布文件，不会启动训练或图像生成。
bash scripts/verify_release.sh
```

默认环境位置为 `$HOME/miniforge3/envs/myenv`。若 Conda 安装在其他位置，设置 `QDE_CONDA_ROOT=/path/to/conda`；若环境名不同，设置 `QDE_ENV_NAME=your_env`。直接使用 Python 入口前需激活该环境。

训练与推理优先加载本项目携带的后端源代码。原机器的 `myenv` 可以复用，其原有 editable 安装路径不会成为终版的运行依赖。

## 保留的训练流程

```bash
# 重新训练：先重建 metadata 和缓存，再按原 6 轮训练配置执行。
bash scripts/train_lora.sh

# 若只复现至所选第 3 轮，可显式设定训练轮数。
QDE_EPOCHS=3 bash scripts/train_lora.sh
```

训练配置：640 对样本，运动模糊与低光各 320 对；512 级长宽比桶，batch 1；LoRA rank 16 / alpha 32 / dropout 0.05；学习率 5e-5；BF16、FP8 基础权重、梯度检查点、16 个 CPU block swap；随机种子 3407。正式开发运行共训练 6 轮，发布的是其中第 3 轮参数。

原实验缓存和机器路径未随包复制。数据准备和缓存入口根据当前项目根目录重新生成它们。不同 GoPro 视频包含同名帧，必须保留 `patches/musubi_unique_cache_keys.patch` 和自动补丁检查入口。

## 日志与数据记录

训练写入 `logs/train.log`，推理写入 `logs/inference.log`；数据准备、独立缓存和完整性检查分别写入同名固定日志。标准输出与错误输出附带本机时区时间戳，每次运行覆盖该流程的旧日志。完整训练的子步骤统一进入 `train.log`。

数据集冻结保留本次训练用的 640 train / 100 val / 200 test 对，不在打包过程中重新抽样或去重。GoPro 按场景隔离；低光部分已发现不同文件名的内容重复，包括验证 `579.png` 与训练 `68.png` 的输入和目标图像素完全一致。因此该低光划分不能被描述为无内容泄漏的严格评测集。详见 `dataset/DATASET_CARD.md`。

本终版只固定所选权重与可迁移的运行文件；历次 smoke、pilot、候选筛选结果和训练日志在原 `qwen_diffusion_enhance` 开发目录完整保留。


## GitHub 完整项目下载

此 Git 仓库提供可浏览的源码和配置；**普通 `git clone` 或 GitHub 的 Source code ZIP 不包含模型、LoRA、图像数据和完整离线资源，也不包含后端的内部 Git 元数据**。全部原始项目文件保存在本仓库 `v1.0.0` Release 的分卷附件中。请按 [GitHub 交付说明](docs/GITHUB_DELIVERY.md) 下载并还原完整目录后运行。完整发布包上传校验完成后才会公开 Release。
