# GitHub 交付与完整还原

本项目约 55.25 GiB。模型最大单个文件约 10 GB，不能作为普通 Git 文件上传。因此同一个 GitHub 仓库采用两个交付部分：

- **Code 文件列表**：可浏览的项目源码、配置、说明、数据清单和后端源码，不含大模型、LoRA、图像数据、离线缓存与后端内部 `.git`。
- **[v1.0.0 Release](https://github.com/SHIBEI1/qwen_lora_enhance_final/releases/tag/v1.0.0)**：完整原项目的未压缩 tar 分卷、逐文件及逐分卷 SHA-256 清单。模型、03 权重、全部数据集、隐藏 `.cache`、后端 `.git`、日志和空目录都包含在内。原项目文件没有改动。

**不要直接从源码 ZIP 或普通 clone 启动训练、安装或推理。先还原完整发布包。** GitHub 自动生成的 `Source code (zip)` / `Source code (tar.gz)` 不是完整发布包。

## 推荐：自动下载并校验还原

仅需 Python 3.11 或更高版本；无需 GitHub 密码，不依赖第三方 Python 包。准备约 112 GiB 可用空间，下载分卷与还原后的项目各占约 56 GiB。

```bash
git clone https://github.com/SHIBEI1/qwen_lora_enhance_final.git
cd qwen_lora_enhance_final

# 选择一个存放完整项目的新父目录，不要覆盖已有的开发备份或源码目录。
python scripts/restore_release.py --destination /mnt/Disk1/kl/github_restored

cd /mnt/Disk1/kl/github_restored/qwen_lora_enhance_final
bash scripts/setup_environment.sh
bash scripts/prepare_dataset.sh
bash scripts/infer.sh --input /path/to/input.png --output outputs/inference/result.png
```

还原程序校验 GitHub 返回的 SHA-256、全部分卷、组合后的 tar 和每一个还原文件；不接受链接或越界路径，不覆盖已有项目目录。下载中断后可执行同一命令，已经下载且校验通过的分卷将复用。

下载分卷默认保存在目标父目录的 `_github_release_parts/`。检查还原项目后可自行移走这些下载分卷，程序不会自动删除它们。

## 手动下载并还原（Linux）

从 Release 下载所有 `qwen_lora_enhance_final.tar.partNNN` 附件、`SHA256SUMS` 和 `FULL_PROJECT_MANIFEST.json`，放在同一目录。不要遗漏任何一个分卷。

```bash
sha256sum --check SHA256SUMS

# 在一个没有同名项目的新目录中运行；按三位序号拼接，不单独解压某个分卷。
cat qwen_lora_enhance_final.tar.part[0-9][0-9][0-9] | tar -xf -

cd qwen_lora_enhance_final
bash scripts/verify_release.sh
```

完整目录内原有 `release_checksums.sha256` 可独立核对冻结的模型、03 权重、数据集和代码。移动目录后运行 `prepare_dataset.sh` 重建绝对路径 metadata。Windows 可用于保存文件；CUDA 训练和推理按照原项目 Linux 环境执行。

## 版本与限制

基础模型为 `Qwen/Qwen-Image-Edit-2511`，revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`；LoRA 为 epoch 03 / step 1920，SHA-256 `a7d603fe5e61af25d7783fc6980a84709899cf480f65bb1867a955e0c8a7ce62`。

原数据划分及已知低光重复内容均原样保留，未重新抽样或去重。相关评测局限见数据说明。本仓库未改变 Qwen、Musubi 或原始数据集的授权条件；后续使用、分发应遵守相应原始许可。没有将第三方资源重新授权为本项目所有。
