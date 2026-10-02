# 终版运行手册

所有 shell 入口自动定位项目根目录，模型和数据路径由该目录派生。默认 Conda 环境为 `myenv`，CUDA 依赖按照 `environment/requirements-lock.txt` 中的已安装版本重建。

## 推理

```bash
bash scripts/infer.sh --input /path/to/input.png --output outputs/inference/enhanced.png
```

默认适配器为 `weights/qwen_image_enhance_lora-000003.safetensors`。推理只需输入图像，固定文本条件由程序提供。`logs/inference.log` 保存本次完整终端输出，图像旁边的 JSON 记录实际推理配置。

也可在激活环境后直接使用 Python：

```bash
source "$HOME/miniforge3/bin/activate" myenv
python scripts/enhance.py --input /path/to/input.png --output outputs/inference/enhanced.png
```

## 部署与重新训练

```bash
bash scripts/setup_environment.sh
bash scripts/prepare_dataset.sh
bash scripts/train_lora.sh
```

项目已携带基础模型，日常运行无需执行 `download_models.sh`；该脚本仅用于模型资产丢失后按固定版本恢复。移动项目后应重新运行数据准备，训练入口也会自动执行该步骤。

发布目录不包含旧的 latent / embedding 缓存。首次重新训练会生成缓存，耗时不属于已选权重的推理时间。新的训练输出写入 `outputs/lora/`，不覆盖终版 `weights/` 中的适配器。

## 迁移完整性

```bash
bash scripts/verify_release.sh
```

该命令按 `release_checksums.sha256` 校验发布文件，写入 `logs/verify_release.log` 与 `artifacts/reports/release_integrity.json`。它核验复制是否完整，不执行图像质量评测。完整模型较大，校验可能需要数分钟。

请一并复制 `.cache/huggingface/`，其中是后端按原始 Qwen repo ID 查找的 tokenizer 和 processor。这些资源已展开为真实文件。
