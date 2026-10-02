# Qwen LoRA Enhance Final — Frozen Dataset

本目录保留此次正式 LoRA 训练所用的数据版本，打包未调整样本或划分。

| 划分 | 对数 | GoPro 去模糊 | 低光增强 |
| --- | ---: | ---: | ---: |
| train | 640 | 320 | 320 |
| val | 100 | 50 | 50 |
| test | 200 | 100 | 100 |

`images/` 内所有输入与目标均为实体复制的原始 RGB PNG。`manifests/*.jsonl` 的 `input` 和 `target` 字段相对本目录定位；`source_input`、`source_target` 中旧 Windows 路径只记录来源，不参与运行。

GoPro 为 1280×720 配对图像，按场景隔离 train/val，test 来自原始 GoPro test。四个验证场景为 `GOPR0372_07_00`、`GOPR0384_11_03`、`GOPR0857_11_00`、`GOPR0868_11_02`，均不出现在本项目训练场景中。

低光为 600×400 配对图像，train/val 从原数据 train 按输入亮度分层划分；test 从原数据 val 抽取。文件路径和文件名不同不等于图像内容不同。

## 已知的内容重复

低光原数据包含重名之外的内容重复。本次开发已经确认验证 `lowlight_val_0016 / 579.png` 与训练 `lowlight_train_0324 / 68.png` 的输入和目标 RGB 像素完全一致。全数据检查还发现 train/val 和 train/test 的其他重复。因此低光验证和测试划分不能作为保证无内容泄漏的独立泛化基准。

终版按用户要求结束优化与测试，保留实际训练数据，以便追溯发布适配器。未来如需科研评测，应先按解码像素及场景进行去重和划分；终版不声称已完成这一步。

## 导入与迁移

数据导入以清单为准，使用当前项目根目录解析路径。训练使用 512 级长宽比桶及配对目标/控制图处理；避免独立裁剪导致内容不对齐，避免自动亮度归一化消除低光退化信号。原始图像不作永久缩放。

移动项目后通过 `bash scripts/prepare_dataset.sh` 重建 Musubi metadata。原数据源特征记录在 `reports/source_profile.json`，抽样种子为 3407。
