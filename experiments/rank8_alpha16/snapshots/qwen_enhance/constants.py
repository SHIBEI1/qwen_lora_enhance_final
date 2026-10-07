"""Stable experiment-wide constants.

The prompt is deliberately fixed: inference receives only an input image from the
caller, while this instruction is supplied internally for every train and infer run.
"""

from __future__ import annotations

FIXED_ENHANCEMENT_PROMPT = (
    "对输入图像进行保真质量增强。在严格保持原始场景内容、物体、几何结构、位置、文字、材质和真实细节不变的前提下，"
    "恢复运动模糊、低照度、噪声、低对比度和色偏造成的图像质量损失；提升清晰度、局部纹理可见性、自然亮度、动态范围、"
    "对比度和真实色彩。不得添加、删除、替换、移动或重绘任何对象与细节；不得生成虚构纹理；避免过度锐化、过度平滑、"
    "过曝、过饱和和不自然的人工痕迹。"
)

# Qwen image-edit pipelines require a negative prompt whenever CFG is used.
NEGATIVE_PROMPT = " "

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
MODEL_VERSION = "edit-2511"
MUSUBI_TAG = "v0.2.15"

EXPECTED_SPLIT_COUNTS = {"train": 640, "val": 100, "test": 200}
EXPECTED_TASK_COUNTS = {
    "train": {"deblur": 320, "lowlight": 320},
    "val": {"deblur": 50, "lowlight": 50},
    "test": {"deblur": 100, "lowlight": 100},
}
