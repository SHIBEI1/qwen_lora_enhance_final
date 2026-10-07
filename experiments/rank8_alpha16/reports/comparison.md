# Rank 8 / alpha 16 versus rank 16 / alpha 32

Primary comparison: matched epoch 3 / step 1920. Secondary: epoch 6 / step 3840.

| Epoch | Task | r8 PSNR | r16 PSNR | Delta PSNR | r8 SSIM | r16 SSIM | Delta SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | deblur | 14.2188 | 14.2510 | -0.0322 | 0.3552 | 0.3580 | -0.0028 |
| 3 | lowlight | 16.1275 | 17.5837 | -1.4562 | 0.5509 | 0.5931 | -0.0422 |
| 6 | deblur | 14.2583 | 14.2575 | +0.0008 | 0.3570 | 0.3566 | +0.0004 |
| 6 | lowlight | 15.5535 | 15.9358 | -0.3823 | 0.5029 | 0.5284 | -0.0255 |

All settings other than LoRA rank and alpha were held fixed; both configurations have alpha/rank=2.
Training data stayed unchanged. Evaluation excludes exact decoded RGB overlap with any training input/target.
This is a six-image, single-seed validation experiment. Lowlight near-duplicate scenes are not certified absent.
Original input metrics and per-image differences are retained in comparison.json; visually inspect the generated panels.
No final release adapter is automatically replaced by this experiment.
