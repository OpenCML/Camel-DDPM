# COCO 2017 接入

优先使用整理好的图片和人工描述，减少逐个网页链接获取的等待。官方提供训练图片约 118K、验证图片 5K，以及标注压缩包，见 [官方下载页](https://cocodataset.org/#download) 和 [下载页源码](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)。使用说明见 [官方条款](https://cocodataset.org/#termsofuse)；保留标注中的来源与 license 字段，不将整套图片视为统一许可。

`tools/fetch-coco-archives.py` 在本机下载、断点续传，再传往服务器 `/mnt/data3/zjwei/Camel-DDPM-data/coco2017/archives`。实际传输使用公开 COCO 存储桶的 HTTPS S3 路径，避免 HTTP 与图片域名的 TLS 差异。每个文件核对对象长度、计算 SHA-256，服务器核对一致后才去掉 `.incoming` 后缀。原始 URL、下载时间、HTTP 元数据和哈希另存为清单。当前启动的顺序为标注、验证图片、训练图片；这不改变训练/验证的用途。

```powershell
python tools/fetch-coco-archives.py
```

本机不解码图片、不训练。服务器后续执行压缩包 CRC、图片解码、描述非空/关联检查和重复检查，保留官方训练/验证划分；同一图片的五条描述不能跨划分。这些是自动数据校验，不要求逐张人工审核。

三个压缩包已完整传至服务器。使用以下命令封存训练数据（仅在服务器执行）：

```bash
python tools/prepare-coco.py \
  --archives /mnt/data3/zjwei/Camel-DDPM-data/coco2017/archives \
  --out /mnt/data3/zjwei/Camel-DDPM-data/coco2017/sealed-rgb64-v1
```

工具先复核传输清单的 SHA-256，再读取标注并校验 ZIP CRC。每张被标注的图片在读取时校验 CRC、完整解码、核对尺寸，以 EXIF 方向校正、RGB 转换、中心裁剪和 Lanczos 缩放产生 64×64 像素。输出保留原始 ZIP 成员定位、图片 license、来源 URL 和全部非空描述。训练由 Camel 随机选择该图的一条描述，验证固定使用第一条；同图描述共享划分。

按官方验证集、训练集顺序处理，去掉原始 JPEG 或处理后 RGB64 完全重复的图片，优先保留验证图片。排除项逐条记录。此版本不声称完成近似重复检测。导入期间写入 `.pending` 目录，全部成功后才封存并改名；损坏图片或缺失标注会中止，不会静默忽略。`--limit-per-split 32` 用于先手动检查接口，不是正式数据集。

COCO 是下一阶段扩大训练和衡量图文学习的基础数据，英文覆盖和数量本身仍不足以支持生产级通用中英文目标；WIT 作为后续多语言补充。当前 RGB64 也是现有小模型的输入规格，不能据此声称已具备高分辨率生产能力。

## 服务器基线实验

2026-10-19，先手动导入每个划分 32 张图片，得到 64 张独立图片和 320 条描述。物理 GPU 2 上的 Camel VAE 两步训练损失为 `0.3179970307283277`、`0.23560036841630538`，梯度范数为 `0.07875252483192208`、`0.08122877845989863`，检查点成功发布。此项仅验证新数据格式和多描述采样接口。

完整导入与 VAE 基线由以下服务器任务执行，日志、状态和检查点均在数据盘，断开 SSH 不会终止任务：

```bash
python -u tools/run-coco-baseline.py \
  --dataset /mnt/data3/zjwei/Camel-DDPM-data/coco2017/sealed-rgb64-v1 \
  --out /mnt/data3/zjwei/Camel-DDPM-data/project-runs/coco-baseline-v1 \
  --gpu 2 --steps 2000 --batch 16
```

任务先完成全部导入，再从随机权重执行 2,000 步 VAE 更新，每 100 步保存检查点。之后分别评估同样 128 张验证图片的随机初始化和训练结果，输出重建 MSE、潜变量矩及四组重建对照图片。累计 32,000 次随机图像抽取不等于遍历一遍训练集。任务不会根据训练 loss 自动宣称模型质量，也不会在未检查 VAE 重建效果前自动开始扩散训练。

实时状态位于 `project-runs/coco-baseline-v1/status.json`，完整日志位于 `project-runs/coco-baseline-v1.log`。`complete` 指本次基线实验完成，`production_ready` 保持 `false`。

本次任务已完成：封存 **118,272 张训练图、5,000 张验证图、616,692 条描述**，排除 14 张原始文件重复和 1 张 RGB64 重复图片。清单 SHA-256 为 `bc6f632dbeb84487e776e69883b23b49d837aeea349b12bb826b56c5b2f50dd4`，完整报告见 [封存摘要](results/coco-sealed-summary.json)。

VAE 在同样 128 张验证图上的重建 MSE 从 `0.31285463916758244` 降到 `0.049561733221832036`，结果见 [基线评估](results/coco-vae-baseline.json)。检查四组原图/重建对照后，确认模型能保留主要色块与粗略布局，但人物、杯子和纹理细节明显丢失。当前小型编码器不能作为生产质量验收通过的依据，也不据此自动启动生产扩散训练。训练后潜变量二阶矩为 `4.78432468950682`，后续必须明确潜变量尺度及扩散日程，不能默认其已接近单位方差。
