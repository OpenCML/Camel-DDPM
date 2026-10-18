# 从零训练的完整小型潜空间模型

该路径在 `88zjwei` 使用项目自己的 `.camel/`，固定空闲的物理 GPU 2。
本机只编辑、传输代码。所有学习组件随机初始化，不加载预训练权重。
它补齐了结构与训练流程，当前 64×64 模型和博物馆数据仍不足以成为生产级通用文生图。
本轮两个阶段各 500 步的指标、重建和生成样图见 [服务器验证记录](results/server-latent-validation.md)。

## 结构与精度

| 组件 | 实现 |
|---|---|
| 图像 VAE | RGB64 → 64 个 8×8 patch → 宽度 128，2 层编码 Transformer → 8×8×4 高斯潜变量；2 层解码 Transformer 重建 RGB |
| VAE 目标 | 像素 MSE + 1e-4 KL，无预训练感知损失 |
| 文本编码器 | 128 个 UTF-8 byte 位置，259 项 one-hot，2 层 Transformer；与去噪器联合训练 |
| 去噪器 | 宽度 128、4 头、4 层，含自注意力、文本交叉注意力、时间投影、SwiGLU、RMSNorm，预测潜变量噪声 |
| 参数量 | VAE 1,116,864；文本编码器和去噪器共 1,906,180 |
| 更新 | Camel 自动微分、全局梯度范数裁剪 1、Adam、EMA 0.999、非有限数检查 |
| 采样 | 独立加载两个模型，DDIM eta=0、classifier-free guidance、VAE 解码、Camel 输出 PPM/NPY |

核心实现为 [networks.cml](../src/networks.cml)。Python 仅处理网络下载、图像 codec、文件和进程管理，以及生成静态参数类型；不执行模型、自动微分或优化器。
生成文件可用 `python3 tools/generate-model-params.py --check` 核对，参数初始化本身仍由 Camel 执行。

仅 `networks.mm` 显式授予 `reassociate, change_precision, finite_math`，Tiles 才选择动态加载的 cuBLAS `COMPUTE_32F_FAST_16BF` 路径。
它将矩阵乘操作数转换为 BF16，FP32 累加并输出。缺失任一授权时保留默认 FP32 FMA 路径；需要 Ampere 或更新 GPU。
图边界、激活存储、主权重、归约、Adam 和 EMA 保留 FP32；反向传播尚未全面使用 BF16。
这是实际 BF16 计算，但不是全局半精度存储 AMP，也不承诺每一种小矩阵都有吞吐提升。
原理参见 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/archive/12.2.0/cublas/)。

## 数据与恢复

新增不可变数据集 `data/openaccess-256-v1`：256 张 Smithsonian CC0 图，247 张训练、9 张验证，174 个分组。
按藏品 ID、图像 SHA-256 和 64 位 dHash 距离 ≤3 合并分组，再按组划分集合。
封存前核对元数据和媒体许可证、原图/RGB 哈希；训练启动时核对 manifest、像素和文字。
标题和主题来自英文博物馆元数据，并非人工稠密描述；没有中文图文数据，不能验证中文对齐。
验证集也很小，分组哈希划分不保证恰好 10%。

每个 checkpoint 保存 model/Adam m/Adam v/EMA 四棵树、全局步数及配置身份。
按全局步数重新设定随机种子，进程锁、原子发布、fsync 和逐张量校验保护恢复。
VAE 176 个张量文件，去噪器 344 个；连续 4 步与 2+2 步恢复均逐字节一致，损坏张量会被拒绝。
配置绑定源文件、运行时二进制和标准库、数据、GPU UUID、超参数及冻结 VAE 的具体 checkpoint。
更新源码或运行时后须开新运行；旧原型应使用其历史源码和运行时复现。
CUDA 驱动和系统 cuBLAS 版本仍应随服务器环境保留，当前身份校验未覆盖这些系统依赖。

## 服务器命令

```bash
cd /home/zjwei/projects/Camel-DDPM
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/zjwei/camel-eval/venv-bench/bin/python
$PY tools/prepare_openaccess.py --out data/openaccess-256 --units saam nasm --shards-per-unit 8 --limit 256
$PY tools/seal-corpus.py data/openaccess-256 data/openaccess-256-v1
$PY tools/latent.py train --stage vae --dataset data/openaccess-256-v1 --out runs/vae-256-mixed-v1 --steps 500 --checkpoint-every 100
$PY tools/latent.py evaluate --run runs/vae-256-mixed-v1 --out runs/vae-256-initial --initial
$PY tools/latent.py evaluate --run runs/vae-256-mixed-v1 --out runs/vae-256-valid
$PY tools/latent.py train --stage diffusion --dataset data/openaccess-256-v1 --autoencoder runs/vae-256-mixed-v1 --out runs/ld-256-mixed-v1 --steps 500 --checkpoint-every 100
$PY tools/latent.py evaluate --run runs/ld-256-mixed-v1 --out runs/ld-256-initial --initial
$PY tools/latent.py evaluate --run runs/ld-256-mixed-v1 --out runs/ld-256-valid
$PY tools/latent.py sample --run runs/ld-256-mixed-v1 --out runs/ld-256-sample --prompt 'A bird on a branch' --steps 30 --guidance 2
# 目标步数是总步数，恢复时源代码、运行时、数据和超参数必须匹配。
$PY tools/latent.py train --stage diffusion --dataset data/openaccess-256-v1 --autoencoder runs/vae-256-mixed-v1 --out runs/ld-256-mixed-v1 --steps 1000 --checkpoint-every 100 --resume
```

输出目录要求全新。默认 batch=2、GPU=2，失败会报错而不会退回 CPU。
每个 checkpoint 段会重启 Camel，计时包含初始化、编译和落盘；这不是最终高吞吐训练调度器。
还缺更大、更广且含中文的图文集、高分辨率训练、激活重计算/累积、分布式训练、全面生成质量评估和服务化验证。
潜变量分布校准、VAE 的视觉细节和文字条件对齐也必须由真实评估决定，不能用训练损失下降代替。
