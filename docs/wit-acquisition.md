# WIT 图片接入流程

本机只进行网络下载、哈希校验与文件中转。图像解码、质量检查、索引及训练留在服务器。
元数据传输见 [生产训练计划](production-training.md)；图片元数据使用官方
[MediaWiki Imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo)。

1. 服务器 `index-wit.py` 筛出中英文候选，SQLite 按图片 URL/描述去重。
2. `export-wit-queue.py` 交替选择中英文图片，按图片去重，并保留同图的全部候选描述。
3. 本机 `fetch-wit-assets.py` 获取 Commons 原始元数据及官方图片/缩略图，保留作者、许可证、来源页面、下载 URL、取回时间与 SHA-256。
4. 每批 tar 在服务器核对 SHA-256 后解包。严格限制 HTTPS 官方来源；缩略图域名包含 `thumb.wikimedia.org`。
5. 服务器 `audit-wit-assets.py` 核对文件清单、哈希、图像解码和实际尺寸，检查许可证声明的一致性，记录 dHash 和仍需复核的条目。
6. 仍需描述质量检查、跨来源重复分组和不可变训练/评估分片；审核输出不会自动成为训练输入。

已实际完成首批 16 张的下载和传输，无下载失败；随后启动了 1000 张的批次。
一次后续快照检查覆盖 128 张图片：全部能解码，英文描述 65 对、中文 63 对；117 张通过机器许可证声明检查，11 张保留复核。
**这不是 117 张已准入训练图片，也不构成法律清权结论。** 所有原始元数据保留，训练准入数仍为零。
抽样预览包含人物、建筑、雕塑、食物、城市场景和物品，覆盖面比此前博物馆试验更广；尚未完成全量描述匹配或分布评估。

```bash
# 服务器导出下载队列；输出目录/文件应全新。
cd /home/zjwei/projects/Camel-DDPM
python3 tools/export-wit-queue.py \
  --index /mnt/data3/zjwei/Camel-DDPM-data/wit/index-1percent-v1/candidates.sqlite \
  --out runs/wit-queue-1000.json --limit 1000
```

```powershell
scp 88zjwei:/home/zjwei/projects/Camel-DDPM/runs/wit-queue-1000.json data/wit/queue-1000.json
python tools/fetch-wit-assets.py --queue data/wit/queue-1000.json --out data/wit/assets --limit 1000 --batch 32
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n10 \
  /home/zjwei/camel-eval/venv-bench/bin/python tools/audit-wit-assets.py \
  --source /mnt/data3/zjwei/Camel-DDPM-data/wit/assets/images \
  --out /mnt/data3/zjwei/Camel-DDPM-data/wit/assets/audit-next
```

每个图片缓存只允许一个下载进程写入；传输失败会中止，不会把失败算作完成。
五次连续获取失败会停止，保留缓存供检查/重试。失败条目与成功清单分开保存。
当前图片批次没有实现并行下载或自动训练准入，不能据此宣称已经具有百万图片的交付吞吐。

服务器 `index-transferred-wit.py` 可在当前任务内等待已校验元数据分片到达，再顺序建立索引；最长等待 12 小时。
已存在但未完成的索引不会被覆盖，而是等待其原进程完成。状态保存在数据目录 `index-progress.json`。
新索引采用 WAL、有限缓存及周期进度输出，避免每批随机键写入都同步较大的回滚日志。
