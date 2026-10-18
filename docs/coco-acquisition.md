# COCO 2017 接入

优先使用整理好的图片和人工描述，减少逐个网页链接获取的等待。官方提供训练图片约 118K、验证图片 5K，以及标注压缩包，见 [官方下载页](https://cocodataset.org/#download) 和 [下载页源码](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)。使用说明见 [官方条款](https://cocodataset.org/#termsofuse)；保留标注中的来源与 license 字段，不将整套图片视为统一许可。

`tools/fetch-coco-archives.py` 在本机下载、断点续传，再传往服务器 `/mnt/data3/zjwei/Camel-DDPM-data/coco2017/archives`。实际传输使用公开 COCO 存储桶的 HTTPS S3 路径，避免 HTTP 与图片域名的 TLS 差异。每个文件核对对象长度、计算 SHA-256，服务器核对一致后才去掉 `.incoming` 后缀。原始 URL、下载时间、HTTP 元数据和哈希另存为清单。当前启动的顺序为标注、验证图片、训练图片；这不改变训练/验证的用途。

```powershell
python tools/fetch-coco-archives.py
```

本机不解码图片、不训练。服务器后续执行压缩包 CRC、图片解码、描述非空/关联检查和重复检查，保留官方训练/验证划分；同一图片的五条描述不能跨划分。这些是自动数据校验，不要求逐张人工审核。

当前仅开始下载和传输，不把压缩包大小、描述数或队列数计为训练图片数。COCO 是下一阶段扩大训练和衡量图文学习的基础数据，英文覆盖和数量本身仍不足以支持生产级通用中英文目标；WIT 作为后续多语言补充。
