# 数字图像处理 Lab05 图像压缩

在解压后的目录打开 PowerShell 或终端，运行以下命令。需要 Python 3.14，若本机已经安装所列版本的依赖，可直接运行第二行。

```powershell
python -m pip install -r requirements.txt
python lab05.py
```

脚本从同目录的 `camera_clean.png` 读取 512×512 灰度原图。原图来自此前的数字图像处理实验素材。每次运行都会在 `results` 中重建以下文件，并在终端打印数据。

| 文件 | 内容 |
| --- | --- |
| `entropy_compare.png` | 自然图、双灰度图、均匀噪声图及其灰度概率直方图 |
| `jpeg_compare.png` | 原图和五档 JPEG 的全图与同一局部对照 |
| `rate_distortion.png` | 五档 JPEG 的 bpp–PSNR 曲线 |
| `dct_quant_compare.png` | 两个 8×8 块的原图、DCT 系数热图与四档量化重建 |
| `jpeg_q10.jpg`、`jpeg_q30.jpg`、`jpeg_q50.jpg`、`jpeg_q70.jpg`、`jpeg_q90.jpg` | 重新编码并解码测量的 JPEG 文件 |
| `lossless.png` | 用于逐像素验证与压缩指标计算的 PNG 文件 |
| `measurements.txt` | 汇总指标、像素矩阵、DCT 系数矩阵及未经量化的重建误差 |

本次使用 Python 3.14.7、NumPy 2.5.4、Pillow 13.0.0 和 Matplotlib 3.11.2。随机噪声种子为 `20261127`。JPEG 和 PNG 文件大小由本机 Pillow 编解码环境实测，其他构建的编解码库可能产生略有差异的文件字节数。
