"""运行数字图像处理 Lab05 的三项实验。"""

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
JPEG_QUALITIES = (10, 30, 50, 70, 90)
QUANT_STEPS = (5, 10, 20, 40)
NOISE_SEED = 20261127
TWO_LEVELS = (64, 192)
ROI = (160, 128, 160, 160)  # 横坐标、纵坐标、宽、高
BLOCKS = (("Smooth", 112, 40), ("Edge", 176, 144))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def entropy_experiment(original, output):
    """计算直方图、概率、熵和理论压缩比。"""
    two_level = np.full_like(original, TWO_LEVELS[0])
    two_level[:, original.shape[1] // 2 :] = TWO_LEVELS[1]
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.integers(0, 256, size=original.shape, dtype=np.uint8)
    samples = (("Natural image", original), ("Two gray levels", two_level), ("Uniform noise", noise))
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), layout="constrained")
    for index, (label, pixels) in enumerate(samples):
        counts = np.bincount(pixels.ravel(), minlength=256)
        probabilities = counts / pixels.size
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log2(positive)))
        ratio = 8 / entropy if entropy > 0 else None
        rows.append({
            "label": label,
            "levels": int(np.count_nonzero(counts)),
            "entropy": entropy,
            "ratio": ratio,
        })

        axes[0, index].imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, index].set_title(label, fontweight="bold")
        axes[0, index].axis("off")
        histogram = axes[1, index]
        histogram.bar(np.arange(256), probabilities * 100, width=1, color="#245581")
        histogram.set(xlim=(-1, 256), xticks=[0, 64, 128, 192, 255], xlabel="Gray level")
        histogram.set_ylabel("Pixel probability (%)")
        histogram.set_title(f"H = {entropy:.4f} bit/pixel")
        histogram.grid(axis="y", alpha=0.22)
        histogram.set_axisbelow(True)

    fig.savefig(output / "entropy_compare.png", dpi=190)
    plt.close(fig)
    return rows


def pixel_mse(reference, decoded):
    # 减法前转成浮点，防止 uint8 下溢。
    residual = reference.astype(np.float64) - decoded.astype(np.float64)
    return float(np.mean(residual ** 2))


def compression_experiment(original, output):
    """保存并重新解码真实文件，再根据文件大小和解码像素计算指标。"""
    source = Image.fromarray(original)
    variants = []
    for quality in JPEG_QUALITIES:
        target = output / f"jpeg_q{quality}.jpg"
        source.save(target, format="JPEG", quality=quality)
        variants.append((f"JPEG Q={quality}", quality, target))
    png = output / "lossless.png"
    source.save(png, format="PNG")
    variants.append(("PNG", None, png))

    rows = []
    comparisons = [("Original", original)]
    for label, quality, target in variants:
        with Image.open(target) as image:
            decoded = np.array(image.convert("L"))
        if decoded.shape != original.shape:
            raise ValueError(f"解码尺寸变化：{target.name}")
        size = target.stat().st_size
        mse = pixel_mse(original, decoded)
        identical = bool(np.array_equal(original, decoded))
        psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else math.inf
        rows.append({
            "label": label,
            "quality": quality,
            "name": target.name,
            "bytes": size,
            "cr": original.size / size,
            "bpp": 8 * size / original.size,
            "mse": mse,
            "psnr": psnr,
            "identical": identical,
        })
        if quality is not None:
            comparisons.append((f"Q = {quality}", decoded))

    jpeg_comparison_figure(comparisons, output)
    jpeg_rows = rows[:-1]
    fig, ax = plt.subplots(figsize=(7.8, 4.0), layout="constrained")
    ax.plot([row["bpp"] for row in jpeg_rows], [row["psnr"] for row in jpeg_rows],
            color="#245581", marker="o", linewidth=2, markersize=6)
    for row in jpeg_rows:
        ax.annotate(f'Q = {row["quality"]}', (row["bpp"], row["psnr"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
    ax.set(xlabel="Bit rate (bit/pixel)", ylabel="PSNR (dB)", title="JPEG rate–distortion curve")
    ax.grid(alpha=0.25)
    ax.margins(x=0.11, y=0.16)
    fig.savefig(output / "rate_distortion.png", dpi=190)
    plt.close(fig)
    return rows


def jpeg_comparison_figure(comparisons, output):
    x, y, width, height = ROI
    fig, axes = plt.subplots(2, len(comparisons), figsize=(12.0, 4.3), layout="constrained")
    for index, (label, image) in enumerate(comparisons):
        overview = axes[0, index]
        overview.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        overview.add_patch(Rectangle((x - 0.5, y - 0.5), width, height, fill=False,
                                     edgecolor="#ec9b19", linewidth=1.05))
        overview.set_title(label, fontweight="bold")
        overview.axis("off")
        closeup = axes[1, index]
        closeup.imshow(image[y : y + height, x : x + width], cmap="gray",
                       vmin=0, vmax=255, interpolation="nearest")
        closeup.axis("off")
    fig.savefig(output / "jpeg_compare.png", dpi=190)
    plt.close(fig)


def dct_basis():
    """构造 8 点正交归一化 DCT-II 基矩阵。"""
    indices = np.arange(8, dtype=np.float64)
    basis = np.sqrt(2 / 8) * np.cos(np.pi / 8 * indices[:, None] * (indices + 0.5))
    basis[0] = np.sqrt(1 / 8)
    return basis


def dct_experiment(original, output):
    """使用同一输入块比较量化步长、系数稀疏性和重建误差。"""
    basis = dct_basis()
    samples = []
    for label, x, y in BLOCKS:
        block = original[y : y + 8, x : x + 8].astype(np.float64)
        coefficients = basis @ (block - 128) @ basis.T
        direct = basis.T @ coefficients @ basis + 128
        ac = coefficients ** 2
        ac[0, 0] = 0
        ac_energy = float(ac.sum())

        quantization = []
        reconstructions = []
        for step in QUANT_STEPS:
            quantized = np.rint(coefficients / step).astype(np.int64)
            reconstructed = basis.T @ (step * quantized) @ basis + 128
            reconstructions.append(reconstructed)
            quantization.append({
                "step": step,
                "nonzero": int(np.count_nonzero(quantized)),
                "mse": pixel_mse(block, reconstructed),
            })
        samples.append({
            "label": label,
            "x": x,
            "y": y,
            "pixels": block.astype(int),
            "coefficients": coefficients,
            "std": float(np.std(block)),
            "dc": float(coefficients[0, 0]),
            "ac_energy": ac_energy,
            "direct_mse": pixel_mse(block, direct),
            "direct_max_error": float(np.max(np.abs(block - direct))),
            "quantization": quantization,
            "reconstructions": reconstructions,
        })

    dct_comparison_figure(samples, output)
    return samples


def dct_comparison_figure(samples, output):
    fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.55), layout="constrained")
    peak = max(float(np.log1p(np.abs(sample["coefficients"])).max()) for sample in samples)
    for index, sample in enumerate(samples):
        row = axes[index]
        row[0].imshow(sample["pixels"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        row[0].set_title(f'{sample["label"]} original\n({sample["x"]}, {sample["y"]})', fontweight="bold")
        heatmap = row[1].imshow(np.log1p(np.abs(sample["coefficients"])), cmap="magma",
                                vmin=0, vmax=peak, interpolation="nearest")
        row[1].set_title("log(1 + |DCT|)\nDC at upper left")
        for ax, stats, reconstructed in zip(row[2:], sample["quantization"], sample["reconstructions"]):
            ax.imshow(reconstructed, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f'q = {stats["step"]}     NZ = {stats["nonzero"]}\nMSE = {stats["mse"]:.2f}')
        for ax in row:
            ax.axis("off")
    fig.colorbar(heatmap, ax=axes[:, 1], orientation="horizontal", shrink=0.84, pad=0.03)
    fig.savefig(output / "dct_quant_compare.png", dpi=190)
    plt.close(fig)


def format_measurements(entropy, compression, dct, shape):
    height, width = shape
    lines = [f"输入图像 {width}×{height}，8-bit 灰度。", "", "[实验一 信息熵]",
             f"双灰度值 {TWO_LEVELS[0]} 和 {TWO_LEVELS[1]}，噪声种子 {NOISE_SEED}。",
             "图像                 有效灰度级    H (bit/像素)       8/H"]
    for row in entropy:
        ratio = f'{row["ratio"]:.6f}' if row["ratio"] is not None else "未定义"
        lines.append(f'{row["label"]:20s} {row["levels"]:8d} {row["entropy"]:16.6f} {ratio:>10s}')

    x, y, roi_width, roi_height = ROI
    lines.extend(["", "[实验二 JPEG 与 PNG]", f"原始像素数据大小 {width * height} B。",
                  f"放大区左上角 (x, y) = ({x}, {y})，大小 {roi_width}×{roi_height}。",
                  "格式           大小 (B)       CR          bpp         MSE       PSNR (dB)"])
    for row in compression:
        psnr = f'{row["psnr"]:.6f}' if math.isfinite(row["psnr"]) else "∞"
        lines.append(f'{row["label"]:12s}{row["bytes"]:10d}{row["cr"]:11.6f}'
                     f'{row["bpp"]:13.6f}{row["mse"]:12.6f}{psnr:>15s}')
    png_equal = "是" if compression[-1]["identical"] else "否"
    lines.append(f"PNG 是否逐像素完全一致：{png_equal}")

    lines.extend(["", "[实验三 DCT 与量化]", "坐标为从零开始的 (x, y)，即横、纵顺序。"])
    for sample in dct:
        lines.extend([
            "",
            f'{sample["label"]} ({sample["x"]}, {sample["y"]})',
            "8×8 原始像素矩阵：",
            np.array2string(sample["pixels"], max_line_width=120),
            "8×8 DCT 系数矩阵：",
            np.array2string(sample["coefficients"], max_line_width=140,
                            formatter={"float_kind": lambda value: f"{value:9.3f}"}),
            f'像素标准差 {sample["std"]:.6f}，DC 系数 {sample["dc"]:.6f}，'
            f'AC 能量 {sample["ac_energy"]:.6f}。',
            f'直接逆变换 MSE {sample["direct_mse"]:.6e}，'
            f'最大绝对误差 {sample["direct_max_error"]:.6e}。',
            "步长 q   非零系数数    重建 MSE",
        ])
        for row in sample["quantization"]:
            lines.append(f'{row["step"]:6d} {row["nonzero"]:11d} {row["mse"]:13.6f}')
    return "\n".join(lines) + "\n"


def run_experiments():
    with Image.open(ROOT / "camera_clean.png") as image:
        original = np.array(image.convert("L"))
    if original.shape != (512, 512):
        raise ValueError("本实验原图应为 512×512 像素。")

    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    entropy = entropy_experiment(original, output)
    compression = compression_experiment(original, output)
    dct = dct_experiment(original, output)
    measurements = format_measurements(entropy, compression, dct, original.shape)
    (output / "measurements.txt").write_text(measurements, encoding="utf-8")
    print(measurements, end="")
    return {"entropy": entropy, "compression": compression, "dct": dct}


if __name__ == "__main__":
    run_experiments()
