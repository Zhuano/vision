from pathlib import Path

import cv2
import numpy as np

DATA_PATH = Path("data/photo.jpg")
RESULTS_DIR = Path("results")

SEED = 42
NOISE_RATIO = 0.05
KSIZE_SMALL = 3
KSIZE_LARGE = 9

# 放大观察的区域与倍数
ZOOM_X1, ZOOM_Y1, ZOOM_X2, ZOOM_Y2 = 200, 120, 400, 320
ZOOM_SCALE = 3


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """在图片左上角写白字黑底标签，方便拼图对比。"""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 14 + 20, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """两张图的平均绝对误差，用来粗略衡量和干净图的差距。"""
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(DATA_PATH))
    if img is None:
        raise SystemExit(f"Cannot read {DATA_PATH}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 固定随机种子：每次运行得到完全相同的噪点位置
    rng = np.random.default_rng(SEED)
    n = int(gray.size * NOISE_RATIO)

    # 不重复地随机取 n 个像素的线性下标，再还原成 (y, x)
    idx = rng.choice(gray.size, size=n, replace=False)
    ys, xs = np.unravel_index(idx, gray.shape)

    # 在副本上撒椒盐，别动干净的 gray
    noisy = gray.copy()
    half = n // 2
    noisy[ys[:half], xs[:half]] = 0       # 椒：黑点
    noisy[ys[half:], xs[half:]] = 255     # 盐：白点
    print(f"Added {n} noise pixels ({NOISE_RATIO:.0%}) with seed={SEED}")

    # 中值滤波：ksize 必须是大于 1 的奇数，输入必须是 uint8
    median3 = cv2.medianBlur(noisy, KSIZE_SMALL)
    median9 = cv2.medianBlur(noisy, KSIZE_LARGE)

    print(f"MAE vs clean gray: noisy={mae(noisy, gray):.2f}, "
          f"median{KSIZE_SMALL}={mae(median3, gray):.2f}, "
          f"median{KSIZE_LARGE}={mae(median9, gray):.2f}")

    # 同一区域放大：INTER_NEAREST 保留真实像素块，避免放大本身引入模糊
    roi = (slice(ZOOM_Y1, ZOOM_Y2), slice(ZOOM_X1, ZOOM_X2))

    def zoom(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image[roi], None, fx=ZOOM_SCALE, fy=ZOOM_SCALE,
                          interpolation=cv2.INTER_NEAREST)

    labels = [
        "Clean gray",
        f"Noisy {NOISE_RATIO:.0%}",
        f"Median {KSIZE_SMALL}x{KSIZE_SMALL}",
        f"Median {KSIZE_LARGE}x{KSIZE_LARGE}",
    ]
    images = [gray, noisy, median3, median9]

    # 四格总览
    panels = [add_label(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), label)
              for im, label in zip(images, labels)]
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(RESULTS_DIR / "denoise_compare.jpg"), grid)

    # 同一区域放大
    zoom_panels = [add_label(cv2.cvtColor(zoom(im), cv2.COLOR_GRAY2BGR), label)
                   for im, label in zip(images, labels)]
    zoom_grid = np.vstack([np.hstack(zoom_panels[:2]), np.hstack(zoom_panels[2:])])
    cv2.imwrite(str(RESULTS_DIR / "denoise_zoom.jpg"), zoom_grid)

    print("Saved results/denoise_compare.jpg and results/denoise_zoom.jpg")


if __name__ == "__main__":
    main()
