from pathlib import Path

import cv2
import numpy as np

DATA_PATH = Path("data/photo.jpg")
RESULTS_DIR = Path("results")

# 下面是参数设定
# 取样像素 (u, v) = (x, y)
PIXEL_U, PIXEL_V = 320, 240

# 裁剪区域：[y1:y2, x1:x2]
CROP_X1, CROP_Y1, CROP_X2, CROP_Y2 = 160, 120, 440, 360

# 局部改色区域
PATCH_X1, PATCH_Y1, PATCH_X2, PATCH_Y2 = 40, 40, 160, 140


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """在图片左上角写白字黑底标签，方便拼图对比。"""
    out = image.copy()
    # 这里使用cv2.rectangle()函数在图像上绘制一个矩形，作为标签的背景
    # 矩形的左上角坐标为(0, 0)，右下角坐标为(len(text) * 14 + 20, 36)，颜色为黑色(0, 0, 0)，填充整个矩形(-1表示填充)
    cv2.rectangle(out, (0, 0), (len(text) * 14 + 20, 36), (0, 0, 0), -1)
    # 这里使用cv2.putText()函数在图像上绘制文本，作为标签的内容
    # 文本的左下角坐标为(10, 26)，字体为cv2.FONT_HERSHEY_SIMPLEX，字体大小为0.8，颜色为白色(255, 255, 255)，线宽为2，线型为cv2.LINE_AA(抗锯齿)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取图像文件，使用cv2.imread()函数读取指定路径的图像文件，返回一个NumPy数组表示图像数据
    img = cv2.imread(str(DATA_PATH))
    if img is None:
        raise SystemExit(f"Cannot read {DATA_PATH}")

    # 1. 图像属性：shape 的顺序是 (高, 宽, 通道)
    h, w, c = img.shape
    print("shape:", img.shape, "dtype:", img.dtype)
    print(f"height h = {h}, width w = {w}, channels c = {c}")

    # 2. 访问单个像素：索引顺序是 [v, u] = [y, x]
    pixel = img[PIXEL_V, PIXEL_U]
    print(f"img[v={PIXEL_V}, u={PIXEL_U}] (B, G, R) = {pixel.tolist()}")

    # 3. 切片裁剪：行在前、列在后；copy() 让结果脱离原图，避免误改原图
    crop = img[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2].copy()
    cv2.imwrite(str(RESULTS_DIR / "crop.jpg"), crop)
    print(f"crop x:{CROP_X1}-{CROP_X2}, y:{CROP_Y1}-{CROP_Y2} -> shape {crop.shape}")

    # 4. 原图副本上把一小块改成红色；BGR 顺序下红色是 (0, 0, 255)
    red_patch = img.copy()
    red_patch[PATCH_Y1:PATCH_Y2, PATCH_X1:PATCH_X2] = (0, 0, 255)

    # 5. 只保留一个通道：另外两个通道置零，shape 仍是 (h, w, 3)
    only_b = np.zeros_like(img)
    only_b[:, :, 0] = img[:, :, 0]

    only_g = np.zeros_like(img)
    only_g[:, :, 1] = img[:, :, 1]

    only_r = np.zeros_like(img)
    only_r[:, :, 2] = img[:, :, 2]

    # 6. 灰度图：shape 变成 (h, w)，三个通道的加权和
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray shape:", gray.shape, "| color shape:", img.shape)

    # 7. 拼对比图：灰度先转回三通道，才能和彩色图 hstack
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panels = [
        add_label(img, "Original"),
        add_label(red_patch, "Red patch"),
        add_label(only_b, "Only B"),
        add_label(only_g, "Only G"),
        add_label(only_r, "Only R"),
        add_label(gray_bgr, "Gray"),
    ]
    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    cv2.imwrite(str(RESULTS_DIR / "compare_channels.jpg"), grid)
    print("Saved results/crop.jpg and results/compare_channels.jpg")


if __name__ == "__main__":
    main()
