from pathlib import Path

import cv2
import numpy as np

# ============================================================
# 相机标定工作流程一览（内容查阅获得，用于学习）
# ============================================================
# 目标：已知棋盘的三维物点和它们在图像中的二维像素位置，
#       反求相机内参 K、畸变系数 dist，以及每张图的位姿 R、t。
#
# 关键概念：
#   · 内角点：棋盘内部黑白方格的交点。本数据集每行 9 个、共 6 行，
#     共 54 个；这不是方格数（方格是 10×7）。
#   · 物点：棋盘角点在三维空间中的坐标。棋盘是平面，所以令 Z=0，
#     相邻角点间距为 1 个"棋盘单位"，不需要换算成米。
#   · 图像点：检测到的角点在图像中的二维像素坐标。
#   · 角点检测：findChessboardCorners 先找到角点的整数像素位置。
#   · 亚像素优化：初检测只能精确到整像素；cornerSubPix 在角点 11×11
#     邻域内利用灰度梯度迭代搜索，把坐标细化到亚像素级。收敛条件是
#     迭代 30 次或位置变化小于 0.001 像素。精度越高，标定越准。
#   · 内参 K：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]；
#     fx/fy 是以像素为单位的焦距，cx/cy 是主点（光轴与成像面的交点）。
#   · 畸变系数 [k1, k2, p1, p2, k3]：k1/k2/k3 是径向畸变（桶形/枕形），
#     p1/p2 是切向畸变（镜头与成像面不严格平行）。
#   · 外参 R, t：每张图中棋盘相对相机的旋转和平移；OpenCV 用旋转向量
#     rvec（Rodrigues 表示）和平移向量 tvec 表示，本脚本只求解不使用。
#   · 重投影误差：用估出的 K、dist、R、t 把三维物点重新投影到图像，
#     与检测到的二维角点比较；所有点误差平方的平均值开根号就是 RMS（像素）。
#     本结果约 0.41 px，说明参数可信；若变成几十像素，优先检查
#     patternSize、objp 顺序和 image_size 是否写对。
#
# 函数/步骤关系（缩进表示调用层级，箭头表示数据流向）：
#
#   main()
#    ├─ 构建 13 张图的路径，检查文件是否存在
#    ├─ 生成 objp（54×3 三维物点，Z=0）
#    ├─ for 每张图:
#    │   ├─ cv2.imread → 原始 BGR 图
#    │   ├─ cv2.cvtColor → 灰度图
#    │   ├─ cv2.findChessboardCorners → 54 个整数像素角点
#    │   ├─ cv2.cornerSubPix → 亚像素角点
#    │   ├─ objpoints.append(objp)
#    │   ├─ imgpoints.append(corners)
#    │   └─ cv2.drawChessboardCorners → corners_1~3.jpg
#    ├─ cv2.calibrateCamera(objpoints, imgpoints, image_size=(宽, 高))
#    │   └─→ rms, camera_matrix, dist_coeffs, rvecs, tvecs
#    │        ├─ np.savez → results/camera_params.npz
#    │        └─ cv2.undistort(src, K, dist) → 去畸变图
#    │             └─ add_label(Before / Undistorted) → np.hstack
#    │                  → results/undistort_compare.jpg
#    └─ 文件末尾 if __name__ == "__main__": main()
#
# 一句话概括函数关系：
#   findChessboardCorners / cornerSubPix 产生 imgpoints，和 objp 一起
#   作为 calibrateCamera 的输入；calibrateCamera 输出的 K、dist 再喂给
#   undistort（去畸变）和 np.savez（保存参数）。add_label 只是拼图辅助函数。
# ============================================================

CALIB_DIR = Path("data/calibration")
RESULTS_DIR = Path("results")

# 内角点：9 列 × 6 行
PATTERN_COLS, PATTERN_ROWS = 9, 6
PATTERN_SIZE = (PATTERN_COLS, PATTERN_ROWS)

# 相邻角点间距设为 1，单位是"棋盘单位"
SQUARE_SIZE = 1.0

# 亚像素优化的终止条件：迭代 30 次或误差小于 0.001
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """在图片左上角写白字黑底标签，方便拼图对比。"""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 14 + 20, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 显式列出 13 张：没有 left10.jpg；不要用 left*.jpg，会把非棋盘的 left.jpg 卷进来
    names = [f"left{i:02d}.jpg" for i in range(1, 10)]
    names += [f"left{i:02d}.jpg" for i in range(11, 15)]
    paths = [CALIB_DIR / name for name in names]

    # 检查文件是否存在
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing calibration images:\n" + "\n".join(missing))

    # 棋盘平面上的三维点：Z=0，相邻点间距 SQUARE_SIZE
    objp = np.zeros((PATTERN_COLS * PATTERN_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_COLS, 0:PATTERN_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 每张图对应的三维物点
    imgpoints = []  # 每张图检测到的二维角点
    used_names = []
    image_size = None
    saved = 0

    for path in paths:
        img = cv2.imread(str(path))
        # 读取图像失败时，img 为 None，跳过
        if img is None:
            print(f"[SKIP] cannot read {path.name}")
            continue

        h, w = img.shape[:2]
        if image_size is None:
            # 记录第一张图的尺寸；后续图必须一致
            image_size = (w, h)  # calibrateCamera 需要 (宽, 高)
        elif (w, h) != image_size:
            # 尺寸不一致的图无法参与同一组标定，跳过
            print(f"[SKIP] size mismatch {path.name}: {(w, h)}")
            continue

        # 检测棋盘角点
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            # flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE的作用是自适应阈值化和归一化图像，增强角点检测的鲁棒性
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        # 尺寸传错时 OpenCV 也可能返回 True，必须核对角点数量
        if not found or corners is None or len(corners) != PATTERN_COLS * PATTERN_ROWS:
            got = 0 if corners is None else len(corners)
            print(f"[FAIL] {path.name}: found={found}, corners={got}")
            continue

        # 把角点位置优化到亚像素精度
        # 具体优化方法是迭代地在角点邻域内搜索最优位置，直到满足终止条件 CRITERIA
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        # 收集标定数据：三维物点和对应的二维角点
        objpoints.append(objp)
        imgpoints.append(corners)
        used_names.append(path.name)
        print(f"[OK] {path.name}: {len(corners)} corners")

        # 保存至少三张角点标注图
        if saved < 3:
            annotated = img.copy()
            cv2.drawChessboardCorners(annotated, PATTERN_SIZE, corners, found)
            cv2.imwrite(str(RESULTS_DIR / f"corners_{saved + 1}.jpg"), annotated)
            saved += 1

    print(f"Detected successfully: {len(used_names)}/{len(paths)}")
    if len(used_names) < 3:
        raise SystemExit("Not enough valid images for calibration.")

    # 调用 cv2.calibrateCamera 进行相机标定，返回 RMS 重投影误差、相机矩阵、畸变系数、旋转向量和平移向量
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"RMS reprojection error: {rms:.4f} px")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs.ravel())

    # 保存标定结果到 .npz 文件，包含相机矩阵、畸变系数和图像尺寸
    np.savez(
        str(RESULTS_DIR / "camera_params.npz"),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
    )
    print(f"Saved results/camera_params.npz (image_size={image_size[0]}x{image_size[1]})")

    # 用第一张成功检测的图做去畸变前后对比
    src = cv2.imread(str(CALIB_DIR / used_names[0]))
    undistorted = cv2.undistort(src, camera_matrix, dist_coeffs)
    compare = np.hstack([add_label(src, "Before"), add_label(undistorted, "Undistorted")])
    cv2.imwrite(str(RESULTS_DIR / "undistort_compare.jpg"), compare)
    print(f"Saved results/undistort_compare.jpg (from {used_names[0]})")


if __name__ == "__main__":
    main()
