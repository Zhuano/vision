from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path("data/calibration")
RESULTS_DIR = Path("results")

# 内角点：9 列 × 6 行。注意这是内角点数，不是方格数（方格是 10×7）。
PATTERN_COLS, PATTERN_ROWS = 9, 6
PATTERN_SIZE = (PATTERN_COLS, PATTERN_ROWS)

# 相邻角点间距设为 1，单位是"棋盘单位"；不需要换算成米。
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
        if img is None:
            print(f"[SKIP] cannot read {path.name}")
            continue

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)  # calibrateCamera 需要 (宽, 高)
        elif (w, h) != image_size:
            print(f"[SKIP] size mismatch {path.name}: {(w, h)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        # 尺寸传错时 OpenCV 也可能返回 True，必须核对角点数量
        if not found or corners is None or len(corners) != PATTERN_COLS * PATTERN_ROWS:
            got = 0 if corners is None else len(corners)
            print(f"[FAIL] {path.name}: found={found}, corners={got}")
            continue

        # 把角点位置优化到亚像素精度
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

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

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"RMS reprojection error: {rms:.4f} px")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs.ravel())

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
