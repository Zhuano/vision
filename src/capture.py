import sys
from pathlib import Path

import cv2

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "photo.jpg"
WINDOW_NAME = "Camera (s: save, q: quit)"


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """先尝试 Windows 的 DirectShow 后端，失败再退回默认后端。

    index=0 表示第一个摄像头；如果有外接摄像头，可能需要改成 1。
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open the camera.")
        print("  - Close apps that may be using it (WeChat / QQ / browser).")
        print("  - Check Settings > Privacy & security > Camera.")
        sys.exit(1)

    print("[OK] Camera opened. Press 's' to save, 'q' to quit.")

    try:
        while True:
            # read() 返回两个值：ret 表示这一帧是否读成功，frame 是 BGR 图像
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read a frame.")
                break

            cv2.imshow(WINDOW_NAME, frame)

            # waitKey 返回 32 位整数，& 0xFF 取出低 8 位，才是按键的 ASCII 码
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # frame 本身就是 BGR，imwrite 需要的也是 BGR，直接存即可
                if cv2.imwrite(str(OUTPUT_PATH), frame):
                    h, w = frame.shape[:2]
                    print(f"[OK] Saved {OUTPUT_PATH} ({w}x{h})")
                else:
                    print(f"[ERROR] Failed to save {OUTPUT_PATH}")
            elif key == ord("q"):
                print("[OK] Bye.")
                break
    finally:
        # 不论正常退出还是中途出错，都释放摄像头、关闭窗口
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
