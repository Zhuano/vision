# sys是Python的标准库之一，提供了一些变量和函数，用于与Python解释器进行交互
import sys

# pathlib是Python的标准库之一，提供了面向对象的文件系统路径操作方法
from pathlib import Path

import cv2

# DATA_DIR是一个Path对象，表示数据目录的路径
DATA_DIR = Path("data")

# OUTPUT_PATH是一个Path对象，表示保存照片的路径
OUTPUT_PATH = DATA_DIR / "photo.jpg"

# WINDOW_NAME是一个字符串，表示摄像头窗口的标题
WINDOW_NAME = "Camera (s: save, q: quit)"


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    定义一个函数open_camera，参数index表示摄像头的索引，默认值为0。
    返回值是一个cv2.VideoCapture对象，用于捕获摄像头的视频流。
    """
    # 对cap对象进行初始化，使用DirectShow接口打开摄像头
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    # 先使用DirectShow接口打开摄像头的原因是，DirectShow是Windows平台上用于捕获和处理多媒体数据的API，
    # 通常比默认接口更稳定和高效。如果使用DirectShow接口打开摄像头失败，则释放资源并尝试使用默认接口打开摄像头
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    return cap


def main() -> None:
    # 这一句的作用是创建数据目录，如果目录已经存在，则不会报错
    # parents=True 表示父目录不存在时一并创建；exist_ok=True 表示目录已存在也不报错
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 调用open_camera函数打开摄像头，索引为0，返回一个cv2.VideoCapture对象
    cap = open_camera(0)

    # 如果摄像头无法打开，则打印错误信息并退出程序
    # sys.exit(1) 中的 1 是退出码，非 0 表示异常退出
    if not cap.isOpened():
        print("[ERROR] Cannot open the camera.")
        print("  - Close apps that may be using it (WeChat / QQ / browser).")
        print("  - Check Settings > Privacy & security > Camera.")
        sys.exit(1)

    # 打印提示信息，告诉用户如何操作摄像头窗口
    print("[OK] Camera opened. Press 's' to save, 'q' to quit.")

    try:
        # True表示无限循环，直到用户按下'q'键退出
        while True:
            # cap.read()方法从摄像头捕获一帧图像，返回两个值：ret表示是否成功读取到一帧图像，
            # frame表示读取到的图像数据：BGR 顺序、shape=(高, 宽, 3)、dtype=uint8 的 NumPy 数组
            ret, frame = cap.read()

            # 判断是否成功读取到一帧图像，如果失败则打印错误信息并退出循环
            if not ret:
                print("[ERROR] Failed to read a frame.")
                break

            # cv2.imshow()方法在窗口中显示图像，窗口标题为WINDOW_NAME，显示的图像为frame
            cv2.imshow(WINDOW_NAME, frame)

            # cv2.waitKey(1)方法等待键盘输入，参数1表示等待1毫秒；返回值是键码，没有按键时为 -1
            # & 0xFF 取低 8 位，字母键的低字节正好等于 ASCII 码
            key = cv2.waitKey(1) & 0xFF

            # 判断按下的键是否为's'，如果是则保存当前帧图像
            if key == ord("s"):
                # cv2.imwrite()方法将图像保存到指定路径，输入图像按 BGR 解释，文件名需要是字符串，
                # 返回值表示是否保存成功
                if cv2.imwrite(str(OUTPUT_PATH), frame):
                    # frame.shape 是 (高, 宽, 通道)，[:2] 取出前两维
                    h, w = frame.shape[:2]
                    # 如果保存成功，则打印保存的文件路径和图像的宽高
                    print(f"[OK] Saved {OUTPUT_PATH} ({w}x{h})")
                else:
                    # 打印错误信息，表示保存图像失败
                    print(f"[ERROR] Failed to save {OUTPUT_PATH}")
            elif key == ord("q"):
                print("[OK] Bye.")
                break
    finally:
        # finally块中的代码无论是否发生异常都会执行，用于释放资源
        # 释放摄像头资源并关闭所有OpenCV窗口
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
