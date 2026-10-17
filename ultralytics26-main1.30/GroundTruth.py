import cv2
import os


def draw_yolo_gt(
    image_path,
    label_path,
    save_path,
    color=(0, 255, 0),
    thickness=2
):
    """
    根据YOLO标签绘制Ground Truth

    Parameters
    ----------
    image_path : str
        原始图片路径
    label_path : str
        YOLO标签(txt)
    save_path : str
        保存路径
    color : tuple
        框颜色(BGR)
    thickness : int
        框线宽
    """

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Cannot open label: {label_path}")

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        data = line.strip().split()

        if len(data) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, data)

        # YOLO -> 像素坐标
        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        # 防止越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # 只画框
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    cv2.imwrite(save_path, img)

    print(f"Saved to: {save_path}")


if __name__ == "__main__":

    image_path = r"datasets/VisDrone/VisDrone2019-DET-val/images/0000001_04527_d_0000008.jpg"
    label_path = r"datasets/VisDrone/VisDrone2019-DET-val/labels/0000001_04527_d_0000008.txt"

    save_path = r"GroundTruth/0000001_04527_d_0000008_GT_visualization.jpg"

    draw_yolo_gt(
        image_path=image_path,
        label_path=label_path,
        save_path=save_path,
        color=(0, 255, 0),      # 绿色
        thickness=2
    )