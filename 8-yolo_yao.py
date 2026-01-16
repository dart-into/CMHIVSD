import subprocess
import sys


def test_yolo(path, res_folder):
    # 定义命令参数（建议拆分为列表，避免路径空格问题）
    command = [
        sys.executable,  # 使用当前Python解释器路径
        "./yolov5/detect.py",
        "--source", path,
        "--weights", "./yolov5/runs/train/exp40/weights/best.pt",
        "--save-txt", "--name",
        res_folder, "--exist-ok"
    ]

    # 执行命令并捕获输出
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # 捕获标准输出和错误
        universal_newlines=True  # 以文本形式返回结果
    )

    # 输出结果
    print("标准输出:", result.stdout)
    print("标准错误:", result.stderr)
    print("返回码:", result.returncode)

    # 检查是否执行成功
    if result.returncode == 0:
        print("检测脚本执行成功！")
    else:
        print("检测脚本执行失败！")


def yolo_deal():
    import os
    import cv2
    from datetime import datetime
    img_id = 'image_all_res/3'
    # 检测结果和原图的文件夹路径
    images_dir = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + img_id + '_twokind/yao_if_detection/1_second'  # 存放含有对应类别药材的图像块文件夹
    output_dir = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + img_id + '_twokind/yao_if_detection/smallimg'  # 存放裁剪结果的文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间为 'YYYY-MM-DD HH:MM:SS' 形式
    formatted_time = current_time.strftime("%Y-%m-%d_%H_%M_%S")
    print(formatted_time)
    res_folder = formatted_time
    test_yolo(images_dir, res_folder)
    result_path = "D:/code/juhua/test_yaocai_quanliucheng/yolov5-mask-42-master/runs/detect/" + res_folder
    print(result_path)
    labels_dir = result_path + "/labels"  # 通过YOLOV5获取的Label文件夹
    # 遍历检测结果文件
    for label_file in os.listdir(labels_dir):
        image_file = label_file.replace('.txt', '.jpg')  # 假设图像为 .jpg 格式
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)
        # 加载图像
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        k = 1.2  # 膨胀系数
        # 读取标签文件中的检测信息
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # 太小的块不要
                if width * w < 15 or height * h < 15:
                    continue
                # 计算检测框的绝对坐标
                abs_x_center = int(x_center * w)
                abs_y_center = int(y_center * h)
                abs_width = int(width * w * k)  # 进行膨胀
                abs_height = int(height * h * k)
                # 计算检测框的左上角和右下角坐标
                x1 = max(0, abs_x_center - abs_width // 2)
                y1 = max(0, abs_y_center - abs_height // 2)
                x2 = min(w, abs_x_center + abs_width // 2)
                y2 = min(h, abs_y_center + abs_height // 2)
                # 从图像中切割检测区域
                cropped_image = image[y1:y2, x1:x2]
                # 保存切割后的图像
                output_path = os.path.join(output_dir, f"{image_file.split('.')[0] + '_' + str(i)}.jpg")
                cv2.imwrite(output_path, cropped_image)
                print(f"Saved cropped image: {output_path}")


if __name__ == '__main__':
    yolo_deal()
