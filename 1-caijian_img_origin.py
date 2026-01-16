import cv2
import numpy as np
import os

folder_path = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/image_all/'
# folder_path = "D:/File/new_yaocai/light_adjust_test/yaotong/new_origin/"
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img_path = folder_path + filename
        print(img_path)
        img_name = filename.split('.')[0]
        # 读取图像
        x1, y1 = 0, 0  # 左上角坐标
        x2, y2 = 960, 660  # 右下角坐标
        print(img_path)
        img = cv2.imread(img_path)
        img = img[y1:y2, x1:x2]
        # 初始化SLIC超像素
        slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=100, ruler=20.0)
        slic.iterate(100)
        save_folder = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/image_all_slic/' + img_name
        # save_folder = "D:/File/new_yaocai/light_adjust_test/yaotong/new_origin_res/" + img_name
        print(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # 获取超像素标签和数量
        label_slic = slic.getLabels()
        number_slic = slic.getNumberOfSuperpixels()
        print('label:', label_slic)
        print('number:', number_slic)
        # 遍历每个超像素块
        for i in range(number_slic):
            # 创建一个掩码，将当前超像素的区域设为1，其他区域为0
            mask = np.zeros(label_slic.shape, dtype=np.uint8)
            mask[label_slic == i] = 255  # 对应超像素区域设置为白色
            # 使用掩码提取该超像素区域
            # superpixel = cv2.bitwise_and(img, img, mask=mask)
            # cv2.imwrite(save_folder + "origi/superpixel_" + str(i) + ".jpg", superpixel)  # 保存为文件
            # 获取当前超像素的外接矩形
            x, y, w, h = cv2.boundingRect(mask)  # x, y 是矩形左上角坐标，w, h 是宽高

            # 根据外接矩形从图像中裁剪出矩形区域
            cropped_img = img[y:y + h, x:x + w]

            # 显示或保存每个超像素的矩形区域
            # cv2.imwrite(save_folder + "/" + img_name + "_superpixel_" + str(i) + ".jpg", cropped_img)  # 保存为文件
        mask_slic = slic.getLabelContourMask(thick_line=True)
        contours, _ = cv2.findContours(mask_slic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_slic = img.copy()
        cv2.imwrite(save_folder + "/img_origin.jpg", img)
        cv2.drawContours(img_slic, contours, -1, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)  # 红色，线宽2
        cv2.imwrite(save_folder + "/img_slic.jpg", img_slic)
