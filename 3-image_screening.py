#  每个小块之间利用梯度直方图计算距离，当梯度除以灰度之后再进行归一化
import numpy as np
import cv2
import os
import csv
import shutil
import matplotlib.pyplot as plt
from scipy.spatial import distance


def hog_hist(img_path, bins):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    # print(img_path)
    # 读取图像并预处理
    # 提取路径中的文件夹和文件名
    folder, filename = os.path.split(image_path)
    folder_name = os.path.basename(folder)
    # 提取文件名中的基本部分（去除扩展名）
    base_name = os.path.splitext(filename)[0]
    image = cv2.imread(img_path)
    print(type(image))
    image = cv2.resize(image, (120, 120), cv2.INTER_LANCZOS4)
    # 将原图转变为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 计算梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) / 4
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) / 4

    # 计算梯度幅值
    magnitude = (np.sqrt(sobelx ** 2 + sobely ** 2)) / 2
    magnitude_n = (magnitude - 0) / 100
    gray_n = (gray - 0) / 255
    final_mag = magnitude_n
    final_mag[final_mag > 1] = 1
    # 绘制梯度强度直方图
    # 手动创建0-1的均匀划分bins
    bin_edges = np.linspace(0.0, 1.0, num=bins + 1)  # 生成11个边界点（10个区间）
    print(bin_edges)
    hist, _ = np.histogram(final_mag, bins=bin_edges)
    # 转换为概率直方图
    prob_hist = hist / hist.sum()
    hist2, _ = np.histogram(gray_n, bins=bin_edges)
    prob_hist2 = hist2 / hist2.sum()
    return final_mag, prob_hist, gray_n


yao_id = "image_all_res/1"
# 创建 CSV 文件并写入表头
csv_dir = "D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/" + yao_id + "_test_csv"
os.makedirs(csv_dir, exist_ok=True)
csv_filename = "D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/" + yao_id + "_test_csv/test_nine_bin20_" + yao_id.split('/')[-1] + "_yao.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['image_name', 'mean2', 'mean2_gray', 'multi', 'if_next'])  # 表头
folder_path = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + yao_id + '_twokind/yao/'
# 获取文件夹中的所有文件名
images_files = []
actual_labels = []
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
for fi in image_files:
    images_files.append(folder_path + fi)
print('image_files_len:', len(images_files))
count_ac = 0
ACC = 0
SUM = 0
target_base_dir = "D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/" + yao_id + "_twokind/yao_if_detection2/"
for image_file in images_files:
    # image_path = folder_path + image_file  superpixel_21.jpg
    image_path = image_file
    # if image_path != 'D:/code/juhua/Split_yao/test/no/9/superpixel_4.jpg':
    #     continue
    print(image_path)
    # 提取路径中的文件夹和文件名
    folder, filename = os.path.split(image_path)
    folder_name = os.path.basename(folder)
    # 提取文件名中的基本部分（去除扩展名）
    base_name = os.path.splitext(filename)[0]
    image = cv2.imread(image_path)
    image_re = cv2.resize(image, (120, 120), cv2.INTER_LANCZOS4)
    # 设置窗口大小
    window_size = 40
    bins = 20
    dis_p = 0.01
    # 获取图像的尺寸
    height, width, _ = image_re.shape
    print(height, width)
    magnitude_o, hist_o, gray_o = hog_hist(image_path, bins)
    count = 0
    # 滑动窗口扫描
    feature_res = []  # 用于存放距离的mean、std
    feature_res_gray = []
    feature_dis = []  # 用于存放梯度直方图之间的距离
    hist_pieces = []  # 用于存放小块的梯度直方图
    hist_pieces_gray = []
    magnitude_pieces = []  # 用于存放小块的梯度强度值
    split_output_folder = './test_res/split_res/'
    split_img_folder = split_output_folder + folder_name + "_" + base_name
    if not os.path.exists(split_img_folder):
        os.makedirs(split_img_folder)
    image_re_filename = os.path.join(split_img_folder, "split_origin.jpg")
    cv2.imwrite(image_re_filename, image_re)
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            count = count + 1
            right_b = j + window_size
            down_b = i + window_size
            magnitude_piece = magnitude_o[i:down_b, j:right_b]
            gray_piece = gray_o[i:down_b, j:right_b]
            magnitude_pieces.append(magnitude_piece)
            # 计算小块图像的直方图特征
            # 手动创建0-1的均匀划分bins
            # max_img_v = np.max(magnitude_o)
            # 生成小区域梯度直方图
            bin_edges = np.linspace(0.0, 1.0, num=bins + 1)  # 生成11个边界点（10个区间）
            hist, _ = np.histogram(magnitude_piece, bins=bin_edges)
            hist_p = hist / hist.sum()
            # 生成小区域灰度直方图
            hist2, _ = np.histogram(gray_piece, bins=bin_edges)
            hist_p2 = hist2 / hist2.sum()
            # # 将两个直方图值拼接到一起
            # hist_new = np.hstack((hist_p, hist_p2))
            hist_pieces.append(hist_p)
            hist_pieces_gray.append(hist_p2)
            bin_la = list(range(bins))
            plt.bar(bin_la, hist_p)  # bins参数决定了直方图的柱子数量
            bin_la2 = list(range(bins, bins*2))
            plt.bar(bin_la2, hist_p2)  # bins参数决定了直方图的柱子数量
            plt.ylim(0, 1.0)
            # 保存直方图为文件
            plt.savefig('D:/code/juhua/test_yaocai_quanliucheng/test_res/hist/histogram_gray_' + folder_name + "_" + base_name + '_' + str(count) + '.png')
            plt.close()
            # 保存切割的图片
            window_o = image_re[i:down_b, j:right_b]
            window_filename = os.path.join(split_img_folder, "split_" + str(count) + ".jpg")
            cv2.imwrite(window_filename, window_o)
    print('hist_pieces_len:', len(hist_pieces))  # 梯度直方图的数量
    print('hist_pieces[0]_len:', len(hist_pieces[0]))  # 梯度直方图的柱子数量
    print('hist_pieces_gray_len:', len(hist_pieces_gray))  # 灰度直方图的数量
    print('hist_pieces_gray[0]_len:', len(hist_pieces_gray[0]))  # 灰度直方图的柱子数量
    blocks_dis = []  # 记录每个小块与其他8个小块之间的欧式距离
    blocks_dis_all = []  # 记录所有距离
    # 计算小块之间梯度直方图的欧式距离
    for i in range(len(hist_pieces)):
        distances = []
        for j in range(len(hist_pieces)):
            if i != j:
                # 计算余弦距离
                # dist_cos = distance.cosine(hist_pieces[i], hist_pieces[j])
                # 计算欧式距离
                dist = np.linalg.norm(hist_pieces[i] - hist_pieces[j])
                distances.append(dist)
                blocks_dis_all.append(dist)
        blocks_dis.append(distances)
    print('blocks_dis_len:', len(blocks_dis))
    print('blocks_dis[0]_len:', len(blocks_dis[0]))
    print('blocks_dis_all_len:', len(blocks_dis_all))
    blocks_mean_dis = []  # 记录每个小块的8个距离的均值
    for m in range(len(blocks_dis)):
        print(blocks_dis[m])
        mean_dis = np.mean(blocks_dis[m])
        blocks_mean_dis.append(mean_dis)
    print('blocks_mean_dis:', blocks_mean_dis)
    # feature_res.append(actual_labels[SUM])
    SUM = SUM + 1
    mean_blocks = np.mean(blocks_mean_dis)
    # feature_res.append(mean_blocks)
    std_blocks = np.std(blocks_mean_dis)
    # feature_res.append(std_blocks)
    print(count_ac)
    blocks_dis_all_sorted = np.sort(blocks_dis_all)
    blocks_dis_all_sorted_save = blocks_dis_all_sorted[:-36]
    print("blocks_dis_all_sorted_save_len:", len(blocks_dis_all_sorted_save[:-18]))
    mean_blocks_36 = np.mean(blocks_dis_all_sorted_save[:-18])
    feature_res.append(mean_blocks_36)
    std_blocks_36 = np.std(blocks_dis_all_sorted_save[:-18])
    # feature_res.append(std_blocks_36)
    # 计算小块灰度直方图之间的欧式距离
    blocks_dis_gray = []
    blocks_dis_all_gray = []
    for i in range(len(hist_pieces_gray)):
        distances = []
        for j in range(len(hist_pieces_gray)):
            if i != j:
                # 计算余弦距离
                # dist_cos = distance.cosine(hist_pieces[i], hist_pieces[j])
                # 计算欧式距离
                dist = np.linalg.norm(hist_pieces_gray[i] - hist_pieces_gray[j])
                distances.append(dist)
                blocks_dis_all_gray.append(dist)
        blocks_dis_gray.append(distances)
    print('blocks_dis_len:', len(blocks_dis_gray))
    print('blocks_dis[0]_len:', len(blocks_dis_gray[0]))
    print('blocks_dis_all_len:', len(blocks_dis_all_gray))
    blocks_mean_dis_gray = []
    for m in range(len(blocks_dis_gray)):
        print(blocks_dis_gray[m])
        mean_dis = np.mean(blocks_dis_gray[m])
        blocks_mean_dis_gray.append(mean_dis)
    print('blocks_mean_dis_gray:', blocks_mean_dis_gray)
    mean_blocks_gray = np.mean(blocks_mean_dis_gray)
    # feature_res_gray.append(mean_blocks_gray)
    std_blocks_gray = np.std(blocks_mean_dis_gray)
    # feature_res_gray.append(std_blocks_gray)
    print(count_ac)
    blocks_dis_all_sorted_gray = np.sort(blocks_dis_all_gray)
    blocks_dis_all_sorted_save_gray = blocks_dis_all_sorted_gray[:-36]
    print("blocks_dis_all_sorted_save_len:", len(blocks_dis_all_sorted_save_gray[:-18]))
    mean_blocks_36_gray = np.mean(blocks_dis_all_sorted_save_gray[:-18])
    feature_res_gray.append(mean_blocks_36_gray)
    std_blocks_36_gray = np.std(blocks_dis_all_sorted_save_gray[:-18])
    # feature_res_gray.append(std_blocks_36_gray)

    # row = [image_file] + blocks_mean_dis + feature_res + blocks_mean_dis_gray + feature_res_gray
    mulit_res = mean_blocks_36 * mean_blocks_36_gray
    if mulit_res < 0.005:
        if_second = 0
        dest_dir = target_base_dir + '0/'
    else:
        if_second = 1
        dest_dir = target_base_dir + '1/'
    row = [image_file] + feature_res + feature_res_gray + [mulit_res, if_second]
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    count_ac = count_ac + 1
    # 创建目标文件夹
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(image_path))
    shutil.copy(image_path, dest_path)
    print(f"移动成功: {image_path} -> {dest_path}")
