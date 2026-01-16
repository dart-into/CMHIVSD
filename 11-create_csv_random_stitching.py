import os
import csv
import pandas as pd
import numpy as np
import random


# 根据药材对应的csv生成对应的test.csv
def my_csv_lie(res_csv_path, small_img_path, num, id_num):
    for w in range(id_num):
        # 获取文件夹中的所有文件名
        small_file = small_img_path
        print(len(small_file))
        small_num = list(range(0, len(small_file)))
        random.shuffle(small_num)
        test_samll_index = small_num[0:len(small_num)]
        with open(res_csv_path, 'a', newline='', encoding='gbk') as file:
            writer = csv.writer(file)
            small_co = 0
            i = 0
            # 将所有图像都用一遍
            while i < len(test_samll_index):
                i = i + 4
                data = []
                for j in range(4):
                    data.append(small_file[test_samll_index[small_co % len(test_samll_index)]])
                    small_co = small_co + 1
                data.append(num)
                # data.append(id_num)
                writer.writerow(data)
            print(small_co)


# folder_path = 'D:/File/smallimg_10kind/small_img_train_self'
yao_id = "image_all_res/17"
csv_path = "./test_quanliucheng/test/" + yao_id + "_test_csv/pred_self.csv"
df = pd.read_csv(csv_path)
# 检查列是否存在
if not {"Small_Image_Path", "Max_Label"}.issubset(df.columns):
    raise ValueError("CSV 文件缺少 Small_Image_Path 或 Max_Label 列")

# 按 Max_Label 分组，提取 Small_Image_Path
grouped_paths = {}
for max_label, group_df in df.groupby("Max_Label"):
    paths_array = group_df["Small_Image_Path"].tolist()  # 转成 Python 列表
    grouped_paths[max_label] = paths_array

res_csv_path = './test_quanliucheng/test/' + yao_id + '_test_csv/smallimg_yao_sui.csv'
with open(res_csv_path, 'w', newline='', encoding='gbk') as file:
    writer = csv.writer(file)
    writer.writerow(['Small Image 1', 'Small Image 2', 'Small Image 3', 'Small Image 4', 'Label'])
id_num = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
print(len(grouped_paths))
for label, paths in grouped_paths.items():
    my_csv_lie(res_csv_path, paths, label, id_num[label])


