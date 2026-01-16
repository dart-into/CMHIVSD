import os
import csv
import pandas as pd
import numpy as np
import random


# 根据药材对应的csv生成对应的test.csv
def my_csv_lie(folder_name, num, id_num, res_csv):
    for w in range(id_num):
        print(folder_name)
        # 获取文件夹中的所有文件名
        small_file = [f for f in os.listdir(folder_name) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        print(len(small_file))
        small_num = list(range(0, len(small_file)))
        random.shuffle(small_num)
        test_samll_index = small_num[0:len(small_num)]
        with open(res_csv, 'a', newline='', encoding='gbk') as file:
            writer = csv.writer(file)
            small_co = 0
            i = 0
            # 将所有图像都用一遍
            while i < len(test_samll_index):
                i = i + 1
                data = []
                for j in range(4):
                    data.append(folder_name + '/' + small_file[test_samll_index[small_co % len(test_samll_index)]])
                small_co = small_co + 1
                data.append(num)
                # data.append(id_num)
                writer.writerow(data)
            print(small_co)


yao_id = "image_all_res/15"
folder_path = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + yao_id + '_twokind/yao_if_detection/smallimg/'
res_csv = './test_quanliucheng/test/' + yao_id + '_test_csv/smallimg_yao_self.csv'

# folder_path = 'D:/File/smallimg_10kind/small_img_test_012356/'
# res_csv = './test_part/smallimg_yao_012356_self.csv'
with open(res_csv, 'w', newline='', encoding='gbk') as file:
    writer = csv.writer(file)
    writer.writerow(['Small Image 1', 'Small Image 2', 'Small Image 3', 'Small Image 4', 'Label'])

# for filename in os.listdir(folder_path):
# id_num = [200, 8, 2, 2, 9, 12, 2, 48, 6, 27]
id_num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
file_path = folder_path
index_now = 0
print(file_path)
my_csv_lie(file_path, 0, id_num[index_now], res_csv)
# for filename1 in os.listdir(file_path):
#     file_path1 = file_path + '/' + filename1
#     print(file_path1)
#     kind = filename1.split('_')[1]
#     print(kind)
#     my_csv_lie(file_path1, kind, id_num[index_now], res_csv)
#     index_now = index_now + 1

