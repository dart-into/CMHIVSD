import argparse
import csv
import glob
import os
import random
import torch
import re
import pandas as pd
from torch import optim, nn
import torchvision.models as models
from torch.utils.data import DataLoader
from MyDataset import HerbalMedicineDataset
import numpy as np
from torchvision import transforms
from swin import SwinTransformer
from sklearn.metrics import recall_score

batch_size = 1
num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test2(model, loader, csv_file_path, csv_file_path_four, r):
    correct = 0
    total = 0
    model.eval()
    predictions_dict = {}  # 初始化字典
    pred_result = []
    # 标签到索引的映射
    label_to_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}  # 根据你的标签定义映射
    num_classes = len(label_to_index)  # 类别数量
    Small_path = []
    Labels_True = []
    Labels_True_four = []
    pred_images = []  # 用于保存拼接成的大图被模型判定为的类别
    Labels_True_smallimg = {}  # 用于存放每幅小图的真实标签
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            negative_images, labels, small_image_paths = data
            negative_images, labels = negative_images.to(device), labels.to(device)
            outputs = model(negative_images)
            # probabilities = torch.softmax(outputs, dim=1)  # 计算预测概率
            value_max, predicted = torch.max(outputs.data, dim=1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            for idx in range(len(labels)):
                Labels_True_four.append(labels[idx].item())
                pred_images.append(predicted[idx].item())
                small_paths = []
                for path_id in range(4):
                    small_paths.append(small_image_paths[path_id][idx])
                    # true_label_name = small_image_paths[path_id][idx].split('/')[-2]
                    # Labels_True.append(int(true_label_name.split('_')[1]))
                Small_path.append(small_paths)

    print('Small_path_len:', len(Small_path))
    print('Small_path[0]_len:', len(Small_path[0]))
    num = 0
    pred_fin_four = []
    print('pred_images_len:', len(pred_images))
    # 大图投票
    while num < len(pred_images):
        pred_line = [0] * 10
        for i in range(1):
            pred_line[pred_images[num]] += 1
            num = num + 1
        line_array = np.array(pred_line)
        max_position = np.argmax(line_array)
        for i in range(1):
            pred_fin_four.append(max_position)
    print('pred_fin_four_len:', len(pred_fin_four))
    with open(csv_file_path_four, 'w', newline='', encoding='gbk') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Small Image 1', 'Small Image 2', 'Small Image 3', 'Small Image 4', 'True_Label', 'Pred_Label'])
    for i in range(len(pred_fin_four)):
        with open(csv_file_path_four, 'a', newline='', encoding='gbk') as file:
            writer = csv.writer(file)
            data = Small_path[i] + [Labels_True_four[i], pred_fin_four[i]]
            writer.writerow(data)
    pred_fin = []
    for i in range(len(Small_path)):
        for j in range(len(Small_path[0])):
            pred_fin.append(pred_fin_four[i])
    print('pred_fin_len:', len(pred_fin))
    new_total = len(pred_fin)
    new_correct = 0
    # for nid in range(len(pred_fin)):
    #     if pred_fin[nid] == Labels_True[nid]:
    #         new_correct = new_correct + 1
    for num in range(len(pred_fin_four)):
        pred_max = pred_fin_four[num]
        # actual_label = Labels_True_four[num]
        small_image_paths = Small_path[num]
        for i in range(len(small_image_paths)):  # 遍历当前样本的所有小图路径
            # print(len(small_image_paths))
            print('small_path:', small_image_paths[i])
            small_image_path = small_image_paths[i]
            # if small_image_path not in Labels_True_smallimg:
            #     Labels_True_smallimg[small_image_path] = actual_label  # 记录该幅小图的真实标签
            if small_image_path not in predictions_dict:
                predictions_dict[small_image_path] = [0] * num_classes  # 初始化计数列表，长度为类别数
            # # 使用映射更新计数
            index = label_to_index[pred_max]  # 获取实际标签的索引
            # print(index)
            predictions_dict[small_image_path][index] += 1
    index = 0
    for path, counts in predictions_dict.items():
        print(f"Path: {path}, Counts: {counts}")
        index = index + 1
    print('index:', index)
    df = pd.DataFrame.from_dict(predictions_dict, orient='index',
                                columns=[f'Count_Label_{label}' for label in label_to_index.keys()])
    df.index.name = 'Small_Image_Path'  # 设置索引名称
    df.reset_index(inplace=True)  # 重置索引以便将其作为列

    # 添加最大计数所在的标签和真实标签列
    df['Max_Count'] = df[[f'Count_Label_{label}' for label in label_to_index.keys()]].max(axis=1)  # 最大计数
    df['Max_Label'] = df[[f'Count_Label_{label}' for label in label_to_index.keys()]].idxmax(axis=1).apply(
        lambda x: int(x.split('_')[-1]))  # 最大计数对应的标签

    # 提取真实标签
    def extract_real_label(file_path):
        # first_number = Labels_True_smallimg[file_path]
        first_number_0 = file_path.split('/')[0]
        first_number = first_number_0.split('_')[1]
        return int(first_number)  # 假设真实标签是第一个数字

    # df['Real_Label'] = df['Small_Image_Path'].apply(extract_real_label)  # 从小图路径提取真实标签

    # 添加 Is_Correct 列，判断 Max_Label 和 Real_Label 是否相等
    # df['Is_Correct'] = (df['Max_Label'] == df['Real_Label']).astype(int)  # 相等为 1，不相等为 0
    # c = (df['Max_Label'] == df['Real_Label']).astype(int)
    # accuracy = c.sum() / len(df)
    # print(f"投票的准确率: {accuracy:.3f}")
    # df['acc_single'] = round(correct / total, 3)
    # 保存到 CSV 文件
    df.to_csv(csv_file_path, index=False)  # 保存 DataFrame 到 CSV 文件
    # print('correct:', new_correct, 'total:', new_total)
    # novote_recall = recall_score(Labels_True, pred_fin, average='macro')
    # new_recall = recall_score(df['Real_Label'], df['Max_Label'], average='macro')
    # print('novote_recall:', novote_recall)
    # print('new_recall:', new_recall)
    # return round(new_correct / new_total, 3), round(accuracy, 3), round(novote_recall, 3), round(new_recall, 3)


acc_single = []
acc_vote = []
recall1 = []
recall2 = []
for i in range(1):
    print('range' + str(i + 2))

    transform_big = transforms.Compose([
        # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_small = transforms.Compose([
        transforms.Resize((224, 224)),  # 这里将小图调整为112x112
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    yao_id = "image_all_res/3"
    csv_file_path = './test_quanliucheng/test/' + yao_id + '_test_csv/pred_self.csv'  # 指定 CSV 文件路径
    csv_file_path_four = './test_quanliucheng/test/' + yao_id + '_test_csv/pred_self_four.csv'  # 指定 CSV 文件路径
    test_csv_path = './test_quanliucheng/test/' + yao_id + '_test_csv/smallimg_yao_self.csv'
    test_dataset = HerbalMedicineDataset(csv_file=test_csv_path,
                                         transform_big=transform_big, transform_small=transform_small)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=0)

    # 加载预训练的 SWIN
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=96, depths=[2, 2, 18, 2],
                            conv_dims=(96, 192, 384, 768), num_heads=[3, 6, 12, 24], num_classes=num_classes).to(device)
    print('分类类别：', num_classes)
    # 自身拼接训练出的模型
    model_path = "./models/swin/small/model_10kind_swin_self_4.mdl"
    # 随机拼接训练出的模型
    # model_path = "./model_10kind_dense/model_10kind_dense169_quan_2.mdl"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    test2(model, test_loader, csv_file_path, csv_file_path_four, i + 1)
    df_small = pd.read_csv(csv_file_path)
    percent_series = df_small["Max_Label"].value_counts(normalize=True) * 100
    smallimg_labels = []
    for label, percent in percent_series.items():
        if percent > 15:
            smallimg_labels.append(label)
    print("15%:", smallimg_labels)
    smallimg_labels = []
    for label, percent in percent_series.items():
        if percent > 10:
            smallimg_labels.append(label)
    print("10%:", smallimg_labels)
#     acc_single.append(test_acc)
#     acc_vote.append(acc_all)
#     recall1.append(novote_recall)
#     recall2.append(new_recall)
#     print("Accuracy Of Test Set:", test_acc * 100.0, "%")
#
# import csv
# from itertools import zip_longest
#
# # 打开CSV文件（如果文件不存在，它将创建一个新文件）
# with open(
#         './test_smallimg_res/output_acc_densenet169_63011-3-577.csv',
#         mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入列名
#     writer.writerow(['acc_single', 'acc_vote', 'recall_single', 'recall_vote'])
#     # 使用 zip_longest 填充较短的数组
#     for item1, item2, item3, item4 in zip_longest(acc_single, acc_vote, recall1, recall2, fillvalue='N/A'):
#         writer.writerow([item1, item2, item3, item4])
