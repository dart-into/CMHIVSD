# resnet50
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn.functional as F
import shutil

# 设置种子以确保可复现
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)  # 如果使用多个 GPU

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# mean = [0.4217, 0.3918, 0.3586]
# std = [0.2562, 0.2450, 0.2364]
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]  # 来正则化


def load_and_preprocess_image(image_path, target_size=(256, 256), crop_size=(224, 224)):
    # print(image_path)
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    transform_c = transforms.RandomCrop(crop_size)
    image = transform_c(image)
    image = transforms.ToTensor()(image)
    transform_n = transforms.Normalize(mean, std)
    image = transform_n(image)
    return image


class Mymodel_resnet(nn.Module):
    def __init__(self, num_classes):
        super(Mymodel_resnet, self).__init__()
        self.net1 = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 512)
        self.batch1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, image):
        image_out = self.net1(image)
        output = self.fc1(image_out)
        output = F.relu(self.batch1(output))
        output = self.dropout1(output)
        output = self.fc2(output)
        return output


def test_model(image_path):
    image = load_and_preprocess_image(image_path).unsqueeze(0).to(device)
    net = Mymodel_resnet(2).to(device)
    # 加载预训练的模型参数
    net.load_state_dict(torch.load('./models/res50_model_best_new_3.pt'))
    net.eval()
    with torch.no_grad():
        output = net(image)
    # 获取预测的类别
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()


if __name__ == '__main__':
    # destination_folder = 'D:/code/juhua/Split_yao/res/3/100/different/'
    yao_id = "image_all_res/4"
    folder_path = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + yao_id + '/'
    target_base_dir = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + yao_id +'_twokind'
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为图片格式（例如 .jpg, .png, .jpeg 等）
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 拼接完整的文件路径
            file_path = os.path.join(folder_path, filename)
            print(filename)
            class_id = test_model(file_path)
            print('class_id:', class_id)
            if int(class_id) == 1:
                target_dir = target_base_dir + '/yao/'
            else:
                target_dir = target_base_dir + '/back/'
            dest_dir = target_dir
            # 创建目标文件夹
            os.makedirs(dest_dir, exist_ok=True)
            src_path = file_path
            if os.path.isfile(src_path):
                folder, filename = os.path.split(src_path)
                folder_name = os.path.basename(folder)
                # dest_path = os.path.join(dest_dir, folder_name + '_' + os.path.basename(src_path))
                dest_path = os.path.join(dest_dir, os.path.basename(src_path))
                shutil.copy(src_path, dest_path)
                print(f"移动成功: {src_path} -> {dest_path}")
            else:
                print(f"文件不存在，跳过: {src_path}")
