# resnet50
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
import csv
import os
from swin import SwinTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置种子以确保可复现
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)  # 如果使用多个 GPU


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row[0]
        label = row[-1]
        image = load_and_preprocess_image(image_path)
        return image, label, image_path


mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]  # 来正则化


def load_and_preprocess_image(image_path, target_size=(224, 224), crop_size=(224, 224)):
    # print(image_path)
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    # transform_c = transforms.RandomCrop(crop_size)
    # image = transform_c(image)
    image = transforms.ToTensor()(image)
    transform_n = transforms.Normalize(mean, std)
    image = transform_n(image)
    return image


class Mymodel_resnet(nn.Module):
    def __init__(self, num_classes):
        super(Mymodel_resnet, self).__init__()
        self.net1 = SwinTransformer(
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24])
        checkpoint = torch.load('./models/swin_small_patch4_window7_224_22kto1k.pth')
        self.net1.load_state_dict(checkpoint['model'])
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


yao_id = "image_all_res/17"
folder_path = 'D:/code/juhua/test_yaocai_quanliucheng/test_quanliucheng/test/' + yao_id + '_twokind/yao_if_detection/bigimage/'
# # 获取文件夹中的所有文件名
images_files = []
actual_labels = []
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
for fi in image_files:
    images_files.append(folder_path + fi)
# 初始化模型并加载权重
model = Mymodel_resnet(num_classes=10).to(device)
model_path = './models/swin/big/SwinTransformer_model_all_new_10kind_1.pt'  # 修改为你要测试的模型路径
model.load_state_dict(torch.load(model_path))
print(model)
model.eval()
# 保存测试结果
results = []

with torch.no_grad():
    for i in range(len(images_files)):
        img_path = images_files[i]
        img = load_and_preprocess_image(img_path)
        img = img.unsqueeze(0)  # 变成 [1, 3, 224, 224]
        output = model(img.to(device))
        _, pred = torch.max(output, 1)
        pred_id = pred.cpu().numpy()
        results.append([img_path, pred_id.item()])
        print('img:', img_path, 'pred_id:', pred_id)
res_csv_path = './test_quanliucheng/test/' + yao_id + '_test_csv/testself_bigimg_10kind_' + yao_id.split('/')[-1] + '.csv'
with open(res_csv_path, 'w', newline='', encoding='gbk') as file:
    writer = csv.writer(file)
    writer.writerow(['Image path', 'Pred Label'])
    for i in range(len(results)):
        writer.writerow(results[i])
