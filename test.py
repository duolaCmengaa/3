import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt #plt 用于显示图片
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.models as models
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import os
from torchvision.models.alexnet import AlexNet_Weights

import os
import torch
from load_data import *

import matplotlib.pyplot as plt
def test_model(model, test_dataloader, device):
    top1_correct = 0
    top5_correct = 0
    total = 0
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 获取Top-1预测
            _, predicted = torch.max(outputs, 1)
            top1_correct += (predicted == labels).sum().item()
            
            # 获取Top-5预测
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            top5_correct += sum([labels[i] in top5_pred[i] for i in range(labels.size(0))])
            
            total += labels.size(0)
    
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    return top1_accuracy, top5_accuracy
if __name__ == '__main__':
    # 加载模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    model_path = os.path.join(result_dir, 'model')
    os.makedirs(model_path, exist_ok=True)
    load_path = os.path.join(model_path, 'firstmodepth_3_1.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8

    train_dataset, train_dataloader, validation_dataset, test_dataset, validation_dataloader, test_dataloader, _ = get_data(batch_size)
    net = models.resnet18()

    in_channel = net.fc.in_features 
    net.fc = nn.Linear(in_channel, 200)  # 修改全连接层输出为200个类别
    net.to(device)

    # 加载模型并将存储位置映射到可用的CUDA设备或CPU
    net.load_state_dict(torch.load(load_path, map_location=device))

    # 测试模型
    train_top1_accuracy, train_top5_accuracy = test_model(net, train_dataloader, device)
    print('Train Top-1 Accuracy: {:.6f}%'.format(100 * train_top1_accuracy))
    print('Train Top-5 Accuracy: {:.6f}%'.format(100 * train_top5_accuracy))

    validation_top1_accuracy, validation_top5_accuracy = test_model(net, validation_dataloader, device)
    print('Validation Top-1 Accuracy: {:.6f}%'.format(100 * validation_top1_accuracy))
    print('Validation Top-5 Accuracy: {:.6f}%'.format(100 * validation_top5_accuracy))

    test_top1_accuracy, test_top5_accuracy = test_model(net, test_dataloader, device)
    print('Test Top-1 Accuracy: {:.6f}%'.format(100 * test_top1_accuracy))
    print('Test Top-5 Accuracy: {:.6f}%'.format(100 * test_top5_accuracy))
