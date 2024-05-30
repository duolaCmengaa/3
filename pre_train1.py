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
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import os
from torchvision.models.alexnet import AlexNet_Weights
from load_data import *
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm
import torch.nn.init as init

# 获取并改造ResNet-18模型：获取ResNet-18模型，并加载预训练模型的权重。将其最后一层（输出层）去掉，换成一个全新的全连接层，该全连接层的输出节点数与本例分类数相同。

def get_ResNet18(classes, pretrained=True, loadfile=None):
    if pretrained:
        ResNet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练的 ResNet-18 模型权重

    if loadfile is not None:
        ResNet18.load_state_dict(torch.load(loadfile))  # 加载本地模型

    # 将所有的参数层进行冻结：设置模型仅最后一层可以进行训练，使模型只针对最后一层进行微调。
    for param in ResNet18.parameters():
        param.requires_grad = False

    # 获取全连接层的输入特征数
    in_features = ResNet18.fc.in_features

    # 修改最后一层为新的全连接层
    ResNet18.fc = nn.Linear(in_features, len(classes))

    # 初始化新的全连接层
    init.xavier_uniform_(ResNet18.fc.weight)
    if ResNet18.fc.bias is not None:
        init.zeros_(ResNet18.fc.bias)

    # 只有最后一层的参数会被更新
    ResNet18.fc.requires_grad = True

    # 打印模型
    # print(ResNet18)

    return ResNet18

if __name__ == '__main__':
    # 迁移学习步骤①：固定预训练模型的特征提取部分，只对最后一层进行训练，使其快速收敛。
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    model_path = os.path.join(result_dir, 'model')
    os.makedirs(model_path, exist_ok=True)
    firstmodepth = os.path.join(model_path, 'firstmodepth_1.pth')
    picture_dir = os.path.join(result_dir, 'pirtures')
    picture_save_path = os.path.join(picture_dir, "train1_batch_sizes_learningrates")
    print("—————————固定预训练模型的特征提取部分，只对最后一层进行训练，使其快速收敛—————————")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    learningrates = [1e-1,2e-2,1e-2,2e-3,1e-3,2e-4,1e-4,2e-4,1e-5]
    batch_sizes = [4,8,16,32]
    '''经过以learningrates = [1e-1,2e-2,1e-2,2e-3,1e-3,2e-4,1e-4,2e-4,1e-5]这些数据预先训练,发现学习率集中在0.002-0.001这一部分准确率较高,于是对这一范围进行细致查找,我第二次令learningrates = [0.005,0.004,0.003],batch_sizes = [4,8,16,32]进行训练,发现准确率并没有提高,为0.38,所以还是用第一次运行的结果'''
    
    
    total_best_acc = 0
 
    for batch_size in batch_sizes:
        train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,_ = get_data(batch_size)
        best_acc_list = []
        

        for lr in learningrates:
            
            # 指定新加的全连接层的学习率
            classes = range(200)
            ResNet18 = get_ResNet18(classes=classes, pretrained=True)  # 实例化模型

            ResNet18.to(device=device)  # 将模型移动到设备上
            net = ResNet18

            optimizer = torch.optim.Adam(net.fc.parameters(), lr=lr)

            epochs = 4
            best_acc = 0.0

            val_accuracy_list = []
            train_accuracy_list=[]
            epochs_list = []
            train_loss_list = []
            val_loss_list = []

            train_num = len(train_dataset)
            val_num = len(validation_dataset)

            # 定义损失函数、优化器和训练、测试函数
            loss_function = nn.CrossEntropyLoss().to(device)

            for epoch in range(epochs):
                # train
                net.train()       
                
                train_bar = tqdm(train_dataloader)
                for step, data in enumerate(train_bar):
                    images, labels = data
                    optimizer.zero_grad()
                    logits = net(images.to(device))
                    loss = loss_function(logits, labels.to(device))
                    loss.backward()
                    optimizer.step()
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

                # validate
                net.eval()
                val_acc = 0.0  # 累计验证集中的所有正确答对的个数
                train_acc=0.0  # 累计训练集中所有正确答对的个数
                val_loss = 0.0 # 累计验证集中所有误差
                train_loss=0.0 # 累积训练集中所有误差
                with torch.no_grad():
                    val_bar = tqdm(validation_dataloader)
                    for train_data in train_bar:
                        train_images,train_labels=train_data
                        train_outputs=net(train_images.to(device))
                        tmp_train_loss=loss_function(train_outputs,train_labels.to(device))
                        train_predict=torch.max(train_outputs,dim=1)[1]
                        train_acc+=torch.eq(train_predict, train_labels.to(device)).sum().item()                
                        train_loss+=tmp_train_loss.item()
                        train_bar.desc = "valid in train_dataset epoch[{}/{}]".format(epoch + 1,epochs)
                    
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        val_outputs = net(val_images.to(device))
                        tmp_val_loss = loss_function(val_outputs, val_labels.to(device))
                        val_predict = torch.max(val_outputs, dim=1)[1]
                        val_acc += torch.eq(val_predict, val_labels.to(device)).sum().item()
                        val_loss+=tmp_val_loss.item()               
                        val_bar.desc = "valid in val_dataset epoch[{}/{}]".format(epoch + 1,epochs)

                train_accurate=train_acc/train_num
                val_accurate = val_acc / val_num

                if(val_accurate > best_acc):
                    best_acc = val_accurate
                    if val_accurate > total_best_acc:
                        total_best_acc = val_accurate
                        best_lr = lr
                        best_batch = batch_size
                        torch.save(net.state_dict(), firstmodepth) # 保存模型
                print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f' 
                %(epoch + 1, train_loss / train_num, train_accurate, val_loss/val_num, val_accurate))

            print(f"当learning_rate为{lr},batch size为{batch_size}时,验证集最高的准确率为{best_acc}")
            best_acc_list.append(best_acc)

        x = range(len(learningrates))
        plt.plot(x, best_acc_list, marker='o', label=f'batch size: {batch_size}')

    plt.xlabel('learning rate')
    plt.ylabel('the accuracy of validation_dataset')
    plt.xticks(x, learningrates)
    # 添加标题
    plt.title('train1 batch sizes and learning rates')
    # 添加图例
    plt.legend()
    plt.savefig(picture_save_path)
    plt.show()
    
    print(f"最好的模型是learning_rate为{best_lr},batch size为{best_batch},此时验证集最高的准确率为{total_best_acc}")