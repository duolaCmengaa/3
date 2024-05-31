# 找到了最好的学习率和batch size,接下来对weight_decay和momentum进行查找
import matplotlib.pyplot as plt #plt 用于显示图片
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import os
from load_data import *
from torch.utils.tensorboard import SummaryWriter
import shutil
from tqdm import tqdm

def save_best_log(src_log_dir, dest_log_dir):
    if os.path.exists(dest_log_dir):
        shutil.rmtree(dest_log_dir)
    shutil.copytree(src_log_dir, dest_log_dir)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    model_path = os.path.join(result_dir, 'model')
    os.makedirs(model_path, exist_ok=True)

    log_dir = os.path.join(result_dir, 'no_pre_log1')
    best_log_dir=os.path.join(result_dir, 'best_no_pre_log1')
    model_save_path = os.path.join(model_path, 'no_pre_model1.pth')
    picture_dir = os.path.join(result_dir, 'pirtures')

    os.makedirs(picture_dir, exist_ok=True)

    picture_save_path = os.path.join(picture_dir, "no_pre_train_weight_decay_momentum")

    max_decreasing_count = 5
    total_best_acc = 0
    
    best_lr = 0.02
    best_batch = 8
    best_wd = 0
    best_mom = 0
 
    train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,_ = get_data(best_batch)
    '''
    wds = [5e-3, 1e-3, 5e-4, 1e-4]
    moms = [0, 0.5, 0.9]
    '''

    wds = [2e-4,9e-5,5e-5,1e-5]
    moms = [0.5, 0.9,0.95]

    for mom in moms:
        best_acc_list = []

        for wd in wds:
                tmp=0 # 记录是否为最好成绩                          
                val_accuracy_list = []
                train_accuracy_list=[]
                epochs_list = []
                train_loss_list = []
                val_loss_list = []

                # 创建一个与保存模型参数的模型结构相同的模型
                model1 = models.resnet18(weights=None)
                classes = range(200)
                    # 获取全连接层的输入特征数
                in_features = model1.fc.in_features

                # 修改最后一层为新的全连接层
                model1.fc = nn.Linear(in_features, len(classes))

                model1.to(device=device)  # 将模型移动到设备上
                net = model1

                for param in net.parameters(): # 所有参数设计为可训练
                    param.requires_grad = True

                optimizer = optim.SGD(net.parameters(), lr=best_lr, weight_decay=wd, momentum=mom)
                # 添加学习率调整器
                scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
                epochs = 90
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
                # 初始化 SummaryWriter
                if os.path.exists(log_dir):
                        shutil.rmtree(log_dir)
                writer = SummaryWriter(log_dir)
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

                    # 调整学习率
                    scheduler.step()

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
                    

                    # 构造各个参数的列表，准备画图
                    val_accuracy_list.append(val_accurate)
                    train_accuracy_list.append(train_accurate)
                    train_loss_list.append(train_loss / train_num)
                    val_loss_list.append(val_loss/ val_num )
                    epochs_list.append(epoch+1)

                    writer.add_scalar('Loss/train', train_loss / train_num, epoch + 1)
                    writer.add_scalar('Loss/val', val_loss / val_num, epoch + 1)
                    writer.add_scalar('Accuracy/train', train_accurate, epoch + 1)
                    writer.add_scalar('Accuracy/val', val_accurate, epoch + 1)

                    writer.add_scalars('Loss', {'test':val_loss / val_num,'train':train_loss / train_num}, epoch + 1)
                    writer.add_scalars('Accuracy', {'test':val_accurate,'train':train_accurate}, epoch + 1)
                    if(val_accurate > best_acc):
                        best_acc = val_accurate

                        decreasing_count = 0  # 如果当前准确率比最佳准确率更好，则重置连续下降次数计数器
                        if val_accurate > total_best_acc:
                            total_best_acc = val_accurate
                            best_wd = wd
                            best_mom = mom
                            tmp=1
                            torch.save(net.state_dict(), model_save_path) # 保存模型


                        else:
                            pass
                    else:
                        decreasing_count += 1  # 否则增加连续下降次数计数器
                    print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f' 
                           %(epoch + 1, train_loss / train_num, train_accurate, val_loss/val_num, val_accurate))
                    # 如果连续下降次数超过阈值，则停止训练
                    if decreasing_count >= max_decreasing_count:
                        print(f"连续{max_decreasing_count}个epoch验证集准确率下降,停止训练。")
                        break
                if tmp ==1:
                    save_best_log(log_dir, best_log_dir) 
                # 关闭 SummaryWriter
                writer.close()
                print(f"当learning_rate为{best_lr},batch size为{best_batch}时,weight_decay为{wd}时,momentum为{mom}时,验证集最高的准确率为{best_acc}")
                best_acc_list.append(best_acc)

        x = range(len(wds))
        plt.plot(x, best_acc_list, marker='o', label=f'momentum: {mom}')

    plt.xlabel('weight_decay')
    plt.ylabel('the accuracy of validation_dataset')
    plt.xticks(x, wds)
    # 添加标题
    plt.title('no pretrain2 weight decay and momentum')
    # 添加图例
    plt.legend()
    plt.savefig(picture_save_path)
    plt.show()

    print(f"最好的模型是learning_rate为{best_lr},batch size为{best_batch}时,weight_decay为{best_wd}时,momentum为{best_mom},此时验证集最高的准确率为{total_best_acc}")


