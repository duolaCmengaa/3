<div align="center">

# 计算机视觉期中作业

李波                

21307110183



</div>

## 任务描述：

微调在ImageNet上预训练的卷积神经网络实现鸟类识别，在[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)数据集上进行训练以实现图像分类。

## 准备:
我们需要前往[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)下载数据集,解压后会产生CUB_200_2011文件夹，结构如下：
```plaintext
├───CUB_200_2011
│   ├───attributes
│   ├───images
│   ├───parts
│   ├───bounding_boxes.txt
│   ├───classes.txt
│   ├───image_class_labels.txt
│   ├───images.txt
│   ├───train_test_split.txt
│   └───README

```

其中，三个Tensorboard的日志文件在result文件夹里，我还将模型权重和Tensorboard的日志文件存放到了网盘[模型权重](https://drive.google.com/drive/folders/1hJrliYm0wZz6FnxXUPgeT6-CUJNMBktr?usp=sharing)需要可以自行下载，其中最好的经过预训练的模型权重为firstmodepth_3_1.pth，最好的没有经过预训练的模型权重为no_pre_model1.pth，firstmodepth_1.pth为固定预训练模型的特征提取部分，只对最后一层进行训练所得到的最好的模型权重，用来进行train2.py，train3.py的训练，firstmodepth_3.pth为使用load_data.py文件里的get_data()函数的数据增强方法transforms1所得到的最好的经过预训练的模型权重，firstmodepth_3_1.pth使用的是transforms2方法

将模型权重，CUB数据集，日志文件按照文件存放目录结构进行存放就可以进行正常的训练和测试
## 文件存放目录结构:

```plaintext
├───CUB_200_2011
├───dataset.py
├───load_data.py
├───print_data.py
├───train1.py
├───train2.py
├───train3.py
├───test.py
├───train_no_pre.py
├───train_no_pre1.py
├───transforms1.py
├───mine_model
│   ├───alex_mine.py
│   ├───res_mine.py
├───result
│   ├───model
│   │   ├───firstmodepth_1.pth
│   │   ├───firstmodepth_3.pth
│   │   ├───firstmodepth_3_1.pth
│   │   ├───no_pre_model1.pth
│   ├───pirtures
│   │   ├───no_pre_train_batch_sizes_learningrates.png
│   │   ├───no_pre_train_batch_sizes_learningrates1.png
│   │   ├───no_pre_train_weight_decay_momentum (1).png
│   │   ├───no_pre_train_weight_decay_momentum.png
│   │   ├───train1_batch_sizes_learningrates.png
│   │   ├───train2_batch_sizes_learningrates.png
│   │   ├───train2_batch_sizes_learningrates1.png
│   │   ├───train2_weight_decay_momentum (1).png
│   │   ├───train2_weight_decay_momentum.png
│   ├───best_pre_log1
│   ├───best_no_pre_log1
│   └───best_pre_log1_1

```

## 模型训练部分：
train1.py，train2.py，train3.py为预训练模型的训练部分，其中train1.py为固定预训练模型的特征提取部分，只对最后一层进行训练的部分，设置了不同的learningrates和batch_sizes组合，然后train2.py是以train1.py得到的模型权重来进行训练，设置了不同的learningrates和batch_sizes组合，然后train3.py是以train1.py得到的模型权重和train2.py得到的learningrates和batch_sizes最优组合来进行训练，设置了不同的weight_decay和momentum组合，其中三个文件都可以直接进行运行。

train_no_pre.py，train_no_pre1.py为没有经过预训练的模型的训练部分，其中train_no_pre.py设置了不同的learningrates和batch_sizes组合，train_no_pre1.py是以train_no_pre.py得到的learningrates和batch_sizes最优组合来进行训练，设置了不同的weight_decay和momentum组合，其中两个文件都可以直接进行运行。
## 测试部分：
直接运行test.py，要注意修改load_path，load_path为要测试的模型的权重地址，其中要注意的是firstmodepth_1.pth，no_pre_model1.pth和firstmodepth_3.pth使用的是load_data.py文件里的get_data()函数的数据增强方法transforms1，firstmodepth_3_1.pth使用的是load_data.py文件里的get_data()函数的数据增强方法transforms2，运行test.py之前要注意把load_data.py文件里的get_data()函数的数据增强方法改为相对应的方法(transforms1,transforms2,transforms3)

## Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化：
下载好best_pre_log1,best_no_pre_log1,best_pre_log1_1，best_pre_log1为firstmodepth_3.pth的日志，best_pre_log1_1为firstmodepth_3_1.pth的日志，best_no_pre_log1为no_pre_model1.pth的日志

打开命令行或终端，将工作路径导航到日志文件所存放的目录，以best_pre_log1为例，在终端直接运行tensorboard --logdir=best_pre_log1就可以实现Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化


## 其他的文件说明：
mine_model文件夹存放了我自己写的两个模型，alex_mine.py为AlexNet，res_mine.py为ResNet，都可以正常导入来进行训练

print_data.py可以打印训练集，验证集，测试集的特点

