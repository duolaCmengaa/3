<div align="center">

# 计算机视觉期中作业

李波                周语诠

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
│   ├───README
│   └───

```

其中，模型权重和Tensorboard的日志文件在result文件夹里，我还将它们存放到了网盘[模型权重](https://drive.google.com/drive/folders/1hJrliYm0wZz6FnxXUPgeT6-CUJNMBktr?usp=sharing)需要可以自行下载，其中firstmodepth_3.pth为预训练模型的权重，no_pre_model1.pth为未经预训练的模型权重

## 文件存放目录结构:

```plaintext
├───CUB_200_2011
├───dataset.py
├───load_data.py
├───pre_train1.py
├───pre_train2.py
├───pre_train3.py
├───predict_example.py
├───test.py
├───train_no_pre.py
├───train_no_pre1.py
├───transforms1.py
├───result
│   ├───model
│   │   ├───firstmodepth_3.pth
│   │   ├───no_pre_model1.pth
│   ├───pirtures
│   │   ├───no_pre_model1.pth
│   │   ├───no_pre_model1.pth
│   │   ├───no_pre_model1.pth
│   │   ├───no_pre_model1.pth
│   │   ├───no_pre_model1.pth
│   ├───best_pre_log1
│   ├───best_no_pre_log1
│   └───

```

## 模型训练部分：
pre_train1.py，pre_train2.py，pre_train3.py为预训练模型的训练部分，其中pre_train1.py为固定预训练模型的特征提取部分，只对最后一层进行训练的部分，设置了不同的learningrates和batch_sizes组合，然后pre_train2.py是以pre_train1.py得到的模型权重来进行训练，设置了不同的learningrates和batch_sizes组合，然后pre_train3.py是以pre_train1.py得到的模型权重以pre_train2.py得到的learningrates和batch_sizes最优组合来进行训练，设置了不同的weight_decay和momentum组合，其中三个文件都可以直接进行运行。

## 参数查找部分：

直接运行find_best_model.py文件可以在候选的一些学习率、两层隐藏层大小、正则化强度，激活函数类型等超参数中进行训练，从而找到最优的超参数。找到各自的最好的超参数后会自动以最优的参数进行训练，最后会将权重文件保存到result文件夹内的best_model文件夹内。

## 测试部分：
直接运行test.py，要注意修改要测试的模型的权重地址

## 绘制loss曲线和accuracy曲线部分：
下载好测试部分所下载的文件，并放到正确的路径，直接运行plot_loss_accuracy.py即可，图片会自动保存到result文件夹内的pictures文件夹内。

## 模型网络参数的可视化部分：
下载好模型权重部分(model.npy),放置到正确的路径，直接运行visualization_parameters.py即可，图片会自动保存到result文件夹内的pictures文件夹内。

## 其他的文件说明：
function1.py包含了定义的一些模型所需要的基本函数

model1.py是模型的基本结构
