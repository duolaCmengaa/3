from dataset import CUB
from torch.utils.data import DataLoader
import os 
import transforms1 
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Subset
from collections import defaultdict

def get_class_distribution1(subset):
    class_distribution = defaultdict(int)
    for idx in subset.indices:
        _, class_id = subset.dataset[idx]  # 获取原始数据集中的类别标签
        class_id+=1
        class_distribution[class_id] += 1

    # 将字典按键（类别标签）排序
    sorted_class_distribution = dict(sorted(class_distribution.items()))
    return sorted_class_distribution




def get_data(batch_size):


    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据集解压后的目录
    dataset_dir = os.path.join(script_dir,'CUB_200_2011')
    images_dir = os.path.join(dataset_dir, 'images')
    path = dataset_dir
 
    # transforms1
    IMAGE_SIZE = 224
    IMAGE_SIZE1 = 256
    train_transforms = transforms1.Compose([
            transforms1.ToCVImage(),
            transforms1.RandomResizedCrop(IMAGE_SIZE),
            transforms1.RandomHorizontalFlip(),
            transforms1.ToTensor(),
            transforms1.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    test_transforms = transforms1.Compose([
        transforms1.ToCVImage(),
        transforms1.Resize(IMAGE_SIZE1),
        transforms1.CenterCrop(IMAGE_SIZE),
        transforms1.ToTensor(),
        transforms1.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    
    '''
    # transforms2
    IMAGE_SIZE = 448
    IMAGE_SIZE1 = 512
    train_transforms = transforms1.Compose([
    transforms1.ToCVImage(),                 # 转换为合适的图像格式
    transforms1.Resize(IMAGE_SIZE1),         # 调整图像大小到512
    transforms1.RandomResizedCrop(IMAGE_SIZE), # 随机裁剪到448
    transforms1.RandomHorizontalFlip(),      # 随机水平翻转
    transforms1.ToTensor(),                  # 转换为张量
    transforms1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ])

    test_transforms = transforms1.Compose([
        transforms1.ToCVImage(),                 # 转换为合适的图像格式
        transforms1.Resize(IMAGE_SIZE1),         # 调整图像大小到512
        transforms1.CenterCrop(IMAGE_SIZE),      # 中心裁剪到448
        transforms1.ToTensor(),                  # 转换为张量
        transforms1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ])
    '''

    '''
    transforms3
    IMAGE_SIZE = 448
    IMAGE_SIZE1 = 512

    train_transforms = transforms1.Compose([
        transforms1.ToCVImage(),                 # 转换为合适的图像格式
        transforms1.Resize(IMAGE_SIZE1),         # 调整图像大小到 512
        transforms1.RandomResizedCrop(IMAGE_SIZE), # 随机裁剪到 448
        transforms1.RandomHorizontalFlip(),      # 随机水平翻转
        transforms1.RandomPerspective(distortion_scale=persp_distortion_scale, p=0.5, interpolation=3), # 随机透视变换
        transforms1.RandomRotation(rotation_range, expand=False, center=None, fill=None),  # 随机旋转
        transforms1.ToTensor(),                  # 转换为张量
        transforms1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ])

    test_transforms = transforms1.Compose([
        transforms1.ToCVImage(),                 # 转换为合适的图像格式
        transforms1.Resize(IMAGE_SIZE1),         # 调整图像大小到 512
        transforms1.CenterCrop(IMAGE_SIZE),      # 中心裁剪到 448
        transforms1.ToTensor(),                  # 转换为张量
        transforms1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ])
    '''

    batch_size1=batch_size

    nw = min([os.cpu_count(), batch_size1 if batch_size1 > 1 else 0, 8])
    train_dataset = CUB(
            path,
            train=True,
            transform=train_transforms,
            target_transform=None
        )
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size1,
        num_workers=nw,
        shuffle=True
    )


    fulltest_dataset = CUB(
            path,
            train=False,
            transform=test_transforms,
            target_transform=None
        )




    # 划分验证集和测试集
    # 提取标签用于分层抽样
    data_ids = fulltest_dataset.data_id
    labels = [int(fulltest_dataset._get_class_by_id(id_)) for id_ in data_ids]

    # 进行分层抽样
    split_ratio = 0.5
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)
    test_idx, val_idx = next(sss.split(data_ids, labels))

    validation_dataset = Subset(fulltest_dataset, val_idx)
    test_dataset = Subset(fulltest_dataset, test_idx)





    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size1,
        num_workers=nw,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size1,
        num_workers=nw,
        shuffle=False
    )



    return train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,fulltest_dataset





def print_data(train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,fulltest_dataset):


    print("用于训练的数据：",len(train_dataset))
    print("用于测试的数据：",len(fulltest_dataset))
    print()

    # 打印用来训练的数据和用来测试的数据的分布
    train_class_distribution = train_dataset.get_class_distribution()
    fulltest_class_distribution = fulltest_dataset.get_class_distribution()

    print("Data For Train Distribution:", train_class_distribution)
    print()
    print("Data For Test Distribution:", fulltest_class_distribution)
    print()



    # 计算类别分布
    val_class_distribution = get_class_distribution1(validation_dataset)
    test_class_distribution = get_class_distribution1(test_dataset)

    print("验证集类别分布：", val_class_distribution)
    print()

    print("测试集类别分布：", test_class_distribution)
    print()

    print("训练集大小：",len(train_dataset))
    print("验证集大小：",len(validation_dataset))
    print("测试集大小：",len(test_dataset))
    print()

    print("train_dataloader:",len(train_dataloader))
    print("validation_dataloader:",len(validation_dataloader))
    print("test_dataloader:",len(test_dataloader))
    print()



    image, class_id=train_dataset.__getitem__(1)
    print(f"训练集例子size:{image}\tlabel:{class_id+1}")

    image, class_id=validation_dataset.__getitem__(0)
    print(f"验证集例子size:{image}\tlabel:{class_id+1}")

    image, class_id=test_dataset.__getitem__(0)
    print(f"测试集例子size:{image}\tlabel:{class_id+1}")

    # 打印训练集的一个数据
    image, class_id = train_dataset.__getitem__(1)
    print(f"训练集例子大小：{image.size()}\t标签:{class_id+1}")

    # 打印验证集的一个数据
    image, class_id = validation_dataset.__getitem__(0)
    print(f"验证集例子大小：{image.size()}\t标签:{class_id+1}")

    # 打印测试集的一个数据
    image, class_id = test_dataset.__getitem__(0)
    print(f"测试集例子大小：{image.size()}\t标签:{class_id+1}")

    return 