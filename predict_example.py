# 载入模型
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(num_ftrs, 200)
model.load_state_dict(torch.load('resnet18_cub.pth'))
model = model.to(device)

# 切换到评估模式
model.eval()

# 加载并预处理单张图像进行预测
from PIL import Image

def predict_image(image_path):
    image = Image.open(image_path)
    image = data_transforms['val'](image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds[0]]

# 示例
print(predict_image('/path/to/test/image.jpg'))
