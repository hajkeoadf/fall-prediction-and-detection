import os
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import copy
# 数据集类
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        for label in ['FALL', 'ADL']:
            for img_name in os.listdir(os.path.join(self.img_dir, label)):
                self.image_paths.append((os.path.join(self.img_dir, label, img_name), 1 if label == 'FALL' else 0))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((400, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 实例化数据集
img_dir = 'img'  # 替换为你的图像路径
dataset = CustomDataset(img_dir, transform=data_transforms)

# 分割数据集为训练集和验证集
from torch.utils.data import random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 构建模型
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

model = ModifiedResNet(num_classes=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和评估模型
num_epochs = 5
best_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 评估模型
    model.eval()
    val_loss = 0
    correct = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())
            correct += (predicted == labels).sum().item()

    accuracy = (correct / len(val_loader.dataset)) * 100
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_weights = copy.deepcopy(model.state_dict())

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_predicted)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['ADL', 'FALL'], yticklabels=['ADL', 'FALL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('mhi_cm.jpg')
plt.close()

# 保存最佳模型和最后模型的权重
torch.save(best_model_weights, 'mhi_best_model.pth')
torch.save(model.state_dict(), 'mhi_last_model.pth')