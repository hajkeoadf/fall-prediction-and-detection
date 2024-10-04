from model import ModifiedResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os 
import pickle as pkl
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

#读取文件的关键点坐标

label_dir = str(os.getcwd())+'/label'

file_name='adl.pkl'
file_path = os.path.join(label_dir, file_name)
with open(file_path, 'rb') as f:
    adl_arr=pkl.load(f)

file_name='fall.pkl'
file_path = os.path.join(label_dir, file_name)
with open(file_path, 'rb') as f:
    fall_arr=pkl.load(f)

# fall_arr=fall_arr.transpose((0, 3, 1, 2))
# adl_arr=adl_arr.transpose((0, 3, 1, 2))


fall_lab_arr=np.ones((fall_arr.shape[0],))
adl_lab_arr=np.zeros((adl_arr.shape[0],))


data_array=np.concatenate((fall_arr, adl_arr), axis=0)
label_array=np.concatenate((fall_lab_arr, adl_lab_arr), axis=0)

# data_array=np.random.rand(1000,2,30,17)
# label_array= np.random.randint(0, 1, size=(1000,))

data_tensor=torch.from_numpy(data_array)
labels_tensor= torch.from_numpy(label_array)

data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
labels_tensor = torch.tensor(labels_tensor, dtype=torch.long)


# 实例化数据集
dataset = CustomDataset(data_tensor, labels_tensor)
# 分割数据集为训练集和验证集
from torch.utils.data import random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# # 数据加载器
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)



# 实例化模型
model = ModifiedResNet(num_classes=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# 初始化最佳模型的性能指标
best_accuracy = 0.0
best_model_weights = None

# 训练和评估模型
for epoch in range(num_epochs):
    # 训练模型
    model.train()  # 确保模型处于训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 评估模型
    model.eval()  # 确保模型处于评估模式
    val_loss = 0
    correct = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():  # 在评估过程中不计算梯度
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predicted=outputs[:,1]
            threshold=0.5
            predicted=predicted>threshold
            all_predicted.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())
            correct += (predicted == labels).sum().item()

    # 计算验证集上的准确率
    val_loss /= len(val_loader.dataset)
    accuracy = (correct / len(val_loader.dataset)) * 100
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # 检查是否是最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_weights = model.state_dict().copy()


# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_predicted)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['ADL', 'FALL'], yticklabels=['ADL', 'FALL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('kpts_cm.jpg')
plt.close()

best_model_path = 'best_modified_resnet_model.pth'
last_model_path = 'last_modified_resnet_model.pth'

torch.save(best_model_weights, best_model_path)
torch.save(model.state_dict(), last_model_path)
print(f'Best Model saved to {best_model_path}')
print(f'Last Model saved to {last_model_path}')