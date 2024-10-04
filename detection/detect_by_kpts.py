import torch
import numpy as np

from model import ModifiedResNet

def Model():
    # 加载模型
    model_path = 'modified_resnet_model.pth'
    model = ModifiedResNet(num_classes=2)  # 确保使用相同的模型结构
    model.load_state_dict(torch.load(model_path))
    return model


def detect(frame):
    model=Model()
    model.eval()  # 将模型设置为评估模式
    # 假设你有一个2x30x17的输入数据
    input_data = frame.astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    # 添加一个批次维度，因为PyTorch期望批次维度
    input_tensor = input_tensor.unsqueeze(0)
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_tensor)
        predicted=output[:,1].item()
        print(predicted)   


frame=np.random.rand(2,30,17)
detect(frame)