import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取Excel文件中的数据
file_path = 'dataset/annotations/svm.xlsx'
df = pd.read_excel(file_path, sheet_name='manual-WYH')

# 假设数据列按照如下顺序排列：特征1, 特征2, ..., 特征N, 标签
# 根据实际数据列进行调整
feature_columns = df.columns[:-1]  # 获取除了最后一列的所有列名作为特征列
label_column = df.columns[-1]      # 最后一列为标签列

# 提取特征和标签
X = df[feature_columns].values
y = df[label_column].values

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM模型
svm_classifier = SVC(kernel='linear')  # 使用线性核函数

# 训练模型
svm_classifier.fit(X_train, y_train)

# 在测试集上进行预测
predictions = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# 输出混淆矩阵
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)