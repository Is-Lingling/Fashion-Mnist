import torch
from torch import nn, optim
from net import FashionNet
from get_data import load_fashion_train_data
from torch.utils.data import DataLoader

# 读取数据
train_dataset = load_fashion_train_data()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 检查环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义网络
net = FashionNet().to(device)

# 定义损失函数
loss_c = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 定义训练批次
epochs = 5

for epoch in range(epochs+1):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_c(outputs,labels)
        loss.backward()
        optimizer.step()
        print(f'finsh:{epoch},loss:{loss:.4f}')
    print(f'finsh:{epoch}')

torch.save(net.state_dict(), './parmar.pth')
