import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 定义一个简单的模型


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# 初始化模型
model = SimpleModel()

# 检查是否有多个GPU可用
if torch.cuda.device_count() > 1:
    print(f"发现 {torch.cuda.device_count()} 个GPU，使用 DataParallel。")
    model = nn.DataParallel(model)

# 将模型移动到GPU上
model.to(torch.device("cuda"))

# 生成一些虚拟数据
x = torch.randn(100, 10).to(torch.device("cuda"))
y = torch.randint(0, 2, (100,)).to(torch.device("cuda"))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
