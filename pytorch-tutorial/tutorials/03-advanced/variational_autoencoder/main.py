#  对 MNIST 手写数字数据集进行学习和图像生成

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists 如果目录不存在则创建它
sample_dir = 'VAE-samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 784  # 图像大小，MNIST图像为28*28=784
h_dim = 400  # 隐藏层维度
z_dim = 20  # 潜在空间维度
num_epochs = 15  # 训练轮数
batch_size = 128  # 批次大小
learning_rate = 1e-3  # 学习率

# MNIST dataset 加载MNIST数据集
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader 创建数据加载器
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# VAE model 定义变分自动编码器（VAE）模型
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)  # 全连接层，将图像向量映射到隐藏层
        self.fc2 = nn.Linear(h_dim, z_dim)  # 全连接层，输出潜在空间的均值
        self.fc3 = nn.Linear(h_dim, z_dim)  # 全连接层，输出潜在空间的对数方差
        self.fc4 = nn.Linear(z_dim, h_dim)  # 全连接层，从潜在空间映射回隐藏层
        self.fc5 = nn.Linear(h_dim, image_size)  # 全连接层，输出重构后的图像向量

    def encode(self, x):
        h = F.relu(self.fc1(x))  # 对第一层全连接层的输出应用ReLU激活函数
        return self.fc2(h), self.fc3(h)  # 返回潜在空间的均值和对数方差

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)  # 计算标准差
        eps = torch.randn_like(std)  # 生成与标准差同形状的随机噪声
        return mu + eps * std  # 重参数化技巧，得到潜在变量z

    def decode(self, z):
        h = F.relu(self.fc4(z))  # 对从潜在空间映射回的隐藏层输出应用ReLU激活函数
        return F.sigmoid(self.fc5(h))  # 对最后一层全连接层的输出应用Sigmoid激活函数，得到重构图像

    def forward(self, x):
        mu, log_var = self.encode(x)  # 对输入图像进行编码
        z = self.reparameterize(mu, log_var)  # 重参数化得到潜在变量z
        x_reconst = self.decode(z)  # 对潜在变量z进行解码得到重构图像
        return x_reconst, mu, log_var  # 返回重构图像、潜在空间的均值和对数方差


model = VAE().to(device)  # 实例化VAE模型并移动到指定设备（GPU或CPU）上
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义Adam优化器

# Start training 开始训练
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward pass 前向传播
        x = x.to(device).view(-1, image_size)  # 将输入图像展平为向量并移动到设备上
        x_reconst, mu, log_var = model(x)  # 得到重构图像、潜在空间的均值和对数方差

        # Compute reconstruction loss and kl divergence 计算重构损失和KL散度
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(
            x_reconst, x, size_average=False)  # 计算重构损失
        kl_div = - 0.5 * torch.sum(1 + log_var -
                                   mu.pow(2) - log_var.exp())  # 计算KL散度

        # Backprop and optimize 反向传播和优化
        loss = reconst_loss + kl_div  # 总损失为重构损失和KL散度之和
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        if (i+1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}],重构损失 Reconst Loss: {:.4f},KL 散度 KL Div: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        # Save the sampled images 保存生成的样本图像
        z = torch.randn(batch_size, z_dim).to(device)  # 生成随机噪声向量
        out = model.decode(z).view(-1, 1, 28, 28)  # 对噪声向量解码并恢复图像形状
        save_image(out, os.path.join(
            sample_dir, 'sampled-{}.png'.format(epoch+1)))  # 保存生成的图像

        # Save the reconstructed images 保存重构后的图像
        out, _, _ = model(x)  # 再次得到重构图像
        x_concat = torch.cat(
            [x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)  # 将原始图像和重构图像在维度3上拼接
        save_image(x_concat, os.path.join(
            sample_dir, 'reconst-{}.png'.format(epoch+1)))  # 保存拼接后的图像
