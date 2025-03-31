import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64  # 潜在空间的维度
hidden_size = 256  # 隐藏层的大小
image_size = 784  # 图像的大小（28*28 = 784，MNIST图像尺寸）
num_epochs = 200  # 训练的轮数
batch_size = 100  # 批次大小
sample_dir = 'samples'  # 保存生成图像的目录

# Create a directory if not exists 如果目录不存在则创建
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing 图像预处理
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],   # 1 for greyscale channels MNIST是灰度图像
                         std=[0.5])])

# MNIST dataset 加载MNIST数据集
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader 创建数据加载器
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# Discriminator 定义判别器
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),  # 全连接层，将图像向量映射到隐藏层
    nn.LeakyReLU(0.2),  # LeakyReLU激活函数
    nn.Linear(hidden_size, hidden_size),  # 第二个全连接层
    nn.LeakyReLU(0.2),  # 第二个LeakyReLU激活函数
    nn.Linear(hidden_size, 1),  # 输出层，输出一个标量
    nn.Sigmoid())  # Sigmoid激活函数，将输出值映射到0到1之间

# Generator 定义生成器
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),  # 全连接层，将潜在向量映射到隐藏层
    nn.ReLU(),  # ReLU激活函数
    nn.Linear(hidden_size, hidden_size),  # 第二个全连接层
    nn.ReLU(),  # 第二个ReLU激活函数
    nn.Linear(hidden_size, image_size),  # 输出层，生成与图像大小相同的向量
    nn.Tanh())  # Tanh激活函数，将输出值映射到-1到1之间

# Device setting 将判别器和生成器移动到指定设备（GPU或CPU）上
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer 定义二元交叉熵损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)  # 判别器的Adam优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)  # 生成器的Adam优化器

# 反归一化函数，将图像值从[-1, 1]转换回[0, 1]


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# 重置梯度的函数
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training 开始训练
total_step = len(data_loader)  # 总步数，即数据加载器中的批次数量
for epoch in range(num_epochs):  # 遍历每一轮训练
    for i, (images, _) in enumerate(data_loader):  # 遍历每个批次
        images = images.reshape(batch_size, -1).to(device)  # 将图像展平为向量并移动到设备上

        # Create the labels which are later used as input for the BCE loss 创建标签，用于计算二元交叉熵损失
        real_labels = torch.ones(batch_size, 1).to(device)  # 真实图像的标签，全为1
        fake_labels = torch.zeros(batch_size, 1).to(device)  # 虚假图像的标签，全为0

        # ================================================================== #
        #                      Train the discriminator   训练判别器                      #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x)) 使用真实图像计算二元交叉熵损失
        # Second term of the loss is always zero since real_labels == 1 因为real_labels == 1，所以损失函数的第二项始终为0
        outputs = D(images)  # 判别器对真实图像的输出
        d_loss_real = criterion(outputs, real_labels)  # 真实图像的损失
        real_score = outputs  # 判别器对真实图像的评分

        # Compute BCELoss using fake images 使用虚假图像计算二元交叉熵损失
        # First term of the loss is always zero since fake_labels == 0 因为fake_labels == 0，所以损失函数的第一项始终为0
        z = torch.randn(batch_size, latent_size).to(device)  # 生成随机噪声向量
        fake_images = G(z)  # 生成器生成的虚假图像
        outputs = D(fake_images)  # 判别器对虚假图像的输出
        d_loss_fake = criterion(outputs, fake_labels)  # 虚假图像的损失
        fake_score = outputs  # 判别器对虚假图像的评分

        # Backprop and optimize 反向传播和优化
        d_loss = d_loss_real + d_loss_fake  # 判别器的总损失
        reset_grad()  # 重置梯度
        d_loss.backward()  # 反向传播计算梯度
        d_optimizer.step()  # 更新判别器的参数

        # ================================================================== #
        #                        Train the generator        训练生成器                      #
        # ================================================================== #

        # Compute loss with fake images 使用虚假图像计算损失
        z = torch.randn(batch_size, latent_size).to(device)  # 生成随机噪声向量
        fake_images = G(z)  # 生成器生成的虚假图像
        outputs = D(fake_images)  # 判别器对虚假图像的输出

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z))) 我们训练生成器来最大化log(D(G(z)))，而不是最小化log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf 原因见论文：https://arxiv.org/pdf/1406.2661.pdf 第3节的最后一段
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize 反向传播和优化
        reset_grad()  # 重置梯度
        g_loss.backward()  # 反向传播计算梯度
        g_optimizer.step()  # 更新生成器的参数

        if (i+1) % 200 == 0:  # 每200步打印一次训练信息
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Save real images  保存真实图像
    if (epoch+1) == 1:  # 在第一轮训练结束时
        images = images.reshape(images.size(0), 1, 28, 28)  # 将图像恢复为原始形状
        save_image(denorm(images), os.path.join(
            sample_dir, 'real_images.png'))  # 保存真实图像

    # Save sampled images 保存生成的图像
    fake_images = fake_images.reshape(
        fake_images.size(0), 1, 28, 28)  # 将生成的图像恢复为原始形状
    save_image(denorm(fake_images), os.path.join(
        sample_dir, 'fake_images-{}.png'.format(epoch+1)))  # 保存生成的图像

# Save the model checkpoints 保存模型参数
torch.save(G.state_dict(), 'G.ckpt')  # 保存生成器的参数
torch.save(D.state_dict(), 'D.ckpt')  # 保存判别器的参数
