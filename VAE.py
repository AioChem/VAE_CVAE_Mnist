import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image


# 变分自编码器
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        # 编码器层
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, latent_size)
        self.fc3 = nn.Linear(512, latent_size)

        # 解码器层
        self.fc4 = nn.Linear(latent_size, 512)
        self.fc5 = nn.Linear(512, input_size)

    # 编码器部分
    def encode(self, x):
        x = F.relu(self.fc1(x))  # 编码器的隐藏表示

        mu = self.fc2(x)  # 潜在空间均值
        log_var = self.fc3(x)  # 潜在空间对数方差
        return mu, log_var

    # 重参数化
    def reparameterize(self, mu, log_var):  # 从编码器输出的均值和对数方差中采样得到潜在变量z
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样得到随机噪声
        return mu + eps * std  # 根据重参数化公式计算潜在变量z

    # 解码器部分
    def decode(self, z):
        z = F.relu(self.fc4(z))  # 将潜在变量 z 解码为重构图像
        return torch.sigmoid(self.fc5(z))  # 将隐藏表示映射回输入图像大小，并应用 sigmoid 激活函数，以产生重构图像

    # 前向传播
    def forward(self, x):  # 输入图像 x
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


# 使用重构损失和 KL 散度作为损失函数
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x.view(-1, input_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD  # 返回二进制交叉熵损失和 KLD 损失的总和作为最终的损失值


if __name__ == '__main__':
    # TODO设定参数
    batch_size = 512  # 批次
    epochs = 20  # 学习轮次
    sample_interval = 10  # 保存结果的轮次
    learning_rate = 1e-3  # 学习率
    input_size = 784  # 输入维度  (28*28)
    latent_size = 256  # 潜在变量维度   小于初始向量维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO加载手写数字数据集
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 将图像转换为张量
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # TODO定义模型
    model = VAE(input_size, latent_size).to(device)
    # TODO优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # TODO开始训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            # TODO处理data数据
            data = data.to(device)
            data = data.view(-1, input_size)

            # TODO进行计算
            predict, mu, log_var = model(data)
            loss = loss_function(predict, data, mu, log_var)  # 计算损失
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)  # # 计算每个周期的训练损失
        print('Epoch [{}/{}], Loss: {:.2f}'.format(epoch + 1, epochs, train_loss))

        # TODO每10轮训练，进行一次图像生成，保存模型参数
        if (epoch + 1) % sample_interval == 0:
            torch.save(model.state_dict(), f'./out_data/VAE/vae{epoch + 1}.pth')
            model.eval()
            with torch.no_grad():
                pic_num = 10
                sample = torch.randn(pic_num, latent_size).to(device)
                sample_img = model.decode(sample).cpu()  # 将随机样本输入到解码器中，解码器将其映射为图像
                save_image(sample_img.view(pic_num, 1, 28, 28), f'./out_data/VAE/sample{epoch}.png', nrow=int(pic_num / 2))
