'''import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(epoch):
    network.train()
    print("开始咯")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        print("111")
        optimizer.zero_grad()
        print("112")
        output = network(data)
        print("113")
        loss = F.nll_loss(output, target)
        loss = loss.cuda()
        print("114")
        loss.backward()
        print("115")
        optimizer.step()
        print("116")
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


network = Net()
network = network.cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
print("开始训练")
start = time.time()
for i in range(0, 1):
    print(i)
    train(1)
end = time.time()
print(f"训练结束，用时{end-start}秒")
test()
print("测试结束")

'''
# 以 CIFAR10 数据集为例，展示一下完整的模型训练套路，完成对数据集的分类问题

import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练的设备
device = torch.device("cuda:1")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 获得数据集的长度 len(), 即length
train_data_size = len(train_data)
test_data_size = len(test_data)

# 格式化字符串, format() 中的数据会替换 {}
print("训练数据集及的长度为: {}".format(train_data_size))
print("测试数据集及的长度为: {}".format(test_data_size))

# 利用DataLoader 来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )


    def forward(self, input):
        input = self.model(input)
        return input

model = Model()
model = model.to(device)                    # 在 GPU 上进行训练

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)                # 在 GPU 上进行训练

# 优化器
learning_rate = 1e-2        # 1e-2 = 1 * (10)^(-2) = 1 / 100 = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 设置训练网络的一些参数
total_train_step = 0                        # 记录训练的次数
total_test_step = 0                         # 记录测试的次数
epoch = 10                                  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()                    # 开始训练的时间
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)               # 将训练的数据放入
        loss = loss_fn(outputs, targets)    # 得到损失值

        optimizer.zero_grad()               # 优化过程中首先要使用优化器进行梯度清零
        loss.backward()                     # 调用得到的损失，利用反向传播，得到每一个参数节点的梯度
        optimizer.step()                    # 对参数进行优化
        total_train_step += 1               # 上面就是进行了一次训练，训练次数 +1

        # 只有训练步骤是100 倍数的时候才打印数据，可以减少一些没有用的数据，方便我们找到其他数据
        if total_train_step % 100 == 0:
            end_time = time.time()          # 训练结束时间
            print("训练时间: {}".format(end_time - start_time))
            print("训练次数: {}, Loss: {}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 如何知道模型有没有训练好，即有咩有达到自己想要的需求
    # 我们可以在每次训练完一轮后，进行一次测试，在测试数据集上跑一遍，以测试数据集上的损失或正确率评估我们的模型有没有训练好

    # 顾名思义，下面的代码没有梯度，即我们不会利用进行调优
    total_test_loss = 0
    total_accuracy = 0                                      # 准确率
    with torch.no_grad():
        for data in test_dataloader:                        # 测试数据集中取数据
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)                # 这里的 loss 只是一部分数据(data) 在网络模型上的损失
            total_test_loss = total_test_loss + loss        # 整个测试集的loss
            accuracy = (outputs.argmax(1) == targets).sum() # 分类正确个数
            total_accuracy += accuracy                      # 相加

    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_loss += 1                                    # 测试完了之后要 +1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()

'''
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss = test_loss.to(device)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(epoch):
    network.train()
    print("开始咯")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        print("111")
        optimizer.zero_grad()
        print("112")
        output = network(data)
        print("113")
        loss = F.nll_loss(output, target)
        loss = loss.to(device)
        print("114")
        loss.backward()
        print("115")
        optimizer.step()
        print("116")
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

device = torch.device("cuda:1")
network = Net()
network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
print("开始训练")
start = time.time()
for i in range(0, 1):
    print(i)
    train(1)
end = time.time()
print(f"训练结束，用时{end-start}秒")
test()
print("测试结束")'''

