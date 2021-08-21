import torch
import os
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import LeNet
from PIL import Image
import time
import numpy as np

#项目根目录
data_dir = '../LeNet-5/'
label_file = 'df_part_7.csv'
test_file = 'test.csv'
num_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 封装函数，就将以上内容作为参数从主文件里调用


# def generate_txt(data_dir, label_file, test_file):
#     # 生成 train.txt
#     with open(os.path.join(data_dir, label_file), 'r') as f:  # 拼接data_dir和label_file路径
#         lines = f.readlines()[0:]
#         print('****************')
#         print('input :', os.path.join(data_dir, label_file))
#         print('start...')
#         listText = open('../LeNet-5/df_part_7.txt', 'a+')  # 创建并打开train.txt文件
#         for l in lines:
#             tokens = l.rstrip().split(',')  # csv存在‘，'使用split
#             idx, label = tokens
#             name = data_dir + 'data/' + idx + ' ' + str(int(label)) + '\n'
#             listText.write(name)
#         listText.close()
#         print('down!')
#         print('****************')


#     with open(os.path.join(data_dir, test_file), 'r') as f1:
#         lines1 = f1.readlines()[0:]  # 表示从第1行，下标为0的数据行开始
#         print('****************')
#         print('input :', os.path.join(data_dir, test_file))
#         print('start...')
#         listText1 = open('../LeNet-5/test.txt', 'a+')  # 创建并打开test.txt文件
#         for l1 in lines1:
#             name1 = data_dir + 'data/' + l1.rstrip() + '\n'  # rstrip()为了把右边的换行符删掉
#             listText1.write(name1)
#         listText1.close()
#         print('down!')
#
#
# generate_txt(data_dir, label_file, test_file)
class MyDataset(Dataset):
    def __init__(self, txt, type, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        self.type = type
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()  # 分割成文件名和标签
            if self.type == "train":
                imgs.append((words[0], int(words[1])))
            else:
                imgs.append(words[0])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.type == "train":
            fn, label = self.imgs[index]
        else:
            fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.type == "train":
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def train():

    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

    train_data = MyDataset(txt='../LeNet-5/df_part_7.txt', type="train", transform=transform)

    # 训练集
    # 分为验证集（0.2）和训练集（0.8）
    batch_size = 70
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))  # np.floor向下取整
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)  # 打乱顺序
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                              sampler=test_sampler,
                                              num_workers=16)

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

# for i, data in enumerate(train_loader):
#     inputs, labels = data
#     print(inputs.shape,labels)

    net = LeNet()
    loss_funcation = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08)
    optimizer = optim.Adadelta(net.parameters(), rho=0.9)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# device = torch.device("cpu")
# print(device)

    device = torch.device("cuda")
    net.cuda()
    test_acc = []*10
    model_loss = []*10

    print("Start training...")

    for epoch in range(10):
        running_loss = 0.0
        time_start = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.to(device), labels.to(device)  #将inputs和labels分配到指定的device中
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_funcation(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #打印数据信息
            if step % 129 == 128:
                with torch.no_grad():
                    outputs = net(test_image.to(device))  # 将test_image分配到指定的device中
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                    # 将test_label分配到指定的device中

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 100, accuracy))
                    test_acc.append(accuracy)
                    model_loss.append(running_loss/100)
                    print('%f s' % (time.perf_counter() - time_start))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './LeNet.pth'
    torch.save(net.state_dict(), save_path)

    plt.plot(num_epoch, test_acc)
    plt.suptitle('Model-Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(num_epoch, model_loss)
    plt.suptitle('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Model-Loss')
    plt.show()


if __name__ == "__main__":
    train()





