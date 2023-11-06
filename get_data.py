import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset


def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return image_data


def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        struct.unpack('>II', f.read(8))
        """
        这行代码的作用是从二进制文件中读取8个字节（4个字节对应两个I，每个I表示一个无符号整数），
        并将这8个字节按照大端字节序（>表示大端）解析为两个无符号整数。
        在MNIST标签文件中，文件头包含了两个32位整数，
        第一个是魔数（magic number），第二个是标签的数量。
        """
        label_data = np.fromfile(f, dtype=np.uint8)
    return label_data


def load_fashion_train_data():
    train_img = './Data/train-images-idx3-ubyte'
    train_lab = './Data/train-labels-idx1-ubyte'

    # 加载图像数据
    train_images = load_mnist_images(train_img)
    # 加载标签数据
    train_labels = load_mnist_labels(train_lab)

    # 数据预处理和转换为PyTorch张量
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    train_images = torch.unsqueeze(train_images, 1)  # 在第二个维度上添加一个维度，将通道数设置为1
    train_labels = train_labels.long()
    train_dataset = TensorDataset(train_images, train_labels)

    return train_dataset


def load_fashion_test_data():
    test_img = './Data/t10k-images-idx3-ubyte'
    test_lab = './Data/t10k-labels-idx1-ubyte'

    # 加载测试图像数据
    test_images = load_mnist_images(test_img)
    # 加载测试标签数据
    test_labels = load_mnist_labels(test_lab)

    # 数据预处理和转换为PyTorch张量
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    test_images = torch.unsqueeze(test_images, 1)  # 在第二个维度上添加一个维度，将通道数设置为1
    test_labels = test_labels.long()

    test_dataset = TensorDataset(test_images, test_labels)

    return test_dataset


if __name__ == '__main__':
    train_loader = load_fashion_train_data()
    # 定义类别标签
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    i = 0
    input_1 = []
    label_1 = []
    for inputs, labels in train_loader:
        input_1.append(inputs)
        label_1.append(labels)
        if i == 20:
            plt.figure(figsize=(10, 8))
            for e in range(15):
                # 可视化数据集中的一些图像
                plt.subplot(3, 5, e+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.title(class_names[int(label_1[e])])
                plt.imshow(input_1[e], cmap=plt.cm.binary)
            plt.show()
            i=100
        i += 1
        if i == 100:
            break


