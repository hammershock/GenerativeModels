import argparse
import tkinter as tk
from enum import Enum
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk


class ConditionalConvVAE(nn.Module):
    """
    条件全卷积VAE变分自动编码器
    """
    NAME = 'ConditionalConvVAE'

    def __init__(self, potential_dim, channels, num_classes=10):
        super(ConditionalConvVAE, self).__init__()
        self.potential_dim = potential_dim
        self.channels = channels

        # 对类别标签进行编码的线性层
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        output_shape = (1024, 4, 4)

        output_dim = output_shape[0] * output_shape[1] * output_shape[2]

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + num_classes, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1),  # output: 1024 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),
        )

        self.enc_mu = nn.Linear(output_dim, potential_dim)  # 均值
        self.enc_log_var = nn.Linear(output_dim, potential_dim)  # 对数方差
        # 解码器
        self.decoder_fc = nn.Linear(potential_dim + num_classes, output_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, output_shape),
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # output: channel x 28 x 28
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        # 将标签嵌入到与图像相同的维度
        labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), x.size(2), x.size(3))

        # 将标签和图像连接起来
        x = torch.cat((x, labels), dim=1)

        # 传入编码器
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_log_var(x)

        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_var

    def decode(self, z, labels):
        # 将标签嵌入并与潜在向量连接起来
        labels = self.label_embedding(labels)
        z = torch.cat((z, labels), dim=1)

        # 传入解码器
        x = self.decoder_fc(z)
        x = self.decoder(x)
        return x

    def forward(self, x, labels):
        z, mu, log_var = self.encode(x, labels)
        reconstructed_x = self.decode(z, labels)
        return reconstructed_x, mu, log_var

class DatasetType(Enum):
    cifar10 = 'cifar10'
    mnist = 'mnist'
    fashion_mnist = 'fashion_mnist'
    svhn = 'svhn'


class Generator:
    def __init__(self, datasetType: DatasetType, z_limit=3.0, device='cpu'):
        self.device = device
        self.z_limit = z_limit

        model_path = f'models/cvae_{datasetType.name}.pth'
        CHANNELS = 1 if datasetType in [DatasetType.mnist, DatasetType.fashion_mnist] else 3
        NUM_CLASSES = 10
        self.model = ConditionalConvVAE(potential_dim=8, channels=CHANNELS, num_classes=NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)

        self.z = torch.randn(1, self.model.potential_dim).to(device).clip(-self.z_limit, self.z_limit)

    def generate(self, label_idx):
        with torch.no_grad():
            labels = torch.LongTensor(np.array([label_idx])).to(self.device)
            # 通过解码器生成图像
            image = self.model.decode(self.z, labels).squeeze().cpu().numpy()
        return image

    def get_value(self, dimension_idx: int):
        return float(self.z[:, dimension_idx])

    def update_z(self, z_dim: int, new_value: float):
        if not 0 <= z_dim < self.model.potential_dim:
            raise ValueError(f'z_dim should between 0 and {self.model.potential_dim - 1}')
        self.z[:, z_dim] = new_value

    def reset_z(self):
        self.z = torch.randn(1, self.model.potential_dim).to(self.device).clip(-self.z_limit, self.z_limit)


class App:
    CIFAR_LABELS_CN = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
    MNIST_LABELS = [str(i) for i in range(10)]
    FASHION_MNIST_LABELS_CN = ["T恤/上衣", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
    SVHN_LABELS = [str(i) for i in range(10)]  # 与MNIST相同，0-9的数字

    def __init__(self, datesetType: DatasetType, image_shape=(600, 600), z_limit=3.0):
        self.root = tk.Tk()
        self.root.title("VAE Image Generator")
        self.z_limit = z_limit
        self.image_shape = image_shape

        self.generator = Generator(datesetType, z_limit=z_limit)

        if datesetType == DatasetType.mnist:
            self.labels = self.MNIST_LABELS
        elif datesetType == DatasetType.cifar10:
            self.labels = self.CIFAR_LABELS_CN
        elif datesetType == DatasetType.fashion_mnist:
            self.labels = self.FASHION_MNIST_LABELS_CN
        elif datesetType == DatasetType.svhn:
            self.labels = self.SVHN_LABELS
        else:
            raise NotImplementedError

        self.label_var = tk.StringVar()
        self.dimension_var = tk.StringVar()

        # 创建滑动条
        self.slider = tk.Scale(self.root, from_=-self.z_limit, to=self.z_limit, resolution=0.01, orient=tk.HORIZONTAL,
                          command=lambda val: self.update_image())
        self.slider.pack()

        # 创建维度索引输入框
        tk.Label(self.root, text="维度索引（0-" + str(self.generator.model.potential_dim - 1) + "）").pack()
        self.dimension_var.set('0')  # 设置默认值
        self.dimension_dropdown = ttk.Combobox(self.root, textvariable=self.dimension_var,
                                               values=[str(i) for i in range(self.generator.model.potential_dim)])
        self.dimension_dropdown.pack()
        self.dimension_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_bar() or self.update_image())

        # 创建类别选择下拉菜单
        tk.Label(self.root, text="类别选择").pack()
        self.label_var.set(self.labels[0])  # 设置默认值
        self.label_dropdown = ttk.Combobox(self.root, textvariable=self.label_var, values=self.labels)
        self.label_dropdown.pack()
        self.label_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_image())

        # 创建图像标签
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # 创建重置按钮
        self.reset_button = tk.Button(self.root, text="Reset z", command=lambda: self.generator.reset_z() or self.update_bar() or self.update_image())
        self.reset_button.pack()

    def update_bar(self):
        self.slider.set(self.generator.get_value(int(self.dimension_var.get())))

    def update_image(self):
        try:
            dim_idx = int(self.dimension_var.get())
            new_value = float(self.slider.get())
        except ValueError:
            print("Please enter a valid integer for the dimension index.")
            return
        self.generator.update_z(dim_idx, new_value)
        # 生成图像
        image = self.generator.generate(self.labels.index(self.label_var.get()))
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))  # 转换为HWC格式
        image_to_show = (image * 255).astype(np.uint8)  # 转换为8位图像
        image_to_show = Image.fromarray(image_to_show).resize(self.image_shape)
        photo = ImageTk.PhotoImage(image_to_show)

        # 更新图像
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 保存引用，防止被垃圾回收

    def run(self):
        self.update_image()
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Choose a dataset.")
    parser.add_argument("--mnist", action="store_true", help="Use the MNIST dataset.")
    parser.add_argument("--cifar10", action="store_true", help="Use the CIFAR-10 dataset.")
    parser.add_argument("--fashion_mnist", action="store_true", help="Use the fashion_mnist dataset.")
    parser.add_argument("--svhn", action="store_true", help="Use the svhn dataset.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mnist:
        app = App(DatasetType.mnist)
    elif args.cifar10:
        app = App(DatasetType.cifar10)
    elif args.fashion_mnist:
        app = App(DatasetType.fashion_mnist)
    elif args.svhn:
        app = App(DatasetType.svhn)
    else:
        app = App(DatasetType.mnist)
        # raise ValueError("Please specify a dataset using --mnist or --cifar10.")
    app.run()
