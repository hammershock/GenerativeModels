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

    def __init__(self, potential_dim, channels, num_attributes=40):
        super(ConditionalConvVAE, self).__init__()
        self.potential_dim = potential_dim
        self.channels = channels

        # 对类别标签进行编码的线性层
        self.attr_embedding = nn.Linear(num_attributes, num_attributes)

        output_shape = (128, 6, 7)

        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        # image_size = (178, 218)
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + num_attributes, 64, kernel_size=3, stride=2, padding=1),  # 89
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 45
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 23
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 12
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 6
            nn.Flatten(),
        )

        self.enc_mu = nn.Linear(output_dim, potential_dim)  # 均值
        self.enc_log_var = nn.Linear(output_dim, potential_dim)  # 对数方差
        # 解码器
        self.decoder_fc = nn.Linear(potential_dim + num_attributes, output_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, output_shape),  # 6
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 12
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 23
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 45
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),  # 89
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((178, 218)),
            nn.Sigmoid()
        )

    def encode(self, x, attributes):
        # 将标签嵌入到与图像相同的维度
        # print(x.shape)  # torch.Size([128, 3, 178, 218])
        embedded_attrs = self.attr_embedding(attributes).unsqueeze(2).unsqueeze(3)
        embedded_attrs = embedded_attrs.expand(embedded_attrs.size(0), embedded_attrs.size(1), x.size(2), x.size(3))

        # 将标签和图像连接起来
        x = torch.cat((x, embedded_attrs), dim=1)

        # 传入编码器
        x = self.encoder(x)
        # print(x.shape)
        mu = self.enc_mu(x)
        log_var = self.enc_log_var(x)

        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_var

    def decode(self, z, labels):
        # 将标签嵌入并与潜在向量连接起来
        labels = self.attr_embedding(labels)
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
    celeba = 'celeba'


class ZParticle:
    def __init__(self, dim=64, alpha=0.1, beta=0.1, gamma=0.03, z_limit=3.0):
        """
        初始化粒子。

        :param dim: 潜在变量z的维度。
        :param alpha: 惯性系数。
        :param beta: 加速度影响系数。
        :param gamma: 随机扰动系数。
        """
        self.z_limit = z_limit
        self.dim = dim
        self.position = np.random.randn(dim)  # 初始位置
        self.velocity = np.zeros(dim)  # 初始速度
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update(self):
        """
        更新粒子的位置和速度。
        """
        # 计算加速度
        acceleration = -self.beta * self.position  # 平方正比

        # 更新速度：受前一时刻惯性、加速度和随机扰动的影响
        self.velocity = self.alpha * self.velocity + acceleration + self.gamma * np.random.randn(self.dim)

        # 更新位置
        self.position += self.velocity
        # print(np.linalg.norm(self.position), np.linalg.norm(self.velocity))
        return self.position

    def set_value(self, dim, new_value):
        self.position[dim] = new_value

    def set_wander_params(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_value(self, dim: int):
        return self.position[dim]

    def reset(self):
        self.position = np.random.randn(self.dim)
        self.velocity = np.zeros(self.dim)

    def tensor(self):
        return torch.FloatTensor(self.position).unsqueeze(0).clip(-self.z_limit, self.z_limit)


class Generator:
    def __init__(self, datasetType: DatasetType, z_limit=3.0, device='cpu'):
        self.device = device
        self.z_limit = z_limit

        model_path = f'models/cvae_{datasetType.name}.pth'
        CHANNELS = 1 if datasetType in [DatasetType.mnist, DatasetType.fashion_mnist] else 3
        self.model = ConditionalConvVAE(potential_dim=64, channels=CHANNELS, num_attributes=40)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)

        # self.z = torch.randn(1, self.model.potential_dim).to(device).clip(-self.z_limit, self.z_limit)
        self.z = ZParticle(64, alpha=0.3, beta=0.3, gamma=0.3, z_limit=z_limit)

    def generate(self, attrs):
        with torch.no_grad():
            attrs = torch.FloatTensor(np.array([attrs])).to(self.device)
            # 通过解码器生成图像
            image = self.model.decode(self.z.tensor().to(self.device), attrs).squeeze().cpu().numpy()
        return image

    def get_value(self, dimension_idx: int):
        return self.z.get_value(dimension_idx)

    def update_z(self, z_dim: int, new_value: float):
        if not 0 <= z_dim < self.model.potential_dim:
            raise ValueError(f'z_dim should between 0 and {self.model.potential_dim - 1}')
        self.z.set_value(z_dim, new_value)

    def wander_z(self):
        self.z.update()

    def reset_z(self):
        self.z.reset()  # = torch.randn(1, self.model.potential_dim).to(self.device).clip(-self.z_limit, self.z_limit)


class App:
    CIFAR_LABELS_CN = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
    MNIST_LABELS = [str(i) for i in range(10)]
    FASHION_MNIST_LABELS_CN = ["T恤/上衣", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
    SVHN_LABELS = [str(i) for i in range(10)]  # 与MNIST相同，0-9的数字
    CELEBA_LABELS_CN = [
        "五点钟胡须", "挑眉", "魅力", "眼袋", "秃头",
        "刘海", "厚嘴唇", "大鼻子", "黑色头发", "金色头发",
        "模糊的", "棕色头发", "浓眉", "圆润", "双下巴",
        "眼镜", "山羊胡", "灰白头发", "浓妆", "高颧骨",
        "男性", "微张嘴", "小胡子", "狭长眼睛", "无胡子",
        "椭圆脸型", "苍白肤色", "尖鼻子", "发际线后退", "红润脸颊",
        "鬓角", "微笑", "直发", "波浪发型", "耳环",
        "戴帽子", "口红", "项链", "领带", "年轻"
    ]

    def __init__(self, datesetType: DatasetType, image_shape=(445, 545), z_limit=3.0, wander_interval=20):
        self.root = tk.Tk()
        self.root.title("VAE Image Generator")

        # 配置变量
        self.z_limit = z_limit  # z边界
        self.image_shape = image_shape  # 显示图像大小
        self.wander_interval = wander_interval  # 随机游走的时间间隔（毫秒）

        # 生成器
        self.generator = Generator(datesetType, z_limit=z_limit)

        # 标签
        if datesetType == DatasetType.mnist:
            self.labels = self.MNIST_LABELS
        elif datesetType == DatasetType.cifar10:
            self.labels = self.CIFAR_LABELS_CN
        elif datesetType == DatasetType.fashion_mnist:
            self.labels = self.FASHION_MNIST_LABELS_CN
        elif datesetType == DatasetType.svhn:
            self.labels = self.SVHN_LABELS
        elif datesetType == DatasetType.celeba:
            self.labels = self.CELEBA_LABELS_CN
        else:
            raise NotImplementedError

        # 状态变量
        self.attribute_var = tk.StringVar()
        self.attribute_value_var = tk.StringVar()
        self.attrs = [-1] * 40
        self.dimension_var = tk.StringVar()
        self.is_wandering = False  # 初始时，粒子不进行随机游走

        # 创建一个Frame容器来放置控制组件
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, sticky="nsew")

        # 创建一个Frame容器来放置图像展示
        image_frame = tk.Frame(self.root)
        image_frame.grid(row=0, column=1, sticky="nsew")

        # 维度索引选框
        tk.Label(control_frame, text="维度索引（0-" + str(self.generator.model.potential_dim - 1) + "）").pack()
        self.dimension_var.set('0')  # 设置默认值
        self.dimension_dropdown = ttk.Combobox(control_frame, textvariable=self.dimension_var,
                                               values=[str(i) for i in range(self.generator.model.potential_dim)])
        self.dimension_dropdown.pack()
        self.dimension_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_bar() or self.update_image())

        dim_slider_frame = tk.Frame(control_frame)
        dim_slider_frame.pack()

        # 维度z值滑动条
        tk.Label(dim_slider_frame, text="维度z值").grid(row=0, column=0, sticky="e")
        self.slider = tk.Scale(dim_slider_frame, from_=-self.z_limit, to=self.z_limit, resolution=0.01,
                               orient=tk.HORIZONTAL, command=lambda val: self.update_image())
        self.slider.grid(row=0, column=1, sticky="e")

        # 重置z值按钮
        self.reset_z_button = tk.Button(control_frame, text="Reset z",
                                        command=lambda: self.generator.reset_z() or self.update_bar() or self.update_image())
        self.reset_z_button.pack()

        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.pack(fill='x', padx=2, pady=20)

        # 属性选择选框
        tk.Label(control_frame, text="属性选择").pack()
        self.attribute_var.set(self.labels[0])  # 设置默认值
        self.attr_dropdown = ttk.Combobox(control_frame, textvariable=self.attribute_var, values=self.labels)
        self.attr_dropdown.pack()
        self.attr_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_attr_var())

        # 属性取值选框
        tk.Label(control_frame, text="属性取值").pack()
        self.attribute_value_var.set('False')  # 设置默认值
        self.attr_val_dropdown = ttk.Combobox(control_frame, textvariable=self.attribute_value_var, values=['True', 'False'])
        self.attr_val_dropdown.pack()
        self.attr_val_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_image())

        # 生成的图像
        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        # 重置属性值按钮
        self.reset_attr_value_button = tk.Button(control_frame, text="Reset attrs", command=lambda: self.reset_attrs() or self.update_attr_var() or self.update_image())
        self.reset_attr_value_button.pack()

        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.pack(fill='x', padx=2, pady=20)

        # 随机游走参数滑动条
        slider_frame = tk.Frame(control_frame)
        slider_frame.pack()
        # Alpha滑动条及其标签
        tk.Label(slider_frame, text="惯性系数 Alpha").grid(row=1, column=0, sticky="e")
        self.alpha_slider = tk.Scale(slider_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                     command=self.update_wander_params)
        self.alpha_slider.grid(row=1, column=1)

        # Beta滑动条及其标签
        tk.Label(slider_frame, text="引力系数 Beta").grid(row=2, column=0, sticky="e")
        self.beta_slider = tk.Scale(slider_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                    command=self.update_wander_params)
        self.beta_slider.grid(row=2, column=1)

        # Gamma滑动条及其标签
        tk.Label(slider_frame, text="随机系数 Gamma").grid(row=3, column=0, sticky="e")
        self.gamma_slider = tk.Scale(slider_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                     command=self.update_wander_params)
        self.gamma_slider.grid(row=3, column=1)

        # 随机游走按钮
        self.wander_button = tk.Button(control_frame, text="Start Wandering", command=self.toggle_wandering)
        self.wander_button.pack()

        self.initialize()

    def initialize(self):
        self.alpha_slider.set(0.13)
        self.beta_slider.set(0.06)
        self.gamma_slider.set(0.24)

        self.generator.wander_z()
        self.update_bar()
        self.update_image()

    def toggle_wandering(self):
        """切换粒子的随机游走状态"""
        self.is_wandering = not self.is_wandering
        self.wander_button.config(text="Stop Wandering" if self.is_wandering else "Start Wandering")
        self.start_wandering()

    def start_wandering(self):
        """开始粒子的随机游走"""
        if self.is_wandering:
            self.generator.wander_z()
            self.update_image()
            self.root.after(self.wander_interval, self.start_wandering)

    def update_wander_params(self, _=None):
        """更新随机游走的参数"""
        alpha = self.alpha_slider.get()
        beta = self.beta_slider.get()
        gamma = self.gamma_slider.get()
        self.generator.z.set_wander_params(alpha, beta, gamma)

    def update_bar(self):
        self.slider.set(self.generator.get_value(int(self.dimension_var.get())))

    def update_attr_var(self):
        var = 'False' if self.attrs[self.labels.index(self.attribute_var.get())] == -1 else 'True'
        self.attribute_value_var.set(var)

    def reset_attrs(self):
        self.attrs = np.random.choice([-1, 1], 40)

    def update_image(self):
        try:
            dim_idx = int(self.dimension_var.get())
            new_value = float(self.slider.get())
        except ValueError:
            print("Please enter a valid integer for the dimension index.")
            return
        self.generator.update_z(dim_idx, new_value)
        # 生成图像

        # 修改属性标签
        self.attrs[self.labels.index(self.attribute_var.get())] = 1 if self.attribute_value_var.get() == 'True' else -1

        image = self.generator.generate(self.attrs)
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


if __name__ == "__main__":
    app = App(DatasetType.celeba)
    app.run()
