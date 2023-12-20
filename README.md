# GenerativeModels 生成式模型
### ConditionalVAEs、GANs

### A8: Generative Models 生成式模型
- 在上个章节中，我们介绍了自动编码器及其变种。`自动编码机(AutoEncoders)`通过学习y=x的恒等变换实现在无监督训练自动提取特征。自动编码机的变种，包括`稀疏自动编码机（SparseAEs）`、`去噪自动编码机（DenoisingAEs）`，通过训练技巧提升了编码的稀疏性和鲁棒性。这一类常常被用于特征提取和降维任务。但是对于生成任务而言，他们并不合适，原因在于原始数据分布在编码器复杂非线性变换的投射下未必是规则的，这不利于我们在生成采样时令解码器得到有意义的结果。
- 变分自动编码机（VAE），借鉴了贝叶斯方法中的变分推断技术，在训练模型学习恒等变换的同时通过引入对编码空间的KL散度进行正则化，使得编码空间在编码过程中保持良好的规范性，有利于在编码空间的采样与插值。
- VAE也可以实现一种有监督的变种：条件变分自动编码机（ConditionalVAE，CVAE），可引入标签数据，实现有标签的生成任务。

VAE的原理：
- 首先，它在AutoEncoder的基础上，假定编码器的结果是一个高斯分布，用于近似后验条件分布P(z|x)，前向传播时对它的均值与方差进行预测。
- VAEs在训练时采用重参数化技巧，计算z=mu + std * eps, （eps从高斯分布上采样得到），使得梯度可以顺利回传至编码器。
- 使用重构误差（y与x的误差）和编码空间的KL散度损失同时对模型进行约束。

本实验工作内容如下：
- 以`CIFAR-10`和`MNIST`数据集为例，展示条件变分自动编码机（ConditionalVAE，CVAE）的训练和评估
- 提供了美丽的UI界面，用于展示潜在空间的连续性
- 复现了CVAE作为贝叶斯模型，在训练时遇到的“后验坍缩”问题

训练过程代码：`ConditionalVAE.ipynb`

运行GUI演示：
```
python cvae_visualizer.py --<数据集>

python cvae_visualizer.py --mnist
python cvae_visualizer.py --cifar10
python cvae_visualizer.py --fashion_mnist
python cvae_visualizer.py --svhn
```

效果如下：

![demo.gif](assets%2Fdemo.gif)

![demo2.gif](assets%2Fdemo2.gif)
