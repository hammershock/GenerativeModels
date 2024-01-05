# GenerativeModels 生成式模型
### 以ConditionalVAEs、GANs为例展示生成式模型的训练和推理

- 在上个章节中，我们介绍了自动编码器及其变种。`自动编码机(AutoEncoders)`通过学习y=x的恒等变换实现在无监督训练自动提取特征。自动编码机的变种，包括`稀疏自动编码机（SparseAEs）`、`去噪自动编码机（DenoisingAEs）`，通过训练技巧提升了编码的稀疏性和鲁棒性。这一类常常被用于特征提取和降维任务。但是对于生成任务而言，他们并不合适，原因在于原始数据分布在编码器复杂非线性变换的投射下未必是规则的，这不利于我们在生成采样时令解码器得到有意义的结果。
- 变分自动编码机（VAE），借鉴了贝叶斯方法中的变分推断技术，在训练模型学习恒等变换的同时通过引入对编码空间的KL散度进行正则化，使得编码空间在编码过程中保持良好的规范性，有利于在编码空间的采样与插值。
- VAE也可以实现一种有监督的变种：条件变分自动编码机（ConditionalVAE，CVAE），可引入标签数据，实现有标签的生成任务。

VAE的原理：
- 首先，它在AutoEncoder的基础上，假定编码器的结果是一个高斯分布，用于近似后验条件分布P(z|x)，前向传播时对它的均值与方差进行预测。
- VAEs在训练时采用重参数化技巧，计算z=mu + std * eps, （eps从高斯分布上采样得到），使得梯度可以顺利回传至编码器。
- 使用重构误差（y与x的误差）和编码空间的KL散度损失同时对模型进行约束。

本实验工作内容如下：
- 以`CIFAR-10`和`MNIST`、`fashion_mnist`、`svhn`和`celeba`数据集为例，展示条件变分自动编码机（ConditionalVAE，CVAE）用作图像生成模型的训练和评估
- 提供了美丽的UI界面，可以展示潜在空间的连续性
- 复现了CVAE作为贝叶斯模型，在训练时遇到的“后验坍缩”问题

celeba人脸数据集训练源代码：`celeba.ipynb`
其他数据集训练源代码：`ConditionalVAE.ipynb`

运行GUI演示：
1. 首先安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. CVAE图像生成：
    ```
    python app.py
    ```
3.
   app功能：
   - 可以切换数据集种类，使用训练好的不同模型文件，模型目录：`models/`
   - 根据不同的标签类别生成图像
   - 提供按钮，开始在latent space随机游走，并展示图像的变化

训练部分源代码请见: `ConditionalVAE.ipynb`, 针对celeba数据集使用的模型在标签嵌入上稍有不同，请见`celeba.ipynb`。

演示效果如下：

![demo_app1.gif](assets%2Fdemo_app1.gif)

![demo_app2.gif](assets%2Fdemo_app2.gif)

![demo_app3.gif](assets%2Fdemo_app3.gif)

![demo_app4.gif](assets%2Fdemo_app4.gif)

![demo_app5.gif](assets%2Fdemo_app5.gif)