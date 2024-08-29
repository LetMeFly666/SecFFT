<!--
 * @Author: LetMeFly
 * @Date: 2024-08-11 10:29:13
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-08-29 11:25:08
-->
# SecFFT: Safeguarding Federated Fine-tuning for Large Vision Language Models against Stealthy Poisoning Attacks in IoRT Networks

## 前言

LLM的FL安全性相关实验。

## 介绍

### 分支

+ [master](https://github.com/LetMeFly666/SecFFT/tree/master): 仓库主分支，最终版本的代码将会发布在这里
+ [paper](https://github.com/LetMeFly666/SecFFT/tree/paper): 论文分支，论文的latex源码，论文中所需的一些图片最终也会添加到这里
+ [z.001.tip-adapter](https://github.com/LetMeFly666/SecFFT/tree/z.001.tip-adapter): 先[使用Tip-Adapter](https://github.com/LetMeFly666/SecFFT/blob/d2b385e040117cdc776e856a2f899c711cce9b78/README.md?plain=1#L329-L331)，并融入了联邦学习框架
+ [wb.001.lora](https://github.com/LetMeFly666/SecFFT/tree/wb.001.lora): [wb](https://github.com/Pesuking)使用lora进行的尝试，对应仓库[Pesuking@SecFFT](https://github.com/Pesuking/SecFFT)

## Log

### Log001 - 2024.8.11_10:42-18:45

首先使用[CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter)的方式进行研究，先将CLIP-Adapter跑起来。

CLIP-Adapter需要先跑起来[CoOp](https://github.com/KaiyangZhou/Dassl.pytorch)，

然后就：[显示了一堆](https://github.com/LetMeFly666/SecFFT/blob/d2b385e040117cdc776e856a2f899c711cce9b78/README.md?plain=1#L25-L324)。

然后准备跑`New version of CLIP-Adapter`：[Tip-Adapter: Training-free CLIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)，

当前将其添加到了分支[z.001.tip-adapter](https://github.com/LetMeFly666/SecFFT/tree/z.001.tip-adapter)下。

数据集：默认的`ImageNet`有一百多G，决定使用同样受支持的`Flowers102`（oxford_flowers）

### Log002 - 2024.8.12_13:28-18:56

读一下代码结构，融入联邦学习中。

明天大概要开会，先融入了再说。。

结果：先读了一下源码，明确了一下数据类型（[6f08b1..](https://github.com/LetMeFly666/SecFFT/commit/6f08b1cc63cffba4cb91aec910a0c04adf5d965d)）

### Log002 - 2024.8.12_18:56-12:51

加上联邦学习框架。

<details><summary>运行结果</summary>

```
round 1's acc: 94.11
round 2's acc: 94.92
round 3's acc: 94.88
round 4's acc: 95.90
round 5's acc: 95.25
round 6's acc: 95.45
round 7's acc: 95.66
round 8's acc: 95.66
round 9's acc: 95.49
round 10's acc: 95.25
```

</details>

融入成功：[de604d..](https://github.com/LetMeFly666/SecFFT/commit/de604d63f39300ff0131f8cf1f546a2c0c3472ce)

### Log003 - 2024.8.13_17:02-2024.8.15_21:32

刚开完60min会，准备之后2天开一次会。

把related work写完

场景 - IoT，机器人 具身智能，聚焦大模型微调

针对这些的攻击与防御（最新的攻击、防御:跟Fine-tuning,VLM相关的）

攻击找10来篇，防御找20来篇。

在overleaf里建个表，找到参考文献就引用上，例如(GlobalCome2023 引用, 简介)

最终把搜索关键词由`("federated learning" OR "distributed learning") AND "vision models" AND "fine-tuning attacks" AND (IoT OR robotics OR "embodied intelligence")`简化为了`"vision models" AND "fine-tuning attacks"`，还一共只搜索出来了5篇。

期间还产生了忘记限制“视觉大模型”的[论文检索结果](https://github.com/LetMeFly666/SecFFT/blob/d2b385e040117cdc776e856a2f899c711cce9b78/README.md?plain=1#L383-L429)。

### Log004 - 2024.8.15_21:36-2024.8.17_23:16

刚开完100min会，下次预计开会时间是周六。

进度安排：今晚把攻击的综述写好。

+ 攻击综述分类：时间隐蔽、空间隐蔽。
+ 防御分类：3-4类。

好家伙，一看时间还真是23:16，这是上一个commit(2024.8.16 0:32)写的时间。

所以使用ChatGPT的帮助肯定必不可少了：[如何调教ChatGPT](https://github.com/LetMeFly666/SecFFT/blob/a8cb910f491dc753d52773062f9e9b2d14e29b33/chat.md?plain=1#L732-L799)。

<details><summary><a href="chat.ChatWithGPT.93d00df4-6564-43a0-82d7-d8b3b4ab6f16.json">ChatLog</a>(json)</summary>

TODO: commit id (hash)

</details>

### Log005 - 2024.8.19_20:49-2024.8.23_22:00

+ 思路：根据问题去找文章。
+ 攻击：跑通两**三**种攻击。

### Log006 - 2024.8.23_23:20-2024.8.25_20:30

小80分钟的会刚结束。

+ Me想的和这次无关的一个之后可以尝试投一个会的小idea：针对恶意攻击检测模型的绕过性攻击。

+ 这两天加班加点找找时域频域的防御：创新是最后一步，不用完全凭空开始想。找个新的，在上面针对某个问题（比如这个最新方法没有考虑我今晚说的不连续）小提升。
+ 写出来：攻击模型、识别模型、具体方法
+ wb侧重一个攻击如何识别，我侧重多次攻击上如何识别（参考轨迹的那篇文章）

明天先写一版，后天晚上线下聊聊。

### Log007 - 2024.8.25_22:20-2024.8.27_18:32

**任务**

轨迹那篇[文章](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html)看仔细。

**结论**

+ 这是一篇介绍攻击的文章。正常的后门植入攻击是直接篡改原始数据（例如往图像上打标记）。本文不修改原始数据，只修改数据对应的标签。

   场景：两个场景
   
   + 一个是众包注释
   + 一个是知识蒸馏。
      
      都是无法修改原始数据但只能操纵数据标签的场景。

   方法：假设攻击者想让带黑框的卡车被预测成鹿。
   
   + 攻击者先自己训练一个模型，篡改原始数据，打上trigger，并将标签标记为鹿。这个模型被称为“专家模型”。记录训练过程中参数的“轨迹”。

      所谓“轨迹”，是每次迭代的模型参数和小批量样本。

      也就是`[(第1个batch的模型参数, 第1个batch的样本), (第2个batch的模型参数, 第2个batch的样本), ...]`

   + 攻击者再另训一个模型（记为目标模型），这个模型不修改原始图像，只篡改图像标签。通过篡改标签使得目标模型的轨迹和专家模型的轨迹尽可能地相似。

      怎么知道标签如何修改呢？作者定义了一个损失函数$L_{\text{param}}(\theta_k, \theta_{k+1}, \phi_{k+1}) = \frac{\|\theta_{k+1} - \phi_{k+1}\|_2}{\|\theta_{k+1} - \theta_k\|_2}$，分子是两个模型之间的欧几里得距离（二范数），分母衡量的是专家模型在这一迭代过程中参数的更新幅度。

      反向传播，根据梯度的方向调整标签。

   结果：
   
   + 相比于baseline的后门植入攻击，所需篡改数据量更少。
   + 在攻击成功率很高的情况下，模型在干净数据集上预测的准确率，相比baseline降低地更少。

+ 依据本文思想能想到的防御思路：

   思路：

   + 记录每个客户端的每轮次的轨迹（参数、轮次），通过计算每个客户端之间的轨迹相似度，从而判断是否有可能的攻击者。
   + 或者中央服务器自己训练一个没有恶意攻击的正常模型，对比客户端轨迹和自己训练得到的正常模型的轨迹。

   缺点：

   + 相当于是又回到了相似度检测，例如余弦相似度。相比之前的工作只是多增加了一个“历史记录”的考虑。对于隐式后门攻击，不知道能否检测出来。
   + 保存每个客户端的每次参数，需要较大的内存或硬盘空间。

+ 相关防御手段调研：

   + 本文目前(2024.8.27)有17次被引用次数，其中暂未发现针对这种攻击提出的防御手段。

### Log008 - 2024.8.27_21:45-2024.8.28_晚

**任务**

+ 大概占3分，找几个比较隐蔽的攻击方式：不连续、...、...  得到结论 单次不准
+ 大概占7分，看文章Geomed（求几何质心），从而识别恶意客户端的曲折攻击

**过程**

*Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning* CCFA，研究指出，在联邦学习中，由于后续的正常更新，后门攻击的效果会逐渐减弱，这表现为攻击成功率在多轮迭代中显著下降，最终可能完全失效。为了量化这种现象，文章引入了一个新的指标——攻击持久性（Attack Persistence），用于衡量后门攻击效果的衰减程度。在前人研究未能广泛探讨如何提高攻击持久性的背景下，作者提出了FCBA方法。该方法通过聚合更多的触发信息，生成更完整的后门模式，从而在全局模型中更好地植入后门。经过训练的后门模型对后续的正常更新具有更强的抗性，使得测试集上的攻击成功率更高。 作者在三个数据集上对这一方法进行了测试，并在不同的设置下评估了两种模型的表现。结果显示，FCBA的持久性优于现有最先进的联邦学习后门攻击方法。在GTSRB数据集上，经过120轮攻击后，FCBA的攻击成功率较基线提升了50%以上。开源。

*Input-Aware Dynamic Backdoor Attack* 382次引用，传统的后门攻击方法通常使用统一的触发器模式，即所有恶意样本使用相同的触发器。这种固定模式虽然有效，但容易被当前的防御方法检测到并减轻其影响。许多防御技术通过寻找这些固定触发器来识别和缓解后门攻击。为了提高攻击的隐蔽性，文章提出了一种输入感知动态后门攻击（Input-Aware Dynamic Backdoor Attack）。在这种方法中，触发器是根据每个输入动态生成的，而不是固定不变的。这意味着不同的输入图像会有不同的触发器，从而打破了现有防御方法的基础假设，使攻击更难被检测到。 1）触发器生成器：文章设计了一个基于自编码器（autoencoder）的触发器生成器，它根据输入图像生成相应的触发器。生成的触发器具有显著的多样性，确保不同输入图像的触发器彼此不同。2）交叉触发测试：为了进一步提高触发器的隐蔽性，研究者引入了一种新的测试方法，称为交叉触发测试（cross-trigger test）。通过该测试，确保为一个输入生成的触发器无法在其他输入上重用，从而进一步提高了攻击的隐蔽性。3）训练目标函数：文章结合了分类损失（classification loss）和多样性损失（diversity loss），以确保生成的触发器既能有效激活后门，又能在不同输入间保持足够的差异性。  实验结果表明，这种动态后门攻击在MNIST、CIFAR-10和GTSRB等标准数据集上取得了接近100%的攻击成功率，并且在使用现有最先进的防御方法时仍然能够成功绕过检测。文章还证明了该方法在图像正则化和网络检查工具（如GradCam）下的稳健性，这进一步验证了这种攻击的隐蔽性。

*Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning*  之前的backdoor方法通常从训练数据集中随机选择触发候选样本，这种做法容易扰乱样本分布，并模糊它们之间的边界，导致主要任务的准确性下降。此外，这些方法使用的触发器通常是手工制作且未经过优化，导致后门映射关系较弱，攻击成功率较低。  为了解决这些问题，本文提出了一种灵活的后门攻击方法，称为触发样本选择与优化（Trigger Sample Selection and Optimization, TSSO）。这一方法受到神经网络分类模式的启发，利用自编码器（Autoencoders）和局部敏感哈希（Locality-Sensitive Hashing）来选择在类边界处的触发候选样本，从而实现精确注入。此外，TSSO通过全局模型和历史结果迭代优化触发器的表示，从而建立一个稳健的映射关系。 文章在四个经典数据集上评估了TSSO方法，特别是在非独立同分布（non-IID）的设置下，TSSO在更少的训练轮次中取得了更高的攻击成功率，并延长了后门攻击的效果。即使在扩展性测试中，部署防御措施的情况下，TSSO仍然能够在只有4%恶意客户端（中毒率为1/640）的情况下达到超过80%的攻击成功率。这展示了TSSO在后门攻击中的高效性和持久性。

*FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information* 提出了一种名为FedRecover的方法，用于在联邦学习（Federated Learning, FL）系统中从中毒攻击中恢复全局模型。FedRecover方法的关键在于利用历史信息来估计客户端的模型更新，而不是在恢复过程中要求客户端重新计算和通信这些更新。这一方法旨在减少恢复过程中的计算和通信开销，同时保持恢复后的全局模型的准确性。

## End

日志里面有很多*有的没的*的记录。边增边删，Perfect!。