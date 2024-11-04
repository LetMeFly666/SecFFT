<!--
 * @Author: LetMeFly
 * @Date: 2024-08-11 10:29:13
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-19 15:54:13
-->
# SecFFT: Safeguarding Federated Fine-tuning for Large Vision Language Models against Stealthy Poisoning Attacks in IoRT Networks

## 前言

LLM的FL安全性相关实验。Changed from [RoseAgg](https://github.com/SleepedCat/RoseAgg).

## 介绍

### 分支

+ [master](https://github.com/LetMeFly666/SecFFT/tree/master): 仓库主分支，最终版本的代码将会发布在这里
+ [paper](https://github.com/LetMeFly666/SecFFT/tree/paper): 论文分支，论文的latex源码，论文中所需的一些图片最终也会添加到这里
+ [z.001.tip-adapter](https://github.com/LetMeFly666/SecFFT/tree/z.001.tip-adapter): 先[使用Tip-Adapter](https://github.com/LetMeFly666/SecFFT/blob/d2b385e040117cdc776e856a2f899c711cce9b78/README.md?plain=1#L329-L331)，并融入了联邦学习框架
+ [wb.001.lora](https://github.com/LetMeFly666/SecFFT/tree/wb.001.lora): [wb](https://github.com/Pesuking)使用lora进行的尝试，对应仓库[Pesuking@SecFFT](https://github.com/Pesuking/SecFFT)。（`git push Let main:wb.001.lora`）
+ [wb.002.clip_lora](https://github.com/LetMeFly666/SecFFT/tree/wb.002.clip_lora): wb使用lora进行的尝试，对应仓库[Pesuking@SecFFT](https://github.com/Pesuking/SecFFT)的分支[f](https://github.com/Pesuking/SecFFT/tree/f)。（`git push Let f:wb.002.clip_lora`）。这个分支后续将会与master分支几乎保持同步。

### 文件/目录

+ `NormalRun`：代码主目录。运行方式可参考`NormalRun/run.bat`。数据等的下载/设置可参考[`RoseAgg`](https://github.com/SleepedCat/RoseAgg)。
+ `getResult`：最终获取实验结果的代码（实验三、实验四）。
+ `.gitignore`：略。
+ `README.md`：本说明文件。

### 中间结果/最终实验

NormalRun的resultWithTime是一个转置的矩阵，第一行`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29`代表表头30轮，每一列是参与者（其实就是0-49），0-19是恶意。

<details><summary>数据来源：</summary>

+ [x] foolsgold: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/foolsgold_NEUROTOXIN_09132315-fmnist`
- [x] fltrust: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist`
- [x] flame: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-30-57-flame-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/flame_NEUROTOXIN_09122330-fmnist`

`../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist/fltrust_15.pth`

</details>

<details><summary>PeftModel</summary>

```
PeftModel(
  (base_model): LoraModel(
    (model): CLIPModel(
      (text_model): CLIPTextTransformer(
        (embeddings): CLIPTextEmbeddings(
          (token_embedding): Embedding(49408, 512)
          (position_embedding): Embedding(77, 512)
        )
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-11): 12 x CLIPEncoderLayer(
              (self_attn): CLIPSdpaAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
              )
              (layer_norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): Linear(in_features=512, out_features=2048, bias=True)
                (fc2): Linear(in_features=2048, out_features=512, bias=True)
              )
              (layer_norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (vision_model): CLIPVisionTransformer(
        (embeddings): CLIPVisionEmbeddings(
          (patch_embedding): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
          (position_embedding): Embedding(50, 768)
        )
        (pre_layrnorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-11): 12 x CLIPEncoderLayer(
              (self_attn): CLIPSdpaAttention(
                (k_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (v_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (q_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (out_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=3072, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=3072, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (fc2): lora.Linear(
                  (base_layer): Linear(in_features=3072, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=3072, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (visual_projection): lora.Linear(
        (base_layer): Linear(in_features=768, out_features=512, bias=False)
        (lora_dropout): ModuleDict(
          (default): Dropout(p=0.1, inplace=False)
        )
        (lora_A): ModuleDict(
          (default): Linear(in_features=768, out_features=16, bias=False)
        )
        (lora_B): ModuleDict(
          (default): Linear(in_features=16, out_features=512, bias=False)
        )
        (lora_embedding_A): ParameterDict()
        (lora_embedding_B): ParameterDict()
        (lora_magnitude_vector): ModuleDict()
      )
      (text_projection): Linear(in_features=512, out_features=512, bias=False)
    )
  )
)
```

</details>

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

所以使用ChatGPT的帮助肯定必不可少了：[如何调教ChatGPT](https://github.com/LetMeFly666/SecFFT/blob/a8cb910f491dc753d52773062f9e9b2d14e29b33/chat.md?plain=1#L732-L799)、[ChatLog(json)](https://github.com/LetMeFly666/SecFFT/blob/98cfc8176cb622d81d28e9adfd958c24b0e1cbdf/chat.ChatWithGPT.93d00df4-6564-43a0-82d7-d8b3b4ab6f16.json)。

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

### Log008 - 2024.8.27_21:45-2024.8.29_17:32

**任务**

+ 大概占3分，找几个比较隐蔽的攻击方式：不连续、...、...  得到结论 单次不准
+ 大概占7分，看文章Geomed（求几何质心），从而识别恶意客户端的曲折攻击

**过程**

*Input-Aware Dynamic Backdoor Attack* 382次引用，传统的后门攻击方法通常使用统一的触发器模式，即所有恶意样本使用相同的触发器。这种固定模式虽然有效，但容易被当前的防御方法检测到并减轻其影响。许多防御技术通过寻找这些固定触发器来识别和缓解后门攻击。为了提高攻击的隐蔽性，文章提出了一种输入感知动态后门攻击（Input-Aware Dynamic Backdoor Attack）。在这种方法中，触发器是根据每个输入动态生成的，而不是固定不变的。这意味着不同的输入图像会有不同的触发器，从而打破了现有防御方法的基础假设，使攻击更难被检测到。 1）触发器生成器：文章设计了一个基于自编码器（autoencoder）的触发器生成器，它根据输入图像生成相应的触发器。生成的触发器具有显著的多样性，确保不同输入图像的触发器彼此不同。2）交叉触发测试：为了进一步提高触发器的隐蔽性，研究者引入了一种新的测试方法，称为交叉触发测试（cross-trigger test）。通过该测试，确保为一个输入生成的触发器无法在其他输入上重用，从而进一步提高了攻击的隐蔽性。3）训练目标函数：文章结合了分类损失（classification loss）和多样性损失（diversity loss），以确保生成的触发器既能有效激活后门，又能在不同输入间保持足够的差异性。  实验结果表明，这种动态后门攻击在MNIST、CIFAR-10和GTSRB等标准数据集上取得了接近100%的攻击成功率，并且在使用现有最先进的防御方法时仍然能够成功绕过检测。文章还证明了该方法在图像正则化和网络检查工具（如GradCam）下的稳健性，这进一步验证了这种攻击的隐蔽性。  动态后门，变来变去，多次结合起来才能更好地看出目的。

*Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning*  之前的backdoor方法通常从训练数据集中随机选择触发候选样本，这种做法容易扰乱样本分布，并模糊它们之间的边界，导致主要任务的准确性下降。此外，这些方法使用的触发器通常是手工制作且未经过优化，导致后门映射关系较弱，攻击成功率较低。  为了解决这些问题，本文提出了一种灵活的后门攻击方法，称为触发样本选择与优化（Trigger Sample Selection and Optimization, TSSO）。这一方法受到神经网络分类模式的启发，利用自编码器（Autoencoders）和局部敏感哈希（Locality-Sensitive Hashing）来选择在类边界处的触发候选样本，从而实现精确注入。此外，TSSO通过全局模型和历史结果迭代优化触发器的表示，从而建立一个稳健的映射关系。 文章在四个经典数据集上评估了TSSO方法，特别是在非独立同分布（non-IID）的设置下，TSSO在更少的训练轮次中取得了更高的攻击成功率，并延长了后门攻击的效果。即使在扩展性测试中，部署防御措施的情况下，TSSO仍然能够在只有4%恶意客户端（中毒率为1/640）的情况下达到超过80%的攻击成功率。这展示了TSSO在后门攻击中的高效性和持久性。 相当于是把单次的攻击变地和正常的训练很类似，因此难以识别。

*Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning* CCFA，研究指出，在联邦学习中，由于后续的正常更新，后门攻击的效果会逐渐减弱，这表现为攻击成功率在多轮迭代中显著下降，最终可能完全失效。为了量化这种现象，文章引入了一个新的指标——攻击持久性（Attack Persistence），用于衡量后门攻击效果的衰减程度。在前人研究未能广泛探讨如何提高攻击持久性的背景下，作者提出了FCBA方法。该方法通过聚合更多的触发信息，生成更完整的后门模式，从而在全局模型中更好地植入后门。经过训练的后门模型对后续的正常更新具有更强的抗性，使得测试集上的攻击成功率更高。作者在三个数据集上对这一方法进行了测试，并在不同的设置下评估了两种模型的表现。结果显示，FCBA的持久性优于现有最先进的联邦学习后门攻击方法。在GTSRB数据集上，经过120轮攻击后，FCBA的攻击成功率较基线提升了50%以上。 核心思想是通过聚合更多的触发信息，生成更复杂和完整的后门模式，从而使得后门在全局模型中植入得更深、更持久。  开源。

*FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information* 提出了一种名为FedRecover的方法，用于在联邦学习（Federated Learning, FL）系统中从中毒攻击中恢复全局模型。FedRecover方法的关键在于利用历史信息来估计客户端的模型更新，而不是在恢复过程中要求客户端重新计算和通信这些更新。这一方法的目的是减少恢复过程中的计算和通信开销，同时保持恢复后的全局模型的准确性。  文章中没有提到是否存储了所有客户端历史上的每次梯度（占据空间过多的问题）。

*Model Poisoning Attacks to Federated Learning via Multi-Round Consistency* 多轮的攻击。PoisonedFL通过引入多轮次一致性（multi-round consistency）和动态攻击幅度调整这两个关键组件，显著提高了攻击效果。该方法不依赖于真实客户端的信息，并且对服务器部署的防御机制具有很强的适应性。 1）多轮次一致性：PoisonedFL通过确保恶意客户端在多个训练轮次中的模型更新方向一致，即使在个别轮次中攻击效果被削弱，累积的攻击效果仍然能够显著地偏移全局模型。2）动态攻击幅度调整：为了避免恶意更新被防御机制完全过滤掉，PoisonedFL动态调整攻击幅度。根据过去轮次的攻击效果，调整恶意更新的强度，以实现攻击的隐蔽性与有效性之间的平衡。  *但是*，这篇文章和我们要检测的攻击正好相反，我们要检测的攻击是那种迂回式攻击，这篇文章的图二说别的攻击可能是“迂回”的，而PoisonedFL结合多轮直奔目标。

### Log009 - 2024.8.29_19:38-2024.9.2_10:35

~~[上次commit](https://github.com/LetMeFly666/SecFFT/blob/557f1469b7693af0f92dfe9a15e300beba527550)是为了开会时在另外一台电脑上看文档，所以充充提交的。~~

开了2h多的会，然后周老师带我和wb去川味滋吃了一顿[坏笑]。这次的目标比较明确。

后天早上9点前写好，Latex版，加上一两张图。

**一、Method(Observation)**

1. 不连续的攻击（时间上不连续）
2. 每次都攻但比较隐蔽
   
   1. 拆分/限缩，单轮次幅度很小
   2. 角度，不直奔目标，迂回曲折达到最终目标

**二、具体方案**

找汇聚问题的数学解决方案。可问问数学系同学。

问题定义：

1. 空间中有一些直线，有没有什么办法找到一个最小的球或圆，把所有的直线或者大部分的直线包括进来？
2. 空间中有一些直线，有没有什么办法找到这些直线的汇聚点

思路：

1. 思路1，来自[安博](https://github.com/aqz6779)：最大通量问题？
2. 思路2，来自KCer：连接每个直线的交点为一个多边形？或者每次连接3个点求三角形的外切圆？tell其找一点到所有直线距离之和最小的思路后：设点的坐标，然后累加所有距离得到一个表达式，让这个表达式值最小，就转化成了一个不等式问题。
3. 思路3，来自我的舞伴wmc：对不相交直线的距离最小化？应该属于有约束的最优化问题，而且是多变量的，应该有很多种解法。
4. 思路4，来自hkx：matlab模拟然后看汇聚点？emm，好像不太可行。
5. 思路n，来自[ChatGPT](https://chatgpt.com/)：找一个点，到所有直线的距离之和最小？
6. 思路n，来自[ChatGPT](https://chatgpt.com/c/25b6a756-917d-4299-a249-ea699b473d54)：几何最小包围球方法（Minimum Enclosing Ball Method）？在谷歌学术上找到了一个[Two algorithms for the minimum enclosing ball problem](https://www.academia.edu/download/88593697/1654.pdf)
7. 思路n，来自[ChatGPT](https://chatgpt.com/c/25b6a756-917d-4299-a249-ea699b473d54)：基于贪心算法的近似方法？1)从某一个候选球心开始，逐渐扩大半径，直到包括尽可能多的直线。

待搜索：

1. 不相交直线的距离最小化
2. 找到一个点到所有直线的距离之和最小
3. 求解器
4. 有约束的最优化

### Log010 - 2024.9.2_23:14-2024.9.3晚

开了大约一个半小时会。下次讨论暂定明晚

> 杨老师新idea：先两条线求最小圆，接着加入第三条线（如果第三条线和这个圆在同一平面就扩大半径，否则就圆变球），接着加入下一条线，...。总体思路是每次加入一条线，扩大半径或升维以包裹全部。可搜：中心距。

1. 画场景图
2. 实验验证距离猜想
3. 理论写高级点

一天有点难捏

+ [[科普中国]-最小圆覆盖算法](https://cloud.kepuchina.cn/newSearch/imgText?id=6969051840972574720)：研究如何寻找能够覆盖平面上一群点的最小圆。这个问题在一般的n维空间中的推广是最小包围球的问题

### Log011 - 2024.9.4_01:03-2024.9.5 10:40

这次讨论了两个多小时，学到了很多东西

1. 上午理论写好
1. 下午场景图画好

如何求球心：[ChatGPT](https://chatgpt.com/c/829bce1b-b20e-493b-9218-7af03ab02346)

**任务**

1. 画图
2. 建模

   近似最小覆盖超球
   
   x是...、球心就是...、半径就是...

   加上一个Top-K

3. 解迭代
4. 信用（效果） -> 实验

**过程**

**隔离森林尝试**

隔离森林：先以二维为例，`通过将 x 坐标、y 坐标和 radius（置信度半径）作为特征输入到隔离森林中`

但是这样置信度相当于只是作为了一个特征，考虑这样一种情况：有10个点，编号0-8是正常点，编号9是异常点。编号0-8的圆心位置很接近，编号9的圆心很偏离。但是编号0的置信度特别高（半径为0.1），而编号1-9的置信度不是很高（半径为1）。这样，将置信度作为一个特征来考虑的话，1-9的“置信度”这一特征就会很接近。但事实是编号0的置信度非常小，非常小才是最好的。这样可能会因为0的半径非常小而将0作为异常点

相当于只把置信度作为了一个普通的特征，而不是置信度越高越好。因此已抛弃。

**局部离群因子（Local Outlier Factor, LOF），使用置信度调整**

LOF是一种现有方法，使用置信度调整是ChatGPT想出来的。（置信度越高，LOF 分数调整得越低）[ChatGPT](https://chatgpt.com/c/c264418d-ec3e-4a65-94a8-dfbf98115e9f)

**基于置信度的加权聚合**

直接依据置信度加权每个客户端上传的梯度就行。当然主观逻辑模型也行，只是会更加复杂一些，不是本次工作的重点。

这样，置信度较低的客户端对模型影响也会较小（例如瞎胡乱训的客户端）。

**小结**

*隔离森林尝试*和*局部离群因子*都属于“将虑置信度也作为异常检测依据的一部分”。但后续发现这种做法似乎并不常见，置信度更适合作为模型聚合或其他决策过程的参考因素。

*基于置信度的加权聚合*是将球心位置作为异常检测的依据，而将半径（置信度）作为聚合权重的依据。

*然后有种调研了一上午，最后发现其实十分简单的感觉。。。*

**异常检验可用博客资源**

+ [【异常检测】数据挖掘领域常用异常检测算法总结](https://www.heywhale.com/mw/project/658d4e937015b4e86be82cb0)

### Log012 - 2024.9.5_11:40-2024.9.6晚

赞哥和我们一起待到了晚上8点半，然后又带我和wb去吃了一顿。

**实验图**

+ [x] database不需要二重框
+ [x] database到空间超球图加上一个箭头
+ [x] 空间超球图每个气泡标注不同的颜色
+ [x] 画一条横穿超球的线，一共四五条线就行

**理论-实验部分**

理论部分3块，参考3篇顶刊来写。

1. 实验环境
2. 数据集描述
3. 对比试验

**理论-轨迹部分**

记得写上伪代码

### Log012 - 2024.9.7_早-2024.9.8_3:21

文字描述

### Log013 - 2024.9.8_8:45-2024.9.9_0:30

代码及实验

1. 实验设置

简略即可

2. 基本

数据集 模型  联邦相关机制（聚合算法）  一定有引用
参数比如学习率 本地轮次 神经元之类的

甚至列表

参数取值表、

3. 攻防：

具体有啥   xx咋了

### Log013 - 2024.9.9_9:30-2024.9.9_20:50

昨晚从8:50和周老师、wb开会到11:32，然后又和wb一起改理论部分到0:30

### Log014 - 2024.9.9_21:40-2024.9.10_2:41

总体任务：

1. 聚合图、理论、代码（今晚）
2. 不信任度融入信任评价（TODO:论文写完后）

具体任务：

- [x] 理论
- [x] 聚合图
- [x] 代码

任务完成，回去睡觉。

### Log015 - 略

### Log016 - 2024.9.13_20:00-2024.9.14_4:54

画实验结果图

+ 最终选用攻击：NEUROTOXIN、MR
+ 最终选用数据集：FMNIST
+ 最终选用防御：cosine、fltrust、flame

实验三进度：

- [x] 左上角：NEUR，COS
- [x] 左下角：MR，COS
- [ ] 一行左二：
- [ ] 二行左二：
- [ ] 一行左三：
- [ ] 二行左三：
- [ ] 右上角：
- [ ] 右下角：

### Log017 - 2024.9.14_12:00-2024.9.14_19:30

赞哥又带我俩去吃饭了。

[Log016](#log016---2024913_2000-2024914_454)废弃。

+ Motivation写长点
+ Motivation引用（限制大小、限制角度、限制符号）
+ 算法不每个部分都提，最后写  我们将算法写成伪代码，第几行到第几行

### Log018 - 2024.9.14_20:10-2024.9.15_11:14

画实验图

和表

和改

和改

和改
和改

## End

日志里面有很多*有的没的*的记录。边增边删，Perfect!。