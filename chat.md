<!--
 * @Author: LetMeFly
 * @Date: 2024-08-18 10:06:39
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-08-29 17:23:10
-->
git push Let wb.001.lora 
error: src refspec wb.001.lora does not match any
error: failed to push some refs to 'github.com:LetMeFly666/SecFFT.git'







我当前处于一个叫main的分支，我想将其作为Let的wb.001.lora分支





如何免密登录到Windows的ssh服务器？





有没有net restart命令？net stop后连接可能就断开了






配置文件的最后两行是什么意思？

```
Match Group administrators
       AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys
```





为何我配置好了sshd_config，重启了sshd服务，添加了.ssh/authorized_keys，登录的时候还是需要输入密码？




$ icacls %USERPROFILE%\.ssh /inheritance:r
%USERPROFILE%.ssh: 系统找不到指定的文件。
已成功处理 0 个文件; 处理 1 个文件时失败






Get-Process -Name sshd确实发现有多个sshed进程在运行





仔细阅读PDF，解释文章`Model Poisoning Attacks to Federated Learning via Multi-Round Consistency`






我的台式机只有一个机械硬盘，为了提升体验，我想购置一块固态硬盘，并将系统装入固态硬盘中，可行吗？






解释这篇文章的`Figure 2: Illustration of the global-model evolution in three training rounds under existing attacks and PoisonedFL. The attack effect self-cancels in existing attacks, while PoisonedFL consistently moves the global model along the same direction.`





我可以选择使用移动固态硬盘吗




外接SSD可以直接安装Windows系统吗？这样我将这个SSD连接到另一台电脑上时，能直接通过设置BIOS而直接启动吗











这样直奔目标不加掩饰会不会很容易被检测？






解释这篇文章`Defending against Backdoor Attacks in Natural Language Generation`
```
The frustratingly fragile nature of neural network models make current natural language generation (NLG) systems prone to backdoor attacks and generate malicious sequences that could be sexist or offensive. Unfortunately, little effort has been invested to how backdoor attacks can affect current NLG models and how to defend against these attacks. In this work, by giving a formal definition of backdoor attack and defense, we investigate this problem on two important NLG tasks, machine translation and dialog generation. Tailored to the inherent nature of NLG models (e.g., producing a sequence of coherent words given contexts), we design defending strategies against attacks. We find that testing the backward probability of generating sources given targets yields effective defense performance against all different types of attacks, and is able to handle the one-to-many issue in many NLG tasks such as dialog generation. We hope that this work can raise the awareness of backdoor risks concealed in deep NLG systems and inspire more future work (both attack and defense) towards this direction.
```





再次详细解释这篇文章
```
*Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning*  之前的backdoor方法通常从训练数据集中随机选择触发候选样本，这种做法容易扰乱样本分布，并模糊它们之间的边界，导致主要任务的准确性下降。此外，这些方法使用的触发器通常是手工制作且未经过优化，导致后门映射关系较弱，攻击成功率较低。  为了解决这些问题，本文提出了一种灵活的后门攻击方法，称为触发样本选择与优化（Trigger Sample Selection and Optimization, TSSO）。这一方法受到神经网络分类模式的启发，利用自编码器（Autoencoders）和局部敏感哈希（Locality-Sensitive Hashing）来选择在类边界处的触发候选样本，从而实现精确注入。此外，TSSO通过全局模型和历史结果迭代优化触发器的表示，从而建立一个稳健的映射关系。 文章在四个经典数据集上评估了TSSO方法，特别是在非独立同分布（non-IID）的设置下，TSSO在更少的训练轮次中取得了更高的攻击成功率，并延长了后门攻击的效果。即使在扩展性测试中，部署防御措施的情况下，TSSO仍然能够在只有4%恶意客户端（中毒率为1/640）的情况下达到超过80%的攻击成功率。这展示了TSSO在后门攻击中的高效性和持久性。
```




再次详细解释这篇文章
```
*Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning* CCFA，研究指出，在联邦学习中，由于后续的正常更新，后门攻击的效果会逐渐减弱，这表现为攻击成功率在多轮迭代中显著下降，最终可能完全失效。为了量化这种现象，文章引入了一个新的指标——攻击持久性（Attack Persistence），用于衡量后门攻击效果的衰减程度。在前人研究未能广泛探讨如何提高攻击持久性的背景下，作者提出了FCBA方法。该方法通过聚合更多的触发信息，生成更完整的后门模式，从而在全局模型中更好地植入后门。经过训练的后门模型对后续的正常更新具有更强的抗性，使得测试集上的攻击成功率更高。作者在三个数据集上对这一方法进行了测试，并在不同的设置下评估了两种模型的表现。结果显示，FCBA的持久性优于现有最先进的联邦学习后门攻击方法。在GTSRB数据集上，经过120轮攻击后，FCBA的攻击成功率较基线提升了50%以上。开源。
```



再次详细解释这篇文章
```
*FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information* 提出了一种名为FedRecover的方法，用于在联邦学习（Federated Learning, FL）系统中从中毒攻击中恢复全局模型。FedRecover方法的关键在于利用历史信息来估计客户端的模型更新，而不是在恢复过程中要求客户端重新计算和通信这些更新。这一方法的目的是减少恢复过程中的计算和通信开销，同时保持恢复后的全局模型的准确性。
```



这篇文章是把客户端的每次历史梯度全部保存在中央服务器上吗？