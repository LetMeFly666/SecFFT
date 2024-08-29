<!--
 * @Author: LetMeFly
 * @Date: 2024-08-18 10:06:39
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-08-29 11:09:07
-->
cp命令显示拷贝进度



深度学习 有哪些比较隐蔽的后门攻击？最好是联邦学习相关的。




有没有不连续的攻击？或者说攻击者不直接向目标诱导模型梯度，而是潜在迂回地诱导模型梯度，最终到达目标地点。




有无`迂回后门攻击（Roundabout Backdoor Attack）`相关的论文





详细解释`持久性后门攻击（Persistent Backdoor Attack）`





解释文章`Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning`的摘要：

```
Backdoors on federated learning will be diluted by subsequent benign updates. This is reflected in the significant reduction of attack success rate as iterations increase, ultimately failing. We use a new metric to quantify the degree of this weakened backdoor effect, called attack persistence. Given that research to improve this performance has not been widely noted, we propose a Full Combination Backdoor Attack (FCBA) method. It aggregates more combined trigger information for a more complete backdoor pattern in the global model. Trained backdoored global model is more resilient to benign updates, leading to a higher attack success rate on the test set. We test on three datasets and evaluate with two models across various settings. FCBA's persistence outperforms SOTA federated learning backdoor attacks. On GTSRB, post-attack 120 rounds, our attack success rate rose over 50% from baseline. The core code of our method is available at https://github.com/PhD-TaoLiu/FCBA.
```




解释文章`Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning`

```
Federated learning systems encounter various security risks, including backdoor, inference and adversarial attacks. Backdoor attacks within this context generally require careful trigger sample design involving candidate selection and automated optimization. Previous methods randomly selected trigger candidates from training dataset, disrupting sample distribution and blurring boundaries among them, which adversely affected the main task accuracy. Moreover, these methods employed non-optimized handcrafted triggers, resulting in a weakened backdoor mapping relationship and lower attack success rates. In this work, we propose a flexible backdoor attack approach, Trigger Sample Selection and Optimization (TSSO), motivated by neural network classification patterns. TSSO employs autoencoders and locality-sensitive hashing to select trigger candidates at class boundaries for precise injection. Furthermore, it iteratively refines trigger representations via the global model and historical outcomes, establishing a robust mapping relationship. TSSO is evaluated on four classical datasets with non-IID settings, outperforming state-of-the-art methods by achieving higher attack success rate in fewer rounds, prolonging the backdoor effect. In scalability tests, even with the defense deployed, TSSO achieved the attack success rate of over 80% with only 4% malicious clients (a poisoning rate of 1/640).
```




仔细阅读这个PDF，详细介绍`Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning`这篇文章的具体细节。






Windows如何开启SSH





解释`Input-Aware Dynamic Backdoor Attack`

```
In recent years, neural backdoor attack has been considered to be a potential security threat to deep learning systems. Such systems, while achieving the state-of-the-art performance on clean data, perform abnormally on inputs with predefined triggers. Current backdoor techniques, however, rely on uniform trigger patterns, which are easily detected and mitigated by current defense methods. In this work, we propose a novel backdoor attack technique in which the triggers vary from input to input. To achieve this goal, we implement an input-aware trigger generator driven by diversity loss. A novel cross-trigger test is applied to enforce trigger nonreusablity, making backdoor verification impossible. Experiments show that our method is efficient in various attack scenarios as well as multiple datasets. We further demonstrate that our backdoor can bypass the state of the art defense methods. An analysis with a famous neural network inspector again proves the stealthiness of the proposed attack. Our code is publicly available.
```




可以远程在linux上执行命令吗






仔细阅读PDF，详细解释`Input-Aware Dynamic Backdoor Attack`这篇文章






可以远程在linux上执行命令吗，直接执行的那种。而不是说我需要先连接上服务器，再执行命令。最好是我使用现成的命令，直接一句命令就在远程Linux服务器上执行。





你理解错了，这样还是要先连接到服务器再执行命令。比如我想在linux上执行一个ffmpeg命令。我应该如何做？不能先登录服务器，再执行命令。





好的，很棒。我想scp传输一些文件，要求传输当前目录下除了share目录的所有文件。





Linux拷贝文件并显示进度





解释`rsync -a --info=progress2 源文件路径 目标路径`






写一个bash脚本，接受一个时间参数，然后显示进度条并且在时间到达时进度条达到100%。

例如`Timer.sh 5`，则显示一个不断更新的进度条，5秒后走完。





Windows tree命令设置最大显示深度






仔细阅读PDF，并解释`FedRecover Recovering from Poisoning Attacks in Federated Learning using Historical Information`这篇文章





git如何加入子git




子模块的修改可以给原仓库提PR吗




`FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information`具体是怎么做的？





存储历史上每次的更新会不会占用过多的空间？





git clone了一个仓库，如何将其作为另一个仓库的一个新分支？






解释`git fetch`命令，以及它都会干什么