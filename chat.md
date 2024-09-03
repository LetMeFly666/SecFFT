<!--
 * @Author: LetMeFly
 * @Date: 2024-08-18 10:06:39
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-02 10:18:58
-->
有没有针对联邦学习的时间上不连续的backdoor攻击？最好是近几年提出的




一定要是论文




多找几篇




第一篇文章很好，第二第三篇没有体现时间上不连续






所以你要帮我找几篇时间上不连续的后门攻击方式




介绍论文`Time-Distributed Backdoor Attacks on Federated Spiking Learning`
```
This paper investigates the vulnerability of spiking neural networks (SNNs) and federated learning (FL) to backdoor attacks using neuromorphic data. Despite the efficiency of SNNs and the privacy advantages of FL, particularly in low-powered devices, we demonstrate that these systems are susceptible to such attacks. We first assess the viability of using FL with SNNs using neuromorphic data, showing its potential usage. Then, we evaluate the transferability of known FL attack methods to SNNs, finding that these lead to suboptimal attack performance. Therefore, we explore backdoor attacks involving single and multiple attackers to improve the attack performance. Our primary contribution is developing a novel attack strategy tailored to SNNs and FL, which distributes the backdoor trigger temporally and across malicious devices, enhancing the attack's effectiveness and stealthiness. In the best case, we achieve a 100 attack success rate, 0.13 MSE, and 98.9 SSIM. Moreover, we adapt and evaluate an existing defense against backdoor attacks, revealing its inadequacy in protecting SNNs. This study underscores the need for robust security measures in deploying SNNs and FL, particularly in the context of backdoor attacks.
```





仔细阅读PDF，并详细介绍其具体流程





如果使用WinToGo的话，可以使用台式机上的资源吗




WinToGo无法访问台式机上的硬盘吗




介绍文章`Time-Distributed Backdoor Attacks on Federated Spiking Learning`关于时间不连续的部分




git clone如何只clone一个分支




一个git仓库的两次commit的hash相同的话会怎样？




结合之前的PDF，再次介绍文章`Beyond traditional threats: A persistent backdoor attack on federated learnin`
```
Backdoors on federated learning will be diluted by subsequent benign updates. This is reflected in the significant reduction of attack success rate as iterations increase, ultimately failing. We use a new metric to quantify the degree of this weakened backdoor effect, called attack persistence. Given that research to improve this performance has not been widely noted, we propose a Full Combination Backdoor Attack (FCBA) method. It aggregates more combined trigger information for a more complete backdoor pattern in the global model. Trained backdoored global model is more resilient to benign updates, leading to a higher attack success rate on the test set. We test on three datasets and evaluate with two models across various settings. FCBA's persistence outperforms SOTA federated learning backdoor attacks. On GTSRB, post-attack 120 rounds, our attack success rate rose over 50% from baseline. The core code of our method is available at https://github.com/PhD-TaoLiu/FCBA.
```



如果是两个不同的分支上进行了相同的提交呢？结果会怎样，直接合并到一个分支里了吗






空间中有一些直线，有没有什么办法找到一个最小的球或圆，把所有的直线或者大部分的直线包括进来






我是23级硕士生，我选了3门培养方案之外的课，可以作为研究方向课吗？





MHS-3.5inch RPi Display 是什么




树莓派长什么样





树莓派如何使用




需要





树莓派不自带存储设备吗




Git中一些文件并没有更改，但是git status中仍然会显示。





请解释一下专硕的`专业实践`和`综合素质实践`




具体应该怎么完成




介绍文章`CoBA: Collusive Backdoor Attacks with Optimized Trigger to Federated Learning`

```
Considerable efforts have been devoted to addressing distributed backdoor attacks in federated learning (FL) systems. While significant progress has been made in enhancing the security of FL systems, our study reveals that there remains a false sense of security surrounding FL. We demonstrate that colluding malicious participants can effectively execute backdoor attacks during the FL training process, exhibiting high sparsity and stealthiness, which means they can evade common defense methods with only a few attack iterations. Our research highlights this vulnerability by proposing a Co llusive B ackdoor A ttack named CoBA . CoBA is designed to enhance the sparsity and stealthiness of backdoor attacks by offering trigger tuning to facilitate learning of backdoor training data, controlling the bias of malicious local model updates, and applying the projected gradient descent technique. By conducting extensive empirical studies on 5 benchmark datasets, we make the following observations: 1) CoBA successfully circumvents 15 state-of-the-art defense methods for robust FL; 2) Compared to existing backdoor attacks, CoBA consistently achieves superior attack performance; and 3) CoBA can achieve persistent poisoning effects through significantly sparse attack iterations. These findings raise substantial concerns regarding the integrity of FL and underscore the urgent need for heightened vigilance in defending against such attacks.
```




再次介绍`Input-Aware Dynamic Backdoor Attack`这篇文章




再次介绍`Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning`这篇文章




有没有不直奔目的，迂回前进式的攻击




给出具体文章





介绍文章`Stealthy Backdoor Attack against Federated Learning through Frequency Domain`
```
Federated Learning (FL) is a beneficial decentralized learning approach for preserving the privacy of local datasets of distributed agents. However, the distributed property of FL and untrustworthy data introducing the vulnerability to backdoor attacks. In this attack scenario, an adversary manipulates its local data with a specific trigger and trains a malicious local model to implant the backdoor. During inference, the global model would misbehave for any input with the trigger to the attacker-chosen prediction. Most existing backdoor attacks against FL focus on bypassing defense mechanisms, without considering the inspection of model parameters on the server. These attacks are susceptible to detection through dynamic clustering based on model parameter similarity. Besides, current methods provide limited imperceptibility of their trigger in the spatial domain. To address these limitations, we propose a stealthy backdoor attack called “Chironex” against FL with an imperceptible trigger in frequency space to deliver attack effectiveness, stealthiness and robustness against various countermeasures on FL. We first design a frequency trigger function to generate an imperceptible frequency trigger to evade human inspection. Then we fully exploit the attacker’s advantage to enhance attack robustness by estimating benign updates and analyzing the impact of the backdoor on model parameters through a task-sensitive neuron searcher. It disguises malicious updates as benign ones by reducing the impact of backdoor neurons that greatly contribute to the backdoor task based on activation value, and encouraging them to update towards benign model parameters trained by the attacker. We conduct extensive experiments on various image classifiers with real-world datasets to provide empirical evidence that Chironex can evade the most recent robust FL aggregation algorithms, and further achieve a distinctly higher attack success rate than existing attacks, without undermining the utility of the global model.
```




从一段HTML中提取passkey的值。

```
<tr><td style="width: 1%" class="rowhead nowrap">passkey</td><td class="rowfollow" style="text-align: left">54sieiouihsihfiu8y3ihsjkhfk</td></tr>
```

这些class可能会变，但是一定会有一个td，内容是passkey，紧接着有一个td，里面是具体的数值，格式是字符串，这个就是我们要找的值。





python的A文件需要引用B文件的CONFIG，而CONFIG中一些数据需要由A文件中的一个函数得到。我应该怎么办





具体来说，我有一个config.py，这个文件会从配置文件中读取配置，并将cookie作为一个方法。
我还有一个BYR.py，里面有一个函数，根据config的cookie获取passkey。
passkey应该作为CONFIG的一个变量，我想在CONFIG初始化的时候，调用BYR.py的getPasskeyByCookie函数，获取passkey。
但是这个函数的执行需要CONFIG的cookie，导致了互相引用。我应该怎么解决比较好？




BYR.py的其他函数依赖CONFIG.py，总不能把BYR.py的函数全部修改掉






现在我的代码结构如下(demo)：

```
# .\BYR.py
from config import CONFIG

def main():
    print(CONFIG.cookie)
    print(CONFIG.passkey)

def getPasskeyByCookie(cookie: str) -> str:
    print(CONFIG.cookie)  # 诶，这里面也能使用CONFIG
    return f'{cookie}123'

if __name__ == '__main__':
    main()
```

```
# .\config.py
class Config:
    def __init__(self) -> None:
        self.cookie = '123'
        self._passkey = None
    
    @property
    def passkey(self) -> str:
        if not self._passkey:
            from BYR import getPasskeyByCookie
            self._passkey = getPasskeyByCookie(self.cookie)
        return self._passkey

CONFIG = Config()
```

```
# .\__init__.py
from .BYR import getPasskeyByCookie
from .config import CONFIG
```

为什么这样就可行了？

我在获取CONFIG.passkey的时候调用了BYR.py的getPasskeyByCookie，而BYR.py的getPasskeyByCookie使用了config.py的CONFIG






最终决定使用求“一个点到所有直线距离之和最短”的方法。





很棒，请返回问题描述、优化目标、求解方法 的latex源码