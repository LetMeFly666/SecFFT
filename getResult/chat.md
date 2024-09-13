pytorch初始化长度为0的一维向量。

介绍pytorch如何cat拼接向量






python dict取键值




```
pklName = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg_NEUROTOXIN-fmnist/model_updates/fmnist_NEUROTOXIN_10.pkl'
with open(pklName, 'rb') as f:
    update: Dict[str, torch.Tensor] = pickle.load(f)
# print(update)
# weight0 = update['base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight']
# weight0.shape  # torch.Size([50, 12288])
userUpdates = [torch.empty(0) for _ in range(50)]
for layerKey, values in update.items():
    print(layerKey)
    print(values)
    print(values.shape)  # torch.Size([50, 12288])
    print(values.shape)
    break
```
补全代码，将数据cat到每一个user中。





我有一个参与数组，其中30列代表30个轮次。第一行的表头，剩下的50行是50个客户端。
```
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
20,39,33,15,23,20,27,36,23,35,20,45,17,21,14,47,30,15,12,13,46,35,23,6,13,4,27,23,43,20
35,8,32,23,45,41,21,10,37,24,41,8,44,33,23,15,4,0,33,32,5,39,6,3,14,30,24,13,44,8
25,23,17,30,42,7,46,49,13,15,15,49,28,1,3,26,20,45,23,38,40,1,46,13,49,27,46,41,45,17
36,19,30,41,43,6,48,9,8,30,21,4,26,30,41,10,12,32,27,46,17,31,2,44,25,7,28,40,30,36
44,7,40,47,22,12,38,45,21,33,19,14,24,36,43,38,38,39,48,28,12,42,7,36,18,15,32,44,33,1
42,20,35,12,17,16,1,2,14,45,13,24,0,26,29,42,39,1,39,5,27,34,43,0,22,20,0,28,13,7
32,6,11,1,27,46,17,17,10,42,36,11,39,17,1,18,48,30,37,25,2,48,15,48,40,24,29,24,5,21
0,30,0,45,31,3,39,37,33,46,1,9,30,42,32,13,2,10,49,49,43,15,21,39,31,35,22,45,29,39
26,26,5,39,8,4,30,34,43,34,12,39,3,40,26,43,26,33,17,10,0,0,38,5,5,22,9,27,39,45
2,47,25,42,40,28,47,19,48,32,26,0,34,46,12,33,13,18,0,12,31,17,11,25,48,14,8,36,16,22
5,22,21,5,49,39,16,27,7,20,24,32,4,12,25,7,14,5,35,42,20,30,37,16,29,1,12,12,36,34
49,41,42,0,16,19,19,30,19,1,45,1,10,23,35,40,16,9,9,16,38,45,35,15,0,19,40,37,24,23
28,24,1,13,48,11,42,20,11,31,44,42,47,5,40,19,40,21,22,31,24,26,17,4,2,40,33,17,40,6
39,14,43,33,20,43,6,26,26,11,48,7,29,16,47,45,31,17,18,9,18,16,18,35,20,46,19,31,46,41
16,10,24,27,30,25,12,11,24,27,0,20,12,11,11,25,5,28,41,3,23,13,0,20,7,3,15,14,23,16
1,17,3,11,6,2,45,8,38,7,25,41,31,22,10,5,44,3,24,36,19,28,16,1,1,6,1,25,11,11
14,13,38,2,46,47,8,15,47,36,31,27,8,10,9,48,34,40,46,34,45,38,42,31,44,11,47,11,27,32
6,18,39,26,19,9,20,33,31,28,34,47,37,13,33,3,24,27,15,37,36,25,27,19,17,32,11,43,8,43
7,36,36,37,36,8,22,46,9,39,8,30,35,49,46,34,28,38,43,27,9,40,40,17,4,43,34,46,35,47
41,0,29,32,9,13,40,43,44,41,4,43,41,8,6,36,8,12,3,4,22,4,26,37,24,38,43,19,9,35
34,42,14,22,34,21,34,6,36,17,35,28,1,39,4,22,23,47,8,17,42,12,41,23,30,48,45,26,32,28
48,21,28,4,11,49,3,35,6,37,49,3,5,19,20,49,29,24,45,19,48,3,44,47,34,8,13,5,10,49
30,2,41,20,29,40,25,21,46,0,29,35,2,14,22,9,7,29,36,21,49,43,1,30,3,49,3,30,26,31
21,31,37,24,5,26,11,24,42,5,37,15,46,24,42,44,25,11,10,20,37,5,22,26,8,10,36,10,15,19
23,28,18,14,12,33,49,7,45,19,42,40,20,0,36,4,32,8,20,15,47,7,31,9,45,0,21,34,18,9
17,27,44,29,38,14,29,28,15,44,5,26,42,44,16,1,41,14,25,44,33,47,14,8,41,37,26,0,0,25
45,40,7,31,28,27,36,23,12,16,28,44,11,6,44,32,42,41,21,35,28,21,36,18,37,39,14,47,1,4
27,15,48,48,26,29,32,29,22,48,23,10,13,18,49,30,33,2,13,43,35,9,20,45,9,25,16,29,14,48
8,44,4,21,18,18,7,44,41,47,9,21,7,38,27,29,15,7,29,33,15,2,8,14,6,17,6,33,20,15
11,5,8,18,24,32,9,41,28,49,10,34,25,45,15,27,37,43,47,22,14,32,30,21,32,21,17,18,42,30
10,43,34,49,35,37,37,14,17,38,47,29,6,15,2,0,47,20,38,0,21,8,48,41,21,42,39,21,49,13
13,16,31,43,44,22,5,32,35,2,17,17,9,48,30,39,22,44,40,45,32,29,5,2,26,12,48,2,22,26
24,37,13,9,32,31,2,31,29,23,39,48,23,29,48,37,35,34,14,8,10,36,24,24,33,23,10,1,47,44
43,11,15,17,15,36,35,42,5,14,30,19,19,47,18,6,18,49,42,30,30,41,49,33,42,36,5,15,4,5
15,1,20,7,13,24,31,40,1,12,22,46,48,43,17,12,10,36,32,26,7,33,39,46,43,28,44,9,21,42
9,9,6,3,10,10,0,16,32,10,6,25,14,34,8,23,1,26,7,1,34,10,32,28,35,9,49,39,2,18
12,3,9,34,7,23,43,4,34,22,3,23,21,27,34,35,36,4,4,24,41,46,33,11,28,2,31,32,41,46
38,12,26,38,37,34,24,5,40,4,46,18,32,28,5,2,17,46,30,7,44,24,9,10,16,16,41,49,28,38
40,32,27,40,0,38,14,1,3,29,32,13,15,37,37,41,49,16,11,29,13,6,45,43,19,33,25,6,12,37
47,29,19,19,41,45,33,25,4,6,16,12,16,2,28,24,46,42,26,40,11,20,34,7,27,44,2,16,34,3
4,34,12,28,4,15,10,48,30,13,18,31,45,25,7,20,6,25,6,48,6,49,25,49,46,13,38,3,38,29
3,46,16,10,14,17,13,3,18,21,11,38,49,32,21,16,0,6,1,11,39,44,29,29,47,29,23,7,7,0
33,48,49,46,25,42,28,13,0,40,2,6,22,41,38,8,11,35,28,39,29,37,19,27,15,31,7,38,48,12
37,49,47,6,33,48,15,18,49,3,27,33,33,20,39,21,21,19,34,23,16,27,28,12,23,18,42,48,37,24
22,25,2,35,1,30,41,38,27,25,40,2,40,31,31,28,45,31,2,6,25,19,13,42,36,45,18,4,17,10
29,38,22,44,2,44,26,22,25,26,33,16,38,4,45,46,43,23,5,41,8,23,47,34,38,5,30,22,25,2
46,35,45,25,3,5,44,0,39,18,14,36,18,35,24,31,19,37,16,14,26,18,10,40,11,26,37,35,31,27
18,45,23,8,21,1,4,39,16,43,7,37,27,3,19,17,27,48,31,2,3,14,12,22,10,41,35,20,6,33
19,33,46,36,47,35,18,12,20,9,38,5,36,9,13,11,9,22,44,47,1,22,3,38,12,34,20,8,19,14
31,4,10,16,39,0,23,47,2,8,43,22,43,7,0,14,3,13,19,18,4,11,4,32,39,47,4,42,3,40
```
这个文件中，第一列的内容（表头除外）为`20、35、...`，代表.pkl文件中每个键对应值的50个维度为客户端`20、35、...`的梯度。
请你处理这个文件，并将.pkl文件中对应的梯度添加到数组中对应的位置。










```
roundNum = 10
dirPath = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg_NEUROTOXIN-fmnist/'
pklName = os.path.join(dirPath, f'model_updates/fmnist_NEUROTOXIN_{roundNum}.pkl')
participantFilePath = os.path.join(dirPath, 'participants/participants.csv')
# participantFileDF = pd.read_csv(participantFilePath)
# # 提取参与者数组（第一列）和轮次（从第二列开始的部分）
# clients = participantFileDF.iloc[1:, 0].tolist()  # 客户端列表
# roundsArray = participantFileDF.iloc[1:, 1:].to_numpy()  # 轮次数据（50行x30列）
# print(roundsArray)
participants = np.genfromtxt(
    participantFilePath, delimiter=",", dtype=None, encoding="utf-8"
)
participants = participants[1:].T
participants_thisRound = participants[roundNum]
print(participants_thisRound)

with open(pklName, 'rb') as f:
    update: Dict[str, torch.Tensor] = pickle.load(f)
# print(update)
# weight0 = update['base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight']
# weight0.shape  # torch.Size([50, 12288])
userUpdates = [torch.empty(0) for _ in range(50)]
for layerKey, values in update.items():
    # print(layerKey)
    # print(values)
    # print(values.shape)  # torch.Size([50, 12288])
    for i in range(values.shape[0]):
        userUpdates[participants_thisRound[i]] = torch.cat((userUpdates[participants_thisRound[i]], values[i]), 0)
        print(userUpdates)
        break
    break
```
报错
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
c:\Users\admin\Desktop\LLM\wb2\Codes\getResult\getUpdates.ipynb Cell 4 line 2
     23 for layerKey, values in update.items():
     24     # print(layerKey)
     25     # print(values)
     26     # print(values.shape)  # torch.Size([50, 12288])
     27     for i in range(values.shape[0]):
---> 28         userUpdates[participants_thisRound[i]] = torch.cat((userUpdates[participants_thisRound[i]], values[i]), 0)
     29         print(userUpdates)
     30         break

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument tensors in method wrapper_CUDA_cat)
```





将这段代码的userUpdates修改为torch.tensor，并将所有数据转移到GPU上
```
roundNum = 10
dirPath = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg_NEUROTOXIN-fmnist/'
pklName = os.path.join(dirPath, f'model_updates/fmnist_NEUROTOXIN_{roundNum}.pkl')
participantFilePath = os.path.join(dirPath, 'participants/participants.csv')
# participantFileDF = pd.read_csv(participantFilePath)
# # 提取参与者数组（第一列）和轮次（从第二列开始的部分）
# clients = participantFileDF.iloc[1:, 0].tolist()  # 客户端列表
# roundsArray = participantFileDF.iloc[1:, 1:].to_numpy()  # 轮次数据（50行x30列）
# print(roundsArray)
participants = np.genfromtxt(
    participantFilePath, delimiter=",", dtype=None, encoding="utf-8"
)
participants = participants[1:].T
participants_thisRound = participants[roundNum]
print(participants_thisRound)

with open(pklName, 'rb') as f:
    update: Dict[str, torch.Tensor] = pickle.load(f)
# print(update)
# weight0 = update['base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight']
# weight0.shape  # torch.Size([50, 12288])
userUpdates = [torch.empty(0) for _ in range(50)]
for layerKey, values in update.items():
    # print(layerKey)
    # print(values)
    # print(values.shape)  # torch.Size([50, 12288])
    for i in range(values.shape[0]):
        trueUser = participants_thisRound[i]
        userUpdates[trueUser] = torch.cat((userUpdates[trueUser], values[i].cpu()), 0)
print(userUpdates[0].shape)
```





```
dirPath = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg_NEUROTOXIN-fmnist/'
pklPrefix = 'fmnist_NEUROTOXIN'
```
通过dirPath得到pklPrefix





不，你得到的结果是2024，而不是'fmnist_NEUROTOXIN'






这段代码将`userUpdates`的格式修改为tensor，而不是list
```
def get_all_user_updates(roundNum: int, dirPath: str) -> torch.Tensor:
    # 设置轮次编号和文件路径
    roundNum = 10
    dirPath = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg-fmnist_NEUROTOXIN/'
    pklPrefix = os.path.basename(os.path.normpath(dirPath)).split('-')[-1]
    pklName = os.path.join(dirPath, f'model_updates/{pklPrefix}_{roundNum}.pkl')
    participantFilePath = os.path.join(dirPath, 'participants/participants.csv')

    # 读取参与者数组
    participants = np.genfromtxt(
        participantFilePath, delimiter=",", dtype=None, encoding="utf-8"
    )
    participants = participants[1:].T
    participants_thisRound = participants[roundNum]  # 获取当前轮次的参与者
    print(participants_thisRound)

    # 加载 .pkl 文件
    with open(pklName, 'rb') as f:
        update: Dict[str, torch.Tensor] = pickle.load(f)

    # 初始化一个列表来存储每个客户端的更新，使用 GPU 张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    userUpdates = [torch.empty((0,), device=device) for _ in range(50)]  # 假设50个客户端

    # 遍历每个层的更新
    for layerKey, values in update.items():
        # 将每个值移到 GPU 上
        values = values.to(device)
        for i in range(values.shape[0]):  # 遍历每个客户端的梯度
            trueUser = participants_thisRound[i]  # 获取当前轮次的客户端ID
            # 如果为空张量
            if userUpdates[trueUser].numel() == 0:
                userUpdates[trueUser] = values[i]  # 直接赋值
            else:
                userUpdates[trueUser] = torch.cat((userUpdates[trueUser], values[i]), 0)  # 在GPU上拼接

    # 打印第一个用户的更新形状来检查结果
    print(userUpdates[0].shape)
    return userUpdates
```






python list[tensor]转tensor






写一个余弦相似度函数，计算一个二维tensor例如torch.Size([50, 2674688])50个客户端之间的余弦相似度。





修改这段代码：
1. num_clients=50
2. 只有余弦相似度是50x50的数组，其余3种防御方式为1x50的数组。




其余三种方式虽然是1x50的数组，但是也要画成二维图。




不，其余的3种防御方式要自己计算两两之间的相似度




很棒！修改这段代码，写一个函数以供其他文件调用。
这个函数接受两个参数，是攻击A和攻击B的`Cosine Similarity Detection`检测方式的余弦相似度的50x50的tensor向量。
攻击A和B分别是：NEUROTOXIN、MR