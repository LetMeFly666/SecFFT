<!--
 * @Author: LetMeFly
 * @Date: 2024-09-13 10:38:02
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-14 01:07:56
-->
# SecFFT - branch[wb.002.clip_lora](https://github.com/LetMeFly666/SecFFT/tree/wb.002.clip_lora)

Changed from [RoseAgg](https://github.com/SleepedCat/RoseAgg).

## TODO 

- [x] 数据画热力图，大图及其小图，以及标注
- [x] latex表格
- [x] 给真实的恶意结果以及识别出来的结果，求出对应的所需的分数。
- [ ] 使用GPT将计算出来的结果汇总成表格，并让其返回latex源码

所需数据：

- [ ] 每轮次模型、梯度
- [ ] 有梯度就有cos相似度、有
- [x] 梯度存在哪儿    resultWithTime/model_update/  使用pickle 拼接
- [ ] 有存分吗
- [ ] 识别结果如何