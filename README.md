<!--
 * @Author: LetMeFly
 * @Date: 2024-09-13 10:38:02
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-14 01:51:09
-->
# SecFFT - branch[wb.002.clip_lora](https://github.com/LetMeFly666/SecFFT/tree/wb.002.clip_lora)

Changed from [RoseAgg](https://github.com/SleepedCat/RoseAgg).

## Nothing

```
"image/svg+xml": [
    "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"800\" height=\"20\">",
    "<text x=\"0\" y=\"15\" fill=\"red\">https://github.com/LetMeFly666/SecFFT/releases/download/v0.0.0/result.2x4.ReLiTu.trueAndRand.png</text>",
    "</svg>"
],
```

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

NormalRun的resultWithTime是一个转置的矩阵，第一行`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29`代表表头30轮，每一列是参与者（其实就是0-49），0-19是恶意。

<!-- + 最终选用攻击：NEUROTOXIN、MR
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
- [ ] 右下角： -->

防御对比：Foolsgold、fltrust、flame、ours
攻击选择：NEUROTOXIN、NEUROTOXIN+大小限缩、NEUROTOXIN+大小方向限缩
数据集选择：fmnist

数据来源：

+ [x] foolsgold: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/foolsgold_NEUROTOXIN_09132315-fmnist`
- [x] fltrust: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist`
- [x] flame: `./NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-30-57-flame-fmnist_NEUROTOXIN`、`./NormalRun/FL_Backdoor_CV/saved_models/Revision_1/flame_NEUROTOXIN_09122330-fmnist`


`../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist/fltrust_15.pth`

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