我有一个50x50的余弦相似度数组，准备画热力图，如何将对角线标注为最浅的颜色




余弦相似度数组是torch.tensor





给一个余弦相似度数组，如何通过聚类识别其中的恶意客户端







不用绘图，只需要写一个函数，给你一个50x50的余弦相似度数组，返回恶意客户端的下标




填写这个表格
```
\begin{table*}[h!]
\small
\centering
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
Attac & Defense & FP & TP & FN & TN & Accuracy & Precision & Recall & Specificity & FPR & FNR & AUC & MCC \\ \hline

\multirow{4}{*}{NERO} &Foolsgold & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & FLTrust & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & Flame & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & SecFFT & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \hline
\end{tabular}
\caption{Classification metrics results under different attack and defense methods}
\end{table*}

```
其中数据为：
```
(10, 0, 35, 5, 0.9, 1.0, 0.6666666666666666, 1.0, 0.0, 0.3333333333333333, np.float64(0.8333333333333333), np.float64(0.7637626158259734))
(6, 0, 35, 9, 0.82, 1.0, 0.4, 1.0, 0.0, 0.6, np.float64(0.7), np.float64(0.5640760748177662))
(10, 0, 35, 5, 0.9, 1.0, 0.6666666666666666, 1.0, 0.0, 0.3333333333333333, np.float64(0.8333333333333333), np.float64(0.7637626158259734))
(15, 0, 35, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, np.float64(1.0), np.float64(1.0))
```
注意只填值，不要填写`np.float64`，小数保留小数点后两位





参考这行的结果再生成两行
```
Attack & Defense & TP & FP & TN & FN & Accuracy & Precision & Recall & Specificity & FPR & FNR & AUC & MCC \\ \hline

\multirow{4}{*}{Original} & Foolsgold & 10 & 0 & 35 & 5 & 0.90 & 1.00 & 0.67 & 1.00 & 0.00 & 0.33 & 0.83 & 0.76 \\ \cline{2-14}
                      & FLTrust   & 6  & 0 & 35 & 9 & 0.82 & 1.00 & 0.40 & 1.00 & 0.00 & 0.60 & 0.70 & 0.56 \\ \cline{2-14}
                      & Flame     & 10 & 0 & 35 & 5  & 0.90 & 1.00 & 0.67 & 1.00 & 0.00 & 0.33 & 0.83 & 0.76 \\ \cline{2-14}
                      & SecFFT    & \textbf{15} & 0 & 35 & 0  & \textbf{1.00} & 1.00 & \textbf{1.00} & 1.00 & 0.00 & 0.00 & \textbf{1.00} & \textbf{1.00} \\ \hline
```
\multirow{4}{*}{Size} & Foolsgold &......
\multirow{4}{*}{Angle} & Foolsgold........
数据要比较类似





对于每一行，把最好的结果加粗（如果有多个一样好的结果则不加粗）





后面每一列的值都要由TP FP TN FN计算得到