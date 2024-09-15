假设恶意客户端使用限制攻击大小的方式进行隐蔽。请你写一个函数，给定一个gradients列表，列表中有50个客户端。其中前15个为恶意客户端。
你需要对这15个恶意客户端进行处理，限制梯度更新的大小，并返回限制后的结果。





很棒，类似`limit_attack_size`写一个`limit_attack_direction`函数，恶意攻击者限制自己的梯度角度与正常客户的梯度较为类似






对于15个恶意客户端，每个客户端随机选择一个良性客户端，并向其方向偏移




我有一个数组`[np.int64(14), np.int64(2), np.int64(13), np.int64(12), np.int64(6), np.int64(1), np.int64(9), np.int64(4), np.int64(5), np.int64(0)]`，如何转为[int, int]



把这个表格翻译成英文
```
\begin{table*}[h!]
\small
\centering
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{攻击方式} & \textbf{防御方式} & \textbf{FP} & \textbf{TP} & \textbf{FN} & \textbf{TN} & \textbf{准确率} & \textbf{精确率} & \textbf{召回率} & \textbf{特异度} & \textbf{误报率} & \textbf{假阴率} & \textbf{AUC} & \textbf{MCC} \\ \hline

\multirow{4}{*}{NERO} &Foolsgold & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & FLTrust & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & Flame & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \cline{2-14}
                       & SecFFT & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ \hline
\end{tabular}
\caption{不同攻击方式和防御方式下的分类指标结果}
\end{table*}
```




填写这个表格
```
\begin{table*}[h!]
\small
\centering
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Attack Method} & \textbf{Defense Method} & \textbf{FP} & \textbf{TP} & \textbf{FN} & \textbf{TN} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{FPR} & \textbf{FNR} & \textbf{AUC} & \textbf{MCC} \\ \hline

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
(10, 0, 34, 6, 0.88, 1.0, 0.625, 1.0, 0.0, 0.375, np.float64(0.8125), np.float64(0.7288689868556626))
(6, 0, 34, 10, 0.8, 1.0, 0.375, 1.0, 0.0, 0.625, np.float64(0.6875), np.float64(0.5383054219239551))
(10, 0, 34, 6, 0.88, 1.0, 0.625, 1.0, 0.0, 0.375, np.float64(0.8125), np.float64(0.7288689868556626))
(15, 0, 34, 1, 0.98, 1.0, 0.9375, 1.0, 0.0, 0.0625, np.float64(0.96875), np.float64(0.9543135154205278))
```
注意只填值，不要填写`np.float64`





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