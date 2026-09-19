# 校正估计器的性质：支持什么，不支持什么

以下为有限总体抽样、凸性与混合分布的直接推导，用来约束方法解释。不是新定理；实际语言模型收益另由实验决定。

固定一个query和一个头，并条件于本步已选择的集合S。令已选token（含尾块）的exp-logit和为Z_S，N个未选完整块各自的exp-logit和为X_1,...,X_N。随机不放回取m个未选块。

`Z_hat = Z_S + (N/m) * sum_{b in sample} X_b`。

若抽样条件于S均匀且独立于未观测块数值，则E[Z_hat|S]=Z。对0<m<N，

`Var(Z_hat|S) = (N^2/m) * (1-m/N) * sample_variance(X_1,...,X_N)`，

其中总体的sample_variance以N-1为分母。m=N时误差为0；N=0时直接返回保留质量1。数值实现减去公共最大值不改变各比率。

保留质量m_h=Z_S/Z，估计为m_hat_h=Z_S/Z_hat。因为取倒数是凸函数，Jensen给出E[m_hat_h|S]>=m_h：分母无偏不意味着保留质量无偏。未选exp-logit和越重尾，少量均匀抽样越可能高估保留质量。这解释为什么两个抽样块未必足以逼近精确对照，不能宣称无偏KL。

由于Z_hat>=Z_S>0，逐次有：

`|m_hat_h-m_h| = Z_S*|Z_hat-Z|/(Z_hat*Z) <= |Z_hat-Z|/Z`。

结合Cauchy-Schwarz：`E|m_hat_h-m_h| <= sqrt(Var(Z_hat))/Z`。该界可能很松，不能当作实测误差或高概率保证。

对于多个头，各头在S内的条件分布记为P_h，稠密条件混合权重w_h=m_h/sum m，估计权重v_h=m_hat_h/sum m_hat。凸混合收缩给出：

`TV(sum_h w_h P_h, sum_h v_h P_h) <= TV(w,v) <= sum_h |m_hat_h-m_h| / sum_h m_h`。

这只直接约束块max pooling之前的token分布。令其在完整块上的max-pool质量总和为A>0，max-pool再归一化后的TV可用`2*TV(token)/A`界住；因为A可能很小，该界不保证实际误差足够小。无完整可见块的query在实现中不参与KL。

相应输出/词预测收益不能从这些不等式推出。这里没有分析索引器共享参数、TopK离散切换、变化中的底座或联合训练的收敛性。

`src/verify_head_mixture_math.py`枚举4选2的全部6个子集，核验均值、方差、比率偏差与误差界，并在100个正概率混合上检查收缩界。记录在 `results/head-mixture-math-v0.json`。这类数值核验辅助排错，不替代上述推导，也不是训练实验。
