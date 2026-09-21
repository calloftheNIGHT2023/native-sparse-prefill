# 同学习率与原始底座：任务表现诊断

这是对旧96道题的事后探索，只有一个种子2026091560。D为密集注意力，S为K16；训练后各使用自己的训练注意力。所有训练断点均256步，底座参照完全移除LoRA适配器，不做优化更新。

|输入|原始底座/D|原始底座/S|D训 LR0.0003|S训 LR0.0003|D训 LR0.001|S训 LR0.001|
|---|---:|---:|---:|---:|---:|---:|
|short|58.33% (56/96)|60.42% (58/96)|64.58% (62/96)|64.58% (62/96)|62.50% (60/96)|67.71% (65/96)|
|no_context|43.75% (42/96)|44.79% (43/96)|41.67% (40/96)|45.83% (44/96)|42.71% (41/96)|42.71% (41/96)|
|long8192|44.79% (43/96)|27.08% (26/96)|54.17% (52/96)|39.58% (38/96)|46.88% (45/96)|41.67% (40/96)|
|long16384|42.71% (41/96)|26.04% (25/96)|45.83% (44/96)|38.54% (37/96)|45.83% (44/96)|34.38% (33/96)|

## 完整配对比较

|输入|A−B|准确率差pp|描述性95%区间pp|改对/新错|
|---|---|---:|---|---:|
|short|sparse_low minus dense_low|+0.00|[-7.29, +7.29]|6/6|
|short|sparse_high minus dense_high|+5.21|[-3.12, +13.54]|10/5|
|short|sparse_high minus dense_low|+3.12|[-5.21, +11.46]|9/6|
|short|dense_low minus base_dense|+6.25|[+1.04, +12.50]|7/1|
|short|dense_high minus base_dense|+4.17|[-2.08, +10.42]|7/3|
|short|sparse_low minus base_sparse|+4.17|[-2.08, +10.42]|7/3|
|short|sparse_high minus base_sparse|+7.29|[-1.04, +15.62]|12/5|
|short|base_sparse minus base_dense|+2.08|[+0.00, +5.21]|2/0|
|no_context|sparse_low minus dense_low|+4.17|[-2.08, +10.42]|7/3|
|no_context|sparse_high minus dense_high|+0.00|[-8.33, +8.33]|9/9|
|no_context|sparse_high minus dense_low|+1.04|[-6.25, +8.33]|7/6|
|no_context|dense_low minus base_dense|-2.08|[-8.33, +4.17]|4/6|
|no_context|dense_high minus base_dense|-1.04|[-9.38, +7.29]|7/8|
|no_context|sparse_low minus base_sparse|+1.04|[-5.21, +7.29]|5/4|
|no_context|sparse_high minus base_sparse|-2.08|[-10.42, +6.25]|7/9|
|no_context|base_sparse minus base_dense|+1.04|[+0.00, +3.12]|1/0|
|long8192|sparse_low minus dense_low|-14.58|[-23.96, -5.21]|5/19|
|long8192|sparse_high minus dense_high|-5.21|[-17.71, +7.29]|17/22|
|long8192|sparse_high minus dense_low|-12.50|[-23.96, -2.08]|9/21|
|long8192|dense_low minus base_dense|+9.38|[+1.04, +17.71]|13/4|
|long8192|dense_high minus base_dense|+2.08|[-6.25, +10.42]|9/7|
|long8192|sparse_low minus base_sparse|+12.50|[+1.04, +23.96]|23/11|
|long8192|sparse_high minus base_sparse|+14.58|[+2.08, +27.08]|28/14|
|long8192|base_sparse minus base_dense|-17.71|[-30.21, -5.21]|12/29|
|long16384|sparse_low minus dense_low|-7.29|[-16.67, +2.08]|8/15|
|long16384|sparse_high minus dense_high|-11.46|[-23.96, +1.04]|13/24|
|long16384|sparse_high minus dense_low|-11.46|[-22.92, +0.00]|10/21|
|long16384|dense_low minus base_dense|+3.12|[-6.25, +12.50]|11/8|
|long16384|dense_high minus base_dense|+3.12|[-6.25, +12.50]|12/9|
|long16384|sparse_low minus base_sparse|+12.50|[-2.08, +26.04]|31/19|
|long16384|sparse_high minus base_sparse|+8.33|[-6.25, +22.92]|28/20|
|long16384|base_sparse minus base_dense|-16.67|[-30.21, -3.12]|16/32|

## 解释边界

原最佳校准配置比较仍是sparse_high对dense_low；不能因为dense_high较差就替换密集基线来宣称质量相同。相同学习率对照帮助检查配置混杂，但不是内部机制的唯一解释。原始底座参照能观察本轮适配前后的任务变化；不代表原生稀疏预训练。

区间按96题配对重采样，事后探索且未经多重比较校正，未显著不等于等价。没有新留出确认，不把最高得分直接当最终方案。长文本使用明确标记TARGET的自建RACE形式。底座移除了LoRA模块，因此其前向时间不能直接解释为训练方法速度差。旧约9.2%训练节时来自原训练轨迹，不能转移给未训练的新方案。

全部6条件通过原校准NLL重放、数据/模型/源代码/断点哈希核验；dense_low和sparse_high共768条预测与旧记录逐项logit核对。总2304条计分预测，18次热身、24个校准窗口，0优化更新。

## UTC时间轴

|任务|开始|结束|退出码|
|---|---|---|---:|
|dense_low|2026-09-15T22:42:43.553622+00:00|2026-09-15T22:43:38.070936+00:00|0|
|sparse_low|2026-09-15T22:43:38.076011+00:00|2026-09-15T22:44:37.540532+00:00|0|
|dense_high|2026-09-15T22:44:37.544352+00:00|2026-09-15T22:45:29.916604+00:00|0|
|sparse_high|2026-09-15T22:45:29.920054+00:00|2026-09-15T22:46:32.164341+00:00|0|
|base_dense|2026-09-15T22:46:32.171192+00:00|2026-09-15T22:47:20.970077+00:00|0|
|base_sparse|2026-09-15T22:47:20.974222+00:00|2026-09-15T22:48:17.142749+00:00|0|
