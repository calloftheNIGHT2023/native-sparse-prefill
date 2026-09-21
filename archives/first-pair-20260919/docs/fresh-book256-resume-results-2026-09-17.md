# 256本新背景的长程检索确认：跨Pod续跑完成

保留原6组和第7组321条，续跑剩余2766条，并做2条保存预测重放；未重跑完整条件。固定旧问题模板与答案词，每本新书一个32K背景、四个反事实答案，以书配对。9组共9261条科研任务记录，2条诊断重放另记；无新训练。

基座门槛：{"short_correct": 4, "long_correct": 1021, "passed": true}

|模型|准确率|四个答案全对的书占比|短题正确数|
|---|---:|---:|---:|
|base-ability|99.71%|98.83%|4/4|
|train0-step64-seed2026091660|95.31%|88.67%|4/4|
|train32-step64-seed2026091660|99.12%|98.05%|4/4|
|train32-step64-seed2026091661|99.02%|96.88%|4/4|
|train0-step64-seed2026091661|97.17%|92.58%|4/4|
|train0-step128-seed2026091660|94.24%|86.72%|4/4|
|train32-step128-seed2026091660|99.02%|98.05%|4/4|
|train32-step128-seed2026091661|99.12%|97.27%|4/4|
|train0-step128-seed2026091661|96.78%|92.58%|4/4|

三项预先固定比较：[{"name": "sparse64restored_minus_dense64", "difference_pp": 2.83203125, "conditional98_333ci_pp": [1.46484375, 4.345703125], "noninferior": true, "per_seed_difference_pp": [3.80859375, 1.85546875]}, {"name": "sparse128restored_minus_dense128", "difference_pp": 3.564453125, "conditional98_333ci_pp": [2.05078125, 5.322265625], "noninferior": true, "per_seed_difference_pp": [4.78515625, 2.34375]}, {"name": "dense64_minus_sparse128restored", "difference_pp": -2.83203125, "conditional98_333ci_pp": [-4.345703125, -1.46484375], "noninferior": true, "per_seed_difference_pp": [-3.7109375, -1.953125]}]

新背景不等于新任务模板，也不等于自然QA；四个反事实答案按同书聚类，两个旧训练种子先平均，三项使用98.333%区间。5pp界限固定。原有自然文本质量与计时来自旧审计，本轮只独立确认背景转移，不把旧结果当新证据。基座若失败，停止后续条件而不改变题。已知QK-Restore不作为原创。
