# QK-Restore 在第二批背景和答案词上的开发验证

这是已知恢复操作的开发验证；本批题此前已暴露，不能称为独立确认。四个原有128步检查点，无新增训练；所有模型统一密集评测。

|训练K|种子|恢复前正确/128|恢复后正确/128|恢复后短题/4|
|---|---:|---:|---:|---:|
|0|2026091660|102|105|4|
|32|2026091660|96|118|4|
|32|2026091661|104|116|4|
|0|2026091661|118|111|4|

主要对照：恢复后的稀疏训练模型，相对于未恢复的密集训练模型。开发筛查通过=True。

{
  "restored_sparse_vs_original_dense": {
    "difference_pp": 5.46875,
    "paired_background_95ci_pp": [
      -2.34375,
      13.28125
    ]
  },
  "restored_sparse_vs_restored_dense": {
    "difference_pp": 7.03125,
    "paired_background_95ci_pp": [
      -1.953125,
      16.015625
    ]
  },
  "sparse_restore_effect": {
    "difference_pp": 13.28125,
    "paired_background_95ci_pp": [
      5.078125,
      22.265625
    ]
  },
  "dense_restore_effect": {
    "difference_pp": -1.5625,
    "paired_background_95ci_pp": [
      -3.90625,
      0.78125
    ]
  },
  "restoration_interaction": {
    "difference_pp": 14.84375,
    "paired_background_95ci_pp": [
      5.859375,
      24.21875
    ]
  }
}

区间按32个背景配对抽样，条件于两个训练种子和同一个人工任务模板；非独立泛化证据。原检查点未改写，48个Q/K LoRA B仅在内存清零，144个其他张量逐一检查不变。校准16次，任务预测532次。实际同质量训练总成本与外部自然任务尚待验证，已有QK-Restore不是原创贡献。
