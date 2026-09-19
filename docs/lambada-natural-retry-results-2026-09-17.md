# LAMBADA真实原文末词预测：固定512题

全词表逐token贪心一致性要求，整个末词都对才算正确；与四选一不同。最大147tokens，不是32K评测。基座全上下文/末32tokens能力门槛：{"full_accuracy": 0.4765625, "short32_accuracy": 0.21484375, "context_gain": 0.26171875, "passed": true}

|模型|上下文|正确/512|正确率|目标词困惑度|
|---|---|---:|---:|---:|
|base-ability|full|244|47.66%|10.8340|
|base-ability|short32|110|21.48%|515.6000|
|qkvo-train0-seed2026091660-restore0|full|249|48.63%|10.6196|
|qkvo-train0-seed2026091660-restore1|full|245|47.85%|10.7302|
|qkvo-train32-seed2026091660-restore0|full|266|51.95%|10.0482|
|qkvo-train32-seed2026091660-restore1|full|246|48.05%|11.1566|
|qkvo-train32-seed2026091661-restore0|full|260|50.78%|10.5130|
|qkvo-train32-seed2026091661-restore1|full|244|47.66%|11.2640|
|qkvo-train0-seed2026091661-restore0|full|248|48.44%|10.8326|
|qkvo-train0-seed2026091661-restore1|full|250|48.83%|10.8844|

预设自然词预测准确率非劣筛查：True。

[
  {
    "contrast": "primary",
    "metric": "accuracy",
    "difference": -0.0068359375,
    "conditional_item95ci": [
      -0.0244140625,
      0.0107421875
    ]
  },
  {
    "contrast": "primary",
    "metric": "target_word_nll",
    "difference": 0.044187837453137035,
    "conditional_item95ci": [
      0.01182345456463736,
      0.07721708360281807
    ]
  },
  {
    "contrast": "symmetric_restored",
    "metric": "accuracy",
    "difference": -0.0048828125,
    "conditional_item95ci": [
      -0.021484375,
      0.0107421875
    ]
  },
  {
    "contrast": "symmetric_restored",
    "metric": "target_word_nll",
    "difference": 0.03662968256685417,
    "conditional_item95ci": [
      0.006377588714030936,
      0.06703909807706622
    ]
  },
  {
    "contrast": "original",
    "metric": "accuracy",
    "difference": 0.0283203125,
    "conditional_item95ci": [
      0.005859375,
      0.05078125
    ]
  },
  {
    "contrast": "original",
    "metric": "target_word_nll",
    "difference": -0.04263429879938485,
    "conditional_item95ci": [
      -0.07676160956157219,
      -0.008105163795926277
    ]
  }
]

首次本项目固定子集评测，来源原始书籍分组不可见，逐题bootstrap可能低估同书相关性；仅两个已有训练种子，未知基座预训练接触。不是官方全测试集，也不是广泛长程能力或原创贡献证明。若基座门槛失败，不运行训练后对照，更不能将其记为稀疏失败。
