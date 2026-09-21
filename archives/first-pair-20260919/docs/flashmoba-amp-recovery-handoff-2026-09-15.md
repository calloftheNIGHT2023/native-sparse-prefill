# 本机准备完成，下一步需要GPU

本轮没有连接云端或启动GPU任务。不能据此判断旧Pod已经停止计费。

准备位置：D:/ChatGPT/projects/native-sparse-prefill。

- 研究协议：docs/flashmoba-amp-recovery-protocol-2026-09-15.md。
- 输入：data/flashmoba-amp-recovery-v0，含train-calibration.npz、单独的report.npz、配置和数据选择来源。
- 入口：scripts/run_amp_recovery_stage.py；单轨迹/恢复入口：scripts/run_amp_recovery.py；事后分析：scripts/analyze_amp_recovery.py。
- CPU真实随机训练检查：results/flashmoba-amp-recovery-cpu-v1。此前v0结果保留；两个版本各13个小模型CPU更新，不是LLM训练。
- CPU模拟控制器检查：results/flashmoba-amp-recovery-controller-cpu-v0，覆盖选择锁、12条训练后才评测、失败停止和超时停止；日志中的mock结果完全是模拟数据，不能作为实验结果。
- 打包入口：scripts/package_amp_recovery.py；增量归档：exports/flashmoba-amp-recovery-input-v0.tar.gz，摘要见同名.json。

## 下一次接入

重启现有48GB卡或租单张48GB卡，然后提供最新SSH命令。先确认余额/实际费率和环境是否需重建；旧地址不保证有效。没有必要先租多张卡或80GB卡。

上传增量归档及清单；若是新Pod，额外上传data/flashmoba-qwen-precision-v0/model完整目录。按manifest逐文件核验，归档仅解压到预定项目根；保留旧结果，不覆盖冲突文件。原子断点必须校验其checkpoint-index.json中SHA后再恢复。

Python3.11和torch2.8.0+cu128环境按已保存freeze重建。freeze内flash_moba本地URL是原版；本轮必须改用归档中的候选wheel。可在新隔离环境安装候选wheel；不要让PYTHONPATH中的旧原版包遮住它。runner核验实际加载扩展的SHA，错误会拒绝启动。

不要把本机CPU检查结果当GPU预检通行证。完整控制器必须先跑三个方法的GPU预检，含新数据的数值检查、16K前反向及断点恢复对照；失败时停止，不调宽门槛。通过后按冻结协议运行9条学习率校准轨迹、3条第二种子复跑，再做6项最终报告评测。

若使用--preflight-only，只跑三项预检就结束；后续正式批次用新输出目录，现有控制器会再次预检。没有实现跳过预检的隐式捷径。

每个进程最多900秒，整个控制器最多7200秒。时间限制不关闭Pod，也不停止计费。批次结束后下载并校验原始结果、全部成功或失败日志、断点、时间轴；之后在RunPod点Stop。需要恢复时单轨迹的--resume使用已验证checkpoint，并提供对应--preflight、相同方法/种子/学习率及新--output目录。

当前确认：本机数据/源码/CPU检查已准备；GPU导入、GPU恢复、数值门槛、耗时和质量全部待新GPU实测。没有增加确认的原创贡献。
