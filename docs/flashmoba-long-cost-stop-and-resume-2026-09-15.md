# 暂停租卡与恢复记录

本轮5个GPU任务于2026-09-15 17:49:31 UTC全部完成，无后台实验队列，0次优化器更新。17:51 UTC实测GPU利用率0%、显存2MiB。

## 已保存

- 原始日志、配置、逐次计时、数值检查、源码、数据清单：exports/flashmoba-long-cost-cloud-stage-v0.tar.gz。SHA256 da0cef4ce30b94ccd1046cc6fc3dc594b69fe10cf5ea5a097d79dc7355e3bc76，64个清单文件，已下载并逐一核验。
- Qwen2.5-0.5B完整8个模型文件已在本机按原始manifest核验，其中model.safetensors为988097824字节，SHA256 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342。
- 原版与候选wheel已在本机核验；pip freeze和Python/驱动版本已落盘到provenance/flashmoba-long-cost-recovery*。
- 本轮无新训练权重。之前质量试验的adapter/checkpoint仍在此前已校验的归档中；它们没有优化器状态，不支持声称逐步精确续训。

## 当前计费状态

Pod为4rvqz1eqrmdl0w。可控制浏览器访问RunPod后显示注册/登录页，无法查看该Pod或执行Stop。因此本记录**没有确认停止计费**，SSH空闲也不代表停止计费。用户在已登录控制台选择此Pod并点击Stop，等待显示Stopped即可。不要用Terminate替代Stop。

RunPod官方说明：Stop后GPU费用停止，保留卷盘仍收费，容器盘会清除。官方链接：https://docs.runpod.io/pods/manage-pods 。实际账户价格、余额及卷盘类型需以控制台为准。/workspace实测为独立fuse挂载，但这一挂载事实不能证明账户的全部持久化策略或配额。

## 恢复

本轮Python环境位于/opt/native-sparse-flashmoba-env-v2，按临时环境处理，**没有整目录备份**。停止后可能需要重建；Python3.11、torch2.8.0+cu128以及记录的依赖版本已保存，原版与候选wheel分别保留。冻结清单内flash_moba本地URL指向原版wheel，不能直接安装后就声称加载了候选；必须恢复候选隔离源码/PYTHONPATH并核对实际加载.so的SHA。流程详见flashmoba-backward-cloud-restore-2026-09-15.md。

新Pod恢复时先核验代码/模型/数据/扩展，再做短数值检查；不要盲目重跑本轮固定输出目录，也不要复用过时SSH地址。本机分析、相关工作核查、实验协议与代码准备不需要运行租用GPU。下一次协议就绪后集中开机运行，完成即备份并Stop。
