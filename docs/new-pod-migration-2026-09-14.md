# 新A40迁移验收与续训入口

新Pod：gcv07u2ggin9lk；主机af3f31cbe85d；直接SSH REDACTED_IPV4:22030。用户给出的代理连接已验证，专用私钥仅留在本机，不复制到云端。GPU为NVIDIA A40，46068MiB。IP和端口仅适用于本次实例。

用户迁移了旧项目根目录，但最新schedule-study-v0目录为空。已从本机补传当前阶段恢复包router-migration-v1.tar.gz，67405816字节、506文件，解压155185166字节。云端与本机归档SHA256均为41f74e6ff2691cfca7d6ca31efe9df9cfe79427e5e413d4304aae16cee611e68；逐文件SHA核验通过。

当前工作目录：/workspace/native-sparse-prefill/router-study-20260914-v1。包含全部当前代码、配置、上游固定版本、文献、三轮相关实验的数据/权重/日志；已关闭的70M大体积旧档案仍保留本机和原项目位置，不是此次继续MQAR所需依赖。原结果不覆盖。

运行环境：/opt/sparse-router-env/bin/python，Python3.11.10、PyTorch2.4.1+cu124、torchvision0.19.1+cu124、NumPy1.26.3。沿用已检查的镜像Torch，新建system-site-packages venv并安装requirements-router-portable.txt中的科研依赖。没有使用旧损坏.venv-cloud。便携纯环境安装替代脚本scripts/setup-router-portable.sh使用[PyTorch官方旧版安装源](https://pytorch.org/get-started/previous-versions/)。/opt环境在再次换容器后仍需重建。

全镜像pip check有一条既存GUI依赖警告：pygobject3.42.1缺pycairo；本实验不导入该组件。已保存完整输出，不把它说成全环境依赖零问题；本实验所需导入、模型前向/反向与权重重算均通过。

18:17:04 UTC完成新GPU迁移核验：三组40轮checkpoint的15360个旧测试答案与旧机完全一致；主干28份及索引器8份优化器参数状态、scheduler_epoch40及主学习率0.0006545084971874737均恢复。CUDA随机权重前向/梯度对照和辅助梯度隔离检查通过，前向最大误差3.5762786865234375e-7。

两项CPU恢复测试在本机和新云端均通过。最初严格字典比较因Torch2.10新增调度字段失败，多线程gather梯度归约也使逐位测试有微小差异；按旧字段逐项检查，并将确定性单步测试限定单线程后，保存/恢复的下一步权重与指标完全一致。原失败日志保留。准备阶段一次性CPU副本上总共12对技术优化器更新（本机9、新机CPU3），科学更新0；GPU迁移/梯度检查0优化更新。跨硬件/并行计算仍不承诺逐位重现完整训练。

18:18:10 UTC启动results/joint-token-router-epoch80-v0，按docs/joint-token-continuation-plan-2026-09-14.md将三组从40轮续到80轮。完整旧结果results/joint-token-router-v0保留；新每组日志和检查点独立，模型从global_step12520接续。入口src/run_router_epoch80_batch.py调用src/resume_joint_token_router.py，恢复两套优化器、调度与RNG。新题只在三组完成后评测。

总500美元、首阶段30美元额度未增加；新机实际费率及累计账单未知，Python进程结束不等于Pod停止计费。

18:24 UTC第一组完成80轮并保存后，末尾日志重复utc参数导致批次暂停。修复只涉及完成事件写法；补丁包router-epoch80-log-fix-v1.tar.gz的两端SHA为247661064969d729c7c9f93a61d702775c94c0c11ec511fef9e6a8d1f66c186f。18:26 UTC使用src/recover_router_epoch80_batch.py核验并跳过已完成第一组，继续其余两组，没有重复科学更新。原始恢复包与MIGRATION_MANIFEST保持不变，补丁后的两个源码文件由新结果中的recovery-source单独存证，不能将原始包manifest视为补丁后活跃目录的全量清单。原错误日志及诊断快照保留，最终状态见80轮报告。

setup-router-portable.sh是为再次迁移准备的纯环境安装替代入口，本次实际采用上述已核验的system-site-packages环境；未把替代安装脚本说成已在Linux完整执行通过。
