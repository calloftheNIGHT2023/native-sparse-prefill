# 本轮云端环境与迁移恢复

当前Pod 4rvqz1eqrmdl0w，直连195.26.233.77:27747。卡为RTX 6000 Ada，实际PyTorch版本2.8.0+cu128，Python3.11。主项目在/workspace/native-sparse-prefill/efficiency-20260915-v1。

本次迁移后环境目录仍在，但torch共享库、部分C++源码、CUTLASS头文件、编译缓存与egg-info不完整。因此“目录复制了”不能作为环境可用的证明。本次原版导入、模型8个文件SHA和12项标准检查已实际通过。

## 存储安排

- /workspace保存输入、结果、每次失败日志、补丁和安装包。
- /opt/native-sparse-flashmoba-env-v2保存可重建Python环境。本Pod的容器盘20GB，安装后约9GB使用；停止/销毁容器可能失去这个环境。
- 持久化盘曾达到配额，df展示的是后端共享文件系统容量，不能当作本Pod配额。不要仅凭df显示空间充足判断安装可以继续。
- 旧.venv-flashmoba-v0和.venv-cloud保留；它们不再是本轮已验证运行时。

## 再次恢复的最短路径

先恢复数据与代码，并按照归档manifest逐文件核验。建立新Python3.11环境，安装torch==2.8.0官方cu128版本，以及pip-freeze中记录的依赖；在确认显存架构/ABI兼容后安装已验证wheel。安装过程用日志记录，不复制一个缺少库文件的旧venv继续使用。

原版wheel：exports/flashmoba-wheels-v0/flash_moba-2.0.0-cp311-cp311-linux_x86_64.whl，SHA256 ac2ea167afe782dd44ddd27f13091d11c7fba33721ec6ee40339f6b4df97aa18。

候选补丁已通过本轮长序列、12项独立正确性检查和16步整合诊断。候选wheel另存exports/flashmoba-barrier-wheels-v0/及manifest，不覆盖原版包。候选验证使用PYTHONPATH指向隔离副本，同时保留原版安装在site-packages中。每份结果记录实际加载扩展路径和SHA，防止两组误用同一扩展。

不需要编译时不要安装完整工具链。需要重编译时，使用固定官方commit、CUTLASS commit和CUDA12.9.1组件SHA，并根据新机器实际CPU配额、内存设置MAX_JOBS。不要在已有输出目录盲目重复restore脚本；本轮v1–v5是逐次恢复/续编证据，包含依赖特定现场的路径，不是可任意重复运行的通用安装器。

## 当前实验的依赖顺序

verify_long_backward_candidate.py → verify_flashmoba_official_v1.py → benchmark_backward_barrier.py → run_backward_barrier_training.py。

最后一项内置前两项的通过检查，并核对实际加载的候选扩展SHA。训练为4条同种子、同初始化、同数据的4步诊断，共16个优化更新。它验证整合和重复性，不用于主张泛化性能或模型质量提升。

候选扩展SHA256：72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c。候选wheel内扩展已与实际验证的.so核对一致，迁移到新机器仍需重新过门槛。
