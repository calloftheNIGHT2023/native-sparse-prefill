# 本次A40环境与恢复位置

2026-09-14实测连接：Pod znbe5sxtm2n9tl，NVIDIA A40，46068MiB；Python3.11.10、PyTorch2.4.1+cu124。租卡价格与账户账单未核实。

本机SSH使用已有的专用密钥文件；不需要复制私钥到云端。RunPod给出的默认id_ed25519路径在本机不存在。本次成功命令：

```powershell
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -i REDACTED_SSH_KEY_PATH -p 22168 root@REDACTED_IPV4
```

IP/端口只适用于此次已验证的Pod，重新部署后可能改变。代理入口也可连接，但本次代理忽略远程命令，因此部署、传输与监测使用直接TCP SSH。

项目位于 /workspace/native-sparse-prefill/schedule-study-v0；五组顺序运行结果在其 results/schedule-screen-v0。运行时Python在 /opt/sparse-schedule-env/bin/python。/opt是容器内存储，停止/重建后不能假定环境还在；所有结果放在/workspace，随后回传本机。

旧 /workspace/native-sparse-prefill/.venv-cloud 存在缺损的包文件，pandas导入被不完整的pyarrow阻断。未删除旧环境；另建环境如下：

```bash
python -m venv --system-site-packages /opt/sparse-schedule-env
/opt/sparse-schedule-env/bin/pip install -r requirements-schedule-cloud.txt
```

这依赖同一已经包含CUDA PyTorch/torchvision的基础镜像；新机器先检查torch与GPU再复用。完整pip freeze和数值预检结果保存在日志目录。wandb在项目入口设为disabled，不发送实验到外部平台。

技术核验：本机43项CPU检查；云端5项稀疏模块检查；CUDA原始/修改前向反向一致性及未来梯度为0。另16次随机token技术更新不算科学实验。代码包v1 SHA256：a5c0ee09265a754b4513e55eb67d6d42a5a5d4f00addd9fb5c9ea0f1f697fa4d。

科学启动脚本 scripts/launch-sparse-schedule-cloud.sh 设置整批1800秒、每组600秒的进程上限。SIGTERM会记录失败和当前权重；检查点文件每轮更新，逐轮历史指标保留在events.jsonl。并非每轮都永久保留单独权重文件。

进程结束不会自动停止RunPod计费。确认本机备份后，可在RunPod控制台停止该Pod；不要将终止并删除存储与停止混同。
