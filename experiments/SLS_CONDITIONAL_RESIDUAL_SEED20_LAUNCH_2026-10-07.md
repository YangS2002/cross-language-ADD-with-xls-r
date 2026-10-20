# SLS条件残差：实现、自检与正式启动记录

## 实际状态

2026-10-07，经用户明确要求立即执行，新增独立模型与训练器，已上传并正式运行。SSH端口30461；训练PID13639。启动30秒后确认进程Rl、GPU利用率85%、占用4022MiB，status.json为training / warmup / epoch1。尚无正式训练EER，不能据此认定方法有效。

启动2分11秒后的再次核验：进程持续运行，预热第1轮已完成全部25380样本、1587步，98.254秒，train_loss=0.3069453，accuracy=0.876911，reserved=3.4707GiB；已进入预热第2轮。该耗时仅适用于冻结预热，不能外推为Transformer微调每轮耗时。

远端仓库：/root/rivermind-data/cross-language-ADD-with-xls-r。

命令：`/opt/conda/envs/SLS/bin/python -u -m tools.train_sls_conditional_residual --config experiments/configs/sls_conditional_residual_seed20.yaml`。

日志：experiments/logs/sls_conditional_residual_seed20.log；输出：models/SLS_conditional_residual_seed20/。全部数据、权重、日志均在持久盘，未删除旧结果、未覆盖src/model.py或train.py、未提交Git。

## 新实现及固定协议

- tools/sls_conditional_residual_model.py：SLS原24层门控梯度路径、完整1024维聚合、原BN/SELU和3×3 max pool后端。新增的参照条件残差在池化前；不再是旧方案仅用参照改变层权重。
- tools/train_sls_conditional_residual.py：sls / self_residual / pretrained_residual三个独立对照，固定seed20，共同冻结预热3轮，之后每臂顺序训练至总epoch10。
- 原始参照固定第9–16层，实际参考只计算到16层；自身参照采用任务模型同一层段detach。
- R为1024→1024线性，仅源域真实80%文件ID SHA256划分训练，目标SLS表征detach；20%源域真实不参与R拟合，只用于记录重建误差。R独立AdamW lr1e-4，分类不更新R。
- U为1024→1024零初始化线性；真实实际修正每帧限于主表征范数0.1倍。分类更新任务SSL、原SLS后端和U，lr1e-6。
- 原data.pad64600、batch16、BF16、原始类别比例、weightedNLL真实0.9/伪造0.1；无重采样、噪声增强、RawBoost或样本顺序研究。
- 第4/8/10轮以最低源域分类训练损失权重评估CH/DE/UK/ES/ITW。R重建loss不用于选分类checkpoint；不开目标域训练。
- C在epoch4增加一次固定correction_off诊断，结果独立存放，不用于选择评分路径，不冒充独立SLS对照。

## 真实数据自检记录

使用8真实+8伪造音频，每臂3次真实优化器更新。三个零修正初始输出一致，初始loss均0.8311399221；更新后loss有限并下降。任务首层/末层有梯度；层门控有梯度；分类不更新预测器；重建不更新SLS/任务骨干/U；冻结原始参考无梯度；教师实际执行层号严格1–16；修正范数逐步检查。

| 臂 | 更新参数字节数 | 最终自检显存reserved GiB |
|---|---:|---:|
| sls | 1338507796 | 10.1230 |
| self_residual | 1346904596 | 10.2129 |
| pretrained_residual | 1346904596 | 18.7305 |

第一次原始参照臂reserved为20.0684GiB，未OOM但略超过预登记20GiB工程限制，自检未通过、未启动正式训练。仅调整前向计算顺序为参考先行，再执行保留反向激活的任务路，未改变模型公式、初始化、超参或数据；重新完成三臂全部自检后通过。自检JSON已同步本地experiments/sls_conditional_residual_seed20_self_test.json。

## 存储与执行预算修订

持久盘剩余约9.7GiB。自检核算磁盘写入峰值预算6059179755字节，约5.64GiB，含三个best、额外原子替换临时文件和安全余量。

不再保存逐轮权重或大Adam恢复文件，仅共享warmup后端与每臂最低分类损失完整推理参数。评估时CPU内存暂存当前训练参数，加载best评估后恢复当前参数和RNG，Adam状态不改变；CPU内存检查充足。代价是异常退出后无法精确恢复Adam，本实验禁止自动重放或将重新初始化优化器冒充续训。

因此采用三个臂顺序各完成至10轮，不是所有臂先到epoch4的阶段式调度；C的4/8/10节点分别比较已有A/B对应节点。C相对A与B均触发预登记明显失败线时停止；未触发停止线不等于成功。

## 监控、分析和后续边界

原有监控rp已更新为本实验，频率改为每2小时，正常无变化保持安静，仅阶段评估、异常、结束或需处理时通知；完整结束或异常不可继续即暂停。这里只采用OpenAI Docs技能核对定时任务设置，未改变实验设计。

同步小型CSV/JSON/日志；报告每个域、四跨语均值/最差域、CH/DE联合、ITW及论文外部baseline差值。外部MLSSL表3 SLS为CH12.26/DE19.71/UK21.13/ES9.69/ITW7.46，DuaBiMamba为17.32/23.57/11.89/15.83/6.71。当前SLS是固定项目训练协议的结构对照，不冒充论文训练完全复现。单种子候选有价值后才考虑1001/7，不自动追加。本方法尚不能声称内容已解耦，不能把自检损失下降或开始运行视为跨语成功。
