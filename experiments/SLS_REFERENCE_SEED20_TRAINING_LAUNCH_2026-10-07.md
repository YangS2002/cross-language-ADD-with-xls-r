# SLS参照层选择seed20：清理、自检与正式启动

2026-10-07，用户要求清理失败训练并重新训练。已启动正式运行，SSH30461，PID5921，日志experiments/logs/sls_reference_seed20.log，输出models/SLS_reference_seed20。确认预热epoch1已真实处理3200条源域音频，累计训练损失0.5993181，GPU90%利用率；没有目标域结果。监控沿用原线程heartbeat ID rp，已重命名更新为SLS实验，每小时检查，未变化时安静。

## 清理范围

核对后删除E3_dual_xlsr_reference下self_reference/frozen_reference各seed20/1001/7的六个best_train_loss.pth，每个1277792688字节，总7666756128字节（约7.14GiB）。逐个核验路径与非符号链接，未递归删除目录。它们是此前已判定失败的双路交叉注意力方法，不是原始权重。无本地大型权重备份，因此不可恢复；小型指标和报告保留。RP八个权重此前已清理。M3/M4参考权重、其他小型历史文件、源数据、原始XLS-R及原有未提交代码未动。

清理后持久盘free13081972736字节（约12.18GiB）。所有新日志、音频、训练权重与恢复状态位于/root/rivermind-data，不使用系统盘存数据。

## 最终训练设置与设计修订

- 配置experiments/configs/sls_reference_seed20.yaml，新训练器tools/train_sls_reference.py；不替换原train.py。
- 模型为完整24层1024维SLS，门控来源self_reference或pretrained_reference，两臂门控输入均detach；后端BN/SELU/max-pool/原尺寸FC，单分类头，没有192维投影或关系loss。
- 原data.pad 64600、无重采样、无RawBoost/加噪、无类别重采样。源域25380音频（真2580/假22800），weightedNLL假0.1/真0.9。保留全部样本，不drop最后批次。
- batch16，BF16，FP32参数/AdamW状态；卷积冻结，24个Transformer与投影/层norm参与任务微调；SSL dropout/layerdrop关闭。梯度裁剪5。
- **正式运行前**将head LR从设计草案1e-4改为1e-6，SSL也1e-6。理由是换回原SLS大FC后，混合源域批次工程测试中1e-4造成损失上升；采用作者main.py默认1e-6，不基于目标域调参。AdamW仍沿用配对训练，不宣称完整复现论文Adam等设置。作者来源：https://github.com/QiShanZhang/SLSforASVspoof-2021-DF/blob/main/main.py 。
- 共享冻结骨干预热3轮，用固定epoch3状态分支、重置优化器，两个分支同起点/RNG。每臂再训练7轮，总10轮，最佳按微调阶段最低源域训练损失，而非目标域选权重。
- 为空间峰值调整执行顺序：先self_reference完整到10，再pretrained_reference；每臂各在总epoch4/8/10完整评估五域。此顺序避免两份Adam恢复文件并存，不改变每臂更新规则和节点比较。明确不是三个节点交错训练；参照臂可在第4轮发现大幅劣化时停止，但self对照届时已完整运行。
- 预登记止损：至少两个跨语域比同节点self高5个百分点且四域均值高3个百分点；或epoch8+跨语均值无收益且ITW高2个百分点。全部中间结果保留，探索性目标域停止不是盲测。
- 评估输出单头真类log probability，原SLS评分方式；不要求不存在的A/B结果。五域样本路径核验全通过；论文baseline是MLSSL表3中的SLS与DuaBiMamba已发表结果，不把本次self-reference称为论文baseline。

## 工程压力测试

最终设置真8/假8混合源域批次，各臂3次完整前向/反向/Adam更新，测试权重丢弃，不当作实验训练结果：

| 臂 | 三步NLL | 峰值reserved GiB |
|---|---|---:|
| self_reference | 0.831140 → 0.402828 → 0.334899 | 10.0840 |
| pretrained_reference | 0.831140 → 0.404850 → 0.337094 | 12.7793 |

两臂初始输出一致，任务首尾层梯度非零，冻结教师无梯度，有限损失/梯度，自检最终SELF_TEST_OK。这只证明工程可运行，不证明检测效果。

更新参数实测1338507796字节；估计两臂best、一份resume及原子替换、新预热与安全余量峰值12048771854字节，比free13081972736小约1.03GB。模型best不保存冻结教师/未更新模块，但约1.34GB更新权重与Adam状态不能省略成只有小头。维持单个active_resume.pth；某臂完成最终评估、写入completed标记后，才删除本运行该臂的临时resume，保留best。warmup仅后端+RNG，不保存冻结骨干或不需要的预热Adam。

评估时加载截至节点best，评估后恢复最后训练状态与RNG，再继续训练；不会用best状态替代最后状态接着优化。每次写入仍动态核验临时文件空间，不降低安全缓冲。异常只记录报告，不自动重跑。

## 记录与后续边界

self_test.json、inventory.json、experiment.json已同步本地。后者保存配置/新模型/训练器/源标签/原初始化哈希。每轮记录损失、源域准确率、真实/伪造覆盖、步数、梯度范数、显存、耗时、最佳标记、磁盘余量；每节点评估记录全部CH/DE/UK/ES/ITW EER、权重epoch、样本数与时间。

当前仅固定seed20筛查，不更换幸运种子；有希望才考虑1001/7，单种子不能证明稳定性。所有大型权重留远端，无Git提交，不关机。报告随真实阶段结果更新，不将启动等同完成。

启动后核验：共同预热epoch1已完成，97.84秒，源域训练NLL0.3069453、准确率87.6911%，覆盖25380条/1587步，峰值reserved3.0762GiB；epoch2已开始。梯度范数字段为裁剪前值。准确率和源域损失不能用于断言跨语言效果，五域EER尚未产生。
