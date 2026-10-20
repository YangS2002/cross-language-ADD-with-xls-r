# SLS参照层选择模型实现与验证

状态：模型已实现并上传远端；未接入新训练器，未启动训练，无目标EER、无有效性/种子稳定性结论，未Git提交。

独立文件tools/sls_reference_model.py，未覆盖原有src/model.py、train.py。ReferenceRoutedSLS提供self_reference/pretrained_reference；FullWidthSLS使用完整24层、每帧1024维，门控输入时间均值，经共享1024→1线性+sigmoid获得层权重。该线性仅输出权重，不降维待分类特征。

分类值仅来自任务模型。两臂门控输入都detach，但门控参数可训练。没有参考/任务特征平均、拼接、差分，没有192维投影、双头或关系保持损失。保留SLS原BN2d、SELU、3×3 stride3 max-pool、22847→1024→2分类器与输出SELU/logsoftmax。拒绝非201帧输入，不更改data.pad来适配网络。

任务XLS-R卷积及未使用预训练头冻结，24层Transformer、post_extract_proj、layer_norm可训练；任务与参考SSL保持eval关闭随机层丢弃/dropout，但任务保留梯度。原始教师所有参数冻结，no_grad计算描述，并保存恢复NumPy RNG避免扰动后端随机序列。支持传入预计算[B,24,1024]描述，但缓存构建与协议签名管理尚未实现。

trainable_payload仅保存完整SLS后端（含BN buffers）和任务骨干requires_grad参数；不保存教师或未更新的原始模块，也不包含Adam状态。恢复需首先加载原始初始化权重，并验证预训练SHA256、参照来源和更新参数集合。该接口不等于训练恢复文件已经实现或解决全部空间预算；更新骨干及SLS的23M以上FC仍有真实存储成本。

## 自检

新增tools/test_sls_reference_model.py。通过SSH30461，用/opt/conda/envs/SLS/bin/python -m tools.test_sls_reference_model --real-checkpoint models/pretrained/xlsr2_300m.pt运行。

- SYNTHETIC_OK：聚合[B,201,1024]、门控[B,24]、池化[B,1,67,341]、概率归一化、任务/门控梯度、教师无梯度、门控输入detach、在线与传入描述一致、初始两种参照等价、轻量payload键集合、错误原始哈希及输入帧数拒绝。
- REAL_SSL_OK：真实原始XLS-R，batch1随机波形64600，BF16，一次前向/反向；输出[1,2]，池化[1,1,67,341]，NLL0.8198463，reserved4.0957GiB。任务第1及第24层梯度非零且有限；冻结卷积/教师无梯度。
- 没有optimizer step、模型checkpoint或实验训练。显存数字仅batch1无Adam状态工程自检，不代表batch16训练峰值。随机波形损失不是检测指标。

后续训练/评估接口应使用log_probs配合NLL损失或等效logits CE，避免把log_probs误当未经说明的其他分数；评估可用真类log概率，固定全域同一评分规则。构建缓存及所有权重仅允许持久数据盘。当前磁盘余量不足既定两臂训练保存峰值，因此本轮不启动。
