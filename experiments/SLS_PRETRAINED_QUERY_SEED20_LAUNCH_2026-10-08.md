# 原始Query引导SLS：seed20实际启动记录

状态：已通过数值/梯度/资源自检，已正式训练；尚无五域EER，不是完成报告。方案见 `SLS_PRETRAINED_QUERY_GUIDED_ADAPTATION_PLAN_2026-10-08.md`。

## 实际执行

- SSH：`root@hn01-ssh.gpuhome.cc:30646`；无远程Codex。
- 正式PID：2798；process_create_time：1791466601.6。
- 启动：2026-10-08 13:36:42 UTC（北京时间21:36:42）。
- 命令：`/opt/conda/envs/SLS/bin/python -u -m tools.train_sls_pretrained_query --config /root/rivermind-data/cross-language-ADD-with-xls-r/experiments/configs/sls_pretrained_query_seed20.yaml`。
- 输出：`models/SLS_pretrained_query_seed20/`。
- 日志：`experiments/logs/sls_pretrained_query_seed20.log`。
- 只新增候选模型、训练器、启动器和配置；没有整体覆盖旧代码、改变data.pad、重采样、加噪或重跑基线。没有删除历史文件、下载权重、关机或Git提交为有效成果。

正式训练从总epoch4开始；原始SSL和既有共同3轮预热SHA已重新计算并通过校验。自检权重被丢弃，正式模型重新构造、加载预热后端、创建新AdamW并恢复共同RNG。原始分支的额外NumPy抽样不改变任务分支随机序列。

## 自检实际结果

- FP32、BF16：native与外部Query路径所有24层和最终分数最大绝对差均为0。
- 原始与任务Query路径均执行24层，无遗漏；对少量源域样本改变Query的最大分数变化为10.0541，只证明路径接通，不证明跨语言机制。
- 原始分支、任务q_proj没有梯度；第1/24层K/V/FFN和SLS门控/FC均有非零有限梯度。
- 固定源batch三次数值更新loss：0.0929458→0.0171504→0.0111866；不当作独立验证结果或目标性能。
- 可训练参数309,436,549；权重tensor约1,237,746,196 bytes，非小增量适配器。
- batch16自检峰值reserved显存11.3398 GiB；自检总27.09秒。
- 原子保存安全预算4GiB；启动前持久盘约18GiB空余。只存一个best，不存Adam恢复状态或每epoch全模型。

详细自检、源样本ID及代码/库/资产签名已同步 `models/SLS_pretrained_query_seed20/self_test.json`、`experiment.json`。这些检查没有目标域训练或用目标结果挑实现。

## 正式训练进度与ETA

2026-10-08 13:40:26 UTC的实际进度：epoch4，800 batch，12,800源样本，累计legacy weighted分类loss=0.0256116，耗时166.29秒；进程仍正常。源损失下降不是跨语成功证据。

首20batch实测5.01秒，线性估计全轮397.59秒，reserved11.2930 GiB；稳态800batch约0.208秒/batch，估计完整一轮约330秒。两者均为估算，必须以完整epoch记录替代，不承诺固定完成时间。

5轮新微调加第4/8两次五域评估，按当前训练速度和历史评估估计启动后45–65分钟；第4轮完整五域预计启动后约15–20分钟。正式阶段75分钟预算，超预算停止且记录；未知IO和五域实测仍会改变ETA。不按中间某个域挑权重或拼接成绩。

## 评估和后续

单SLS头，CH/DE/UK/ES/ITW在总4/8完整评估；截至每阶段最低完整源分类loss选权重，逐域及均值完整记录，评估后恢复当前训练权重及RNG。第4轮仅预登记严重失败停止；其余到8，之后不延长。扩种子仅由登记闸门决定，未完成前不声称稳定。

更新已有rp heartbeat为本候选，每1小时检查；正常无完整阶段变化不通知。OpenAI Docs技能只用于核对监控管理，不影响模型或训练协议。脚本即时评估/停止，不等待heartbeat。异常不自动重跑，无Adam恢复文件不能精确续训。结果完成后同步小型记录并写结论，失败或无法继续时暂停监控。

本报告是启动时快照，不是实时进度；当前没有本候选目标EER或成功结论。
