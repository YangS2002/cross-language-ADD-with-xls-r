# PSR失败实验清理清单

用户明确要求清理并重新设计。本次只清理最新PSR失败候选，不扩大到历史实验或数据。

- 删除远端普通文件/root/rivermind-data/cross-language-ADD-with-xls-r/models/SLS_subspace_restoration_seed20/psr/best.pth。
- 删除前真实大小1338647840字节，SHA256为14e9fa5e5a20f857f8d766c9ec73b048fedefd1c7c4d22b42efb73c814108cc9；readlink核验不是另一个文件的符号链接。
- 删除前确认训练和停止守护均已退出，status=stopped_by_user_epoch8_rule；没有删除正在使用的权重。
- 删除后文件不存在，持久盘空闲6184112128字节（约5.76GiB）。没有备份，不能原样恢复此checkpoint；不能把无Adam状态的重训称精确恢复。
- 已保留本地和远端全部小型epoch_results/evaluation_results、eval_epoch4/8 JSON、配置签名、停止决定和失败报告。独立PSR代码保留作失败研究记录，但方法已退役，不作有效成果提交，不继续运行。
- 原始xlsr2_300m.pt、固定SLS基线、共同warmup、全部数据和其他历史权重均未动；无递归清理、无整体覆盖src/model.py/train.py、无关机。
- 原rp监控已实际更新为PAUSED并核验，避免失败实验结束后继续消耗监控额度；该暂停只针对已终止实验监控，不代表研究目标已完成。

后续设计见SLS_CONTENT_ADVERSARIAL_REDESIGN_2026-10-07.md。本次未启动新训练、未下载ASR或数据、未把失败方法改名重跑。
