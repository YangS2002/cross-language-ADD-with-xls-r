# UK/ES 数据转存记录（2026-10-03）

本次只传输数据，不训练模型、不修改预处理或评估指标。来源为本机 `C:\Users\26600\Downloads\Compressed`；远端 SSH 为 `root@hn01-ssh.gpuhome.cc:30351`，持久存储目录为 `/root/rivermind-data/datasets/mlaad_eval/`。上传至独立暂存目录后，远端 SHA-256 与本机逐文件相同，才移入正式目录；原有文件未覆盖。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `mlaad_eval_es_clip.zip` | 2,465,507,736 | `b80181927bee7d10bd29aef99ec39a0046779057b288aa6245d40804a23e2826` |
| `mlaad_eval_uk_clip.zip` | 2,611,018,552 | `91939a78cd5ea9f5c0ccfff85e53404ceda59acc981bac9c17e81a98e141aa63` |
| `mlaad_eval_es.txt` | 1,740,451 | `95775f30cae815708dc595e98b5915a751844d7bfe63f0d9179312b94ccd706b` |
| `mlaad_eval_uk.txt` | 1,009,040 | `a4cf6d3883a6fa94d06c79754a49ffcc480a4fbcbf5efa5ce13b39fc762ea0ab` |

为兼容现有 `tools/evaluate_four_domains_shared_xlsr.py` 的固定路径，在 `/root/rivermind-fs/` 建立两份 ZIP 的软链接，在 `/root/rivermind-fs/exp/` 建立两份标签的软链接，均指向持久存储上的原件；没有重复存放压缩包。`/root/rivermind-fs` 当前位于容器 overlay，未来若该目录重建，需重建软链接，持久数据盘的原件不受影响。

使用现有 `tools/evaluate_e3_dual_xlsr.py --domains uk es --preflight` 验证通过：UK 标签 9,709 条（真 4,709、假 5,000）；ES 13,655 条（真 6,655、假 7,000）。全部标签路径均能在对应 ZIP 中找到，并成功读取首末音频。预检没有加载模型或计算 EER。传输完成后持久盘剩余空间约 14 GB。未传输 ITW。
