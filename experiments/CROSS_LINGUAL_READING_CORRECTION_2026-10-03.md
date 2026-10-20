# 跨语言文献阅读纠偏（2026-10-03）

状态：文献阅读记录；未设计、启动或筛选新实验。核心问题是**英语源域训练、目标语言零样本的语音伪造检测**。DE/UK/ES 是非英语目标语言；ITW 是英语跨语料压力测试，不能混称跨语言。跨数据集研究仅在同时明确语言因素时进入核心证据链。

## 文献筛选规则

1. 首先核对训练语言和测试语言是否真正分离、目标语言是否被用于适配/调参。
2. 再核对生成器、真实音频来源、通道、说话人是否共同变化。若共同变化，只能称跨语言**兼**跨域，不可把退化单因归于语言。
3. 区分“仅目标伪造样本的检测率”和目标域同时有真实/伪造样本的 EER。二者不能横向比较。
4. 分清冻结 XLS-R 与端到端微调，以及是否使用目标语数据。论文指标不直接作为本项目相同协议基线。

## 已核对方法及实验协议的核心文献

| 文献 | 与本研究的直接关系 | 重要限制 |
|---|---|---|
| [DECRO / Ba et al., WWW 2023，数据集官方说明](https://github.com/petrichorwq/DECRO-dataset) | 英语与汉语部分采用相同**类型**的合成算法，是研究语言差异的重要相对受控资源。 | 原论文全文未获取；[作者公开代码说明](https://github.com/petrichorwq/Cross-lingual_DF_detection)其 fine-tune/联合训练/ADA/MMD 方案使用中文训练音频，不能作为本项目零目标语样本的方法依据。相同算法类型不等于所有生成条件完全相同。 |
| [Moreno et al., SPSC 2025](https://www.isca-archive.org/spsc_2025/moreno25_spsc.pdf) | 英语 ASVspoof 5 训练并微调 XLS-R+AASIST，目标 10 语在标准化 MMS/VITS 合成条件下仍有很大分数差异；乌克兰语尤其难。支持“语言相关的合成/检测交互值得单独研究”。 | 受控分析只用每种语言的 **1000 条伪造音频**，报告伪造分数而非各语言真伪 EER；不同语言的 TTS 是各自训练的模型，不能说严格只变了语言。 |
| [Are Audio DeepFake Detection Models Polyglots?, 2024/25](https://arxiv.org/html/2412.17924v2) | 对英语训练模型做 DE/ES/UK 等语言测试，另比较目标语微调、其他语言微调及排除目标语言的训练；明确区分多种跨语设置。 | 部分策略使用目标语数据，不符合本项目零目标样本；MLAAD 各语言生成器组合不完全一致，尤其 UK 用不同生成器补足。其漂亮的英语→UK 数值不能直接当作纯语言迁移效果。 |
| [Language Orthogonalization for Zero-Shot Cross-Lingual Audio Deepfake Detection, 2026 预印本](https://arxiv.org/html/2609.16458v1) | 严格留出目标语言，用源域真实音频拟合 LID→冻结 SSL 表征回归，再消除语言可预测部分；包含 XLS-R-300M。直接说明“去语言子空间”已有先例。 | 冻结 SSL + mean/std + 逻辑回归，与端到端微调不同；在单源语的近语言 XLS-R 分组，EER 3.33→4.05，不能宣称所有语言对均获益。 |
| [XMAD-Bench, EACL 2026](https://aclanthology.org/2026.findings-eacl.162.pdf) | 明确分开语言、语料和生成器轴；多源语言训练到英语/汉语时，见过生成器的跨语测试和未见生成器的跨语测试难度不同。 | 训练语种含阿拉伯语、德语、罗马尼亚语、俄语、西班牙语，不是本项目英语单源训练；并使用数据增强、域内挑权重，数字不可直接比较。 |
| [When Spoof Detectors Travel / LRLspoof, 2026 预印本](https://arxiv.org/html/2603.02364) | 66 语言×24 TTS，固定检测器及生成器比较语言；指出语言与生成器的交互。 | 目标集**只有伪造语音**，用外部 EER 阈值后报告 spoof rejection rate，不提供各目标语言 EER；不能用其指标替代我们的 DE/UK/ES EER。 |
| [Heterogeneity over Homogeneity, NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.160.pdf) | 冻结多语言 PTM 对比提供一组跨语/跨语料事实。附录真正跨语料的 XLS-R：ASV 训练→DECRO-C 在 120 维为 22.78% EER、240 维为 21.33%。 | 主表中 DECRO-C 约 1% EER 是**在中文集单独训练模型**所得，不能当英语→中文零样本成绩；ITW 不是异语言。 |

## 文献链下一轮优先阅读

优先追 **DECRO → Moreno / Polyglots → Language Orthogonalization** 的共同问题，而非泛化检测器架构：

- [SEA-Spoof, 2025](https://arxiv.org/abs/2509.19865)：核查六种语言、同/异生成器设置与 leave-language-out 协议。
- [MLAAD 数据集论文](https://arxiv.org/abs/2401.09512)：核查本地使用的数据版本、语种和 TTS 组合；论文数据集已多次扩版，不混用统计。
- [XMAD-Bench](https://aclanthology.org/2026.findings-eacl.162.pdf) 的 cross-lingual × cross-domain 交叉表：逐项核对是否能分辨两个迁移轴。
- [XLS-R 原论文](https://arxiv.org/abs/2111.09296)：只作为跨语言表征预训练依据，不把语音识别/翻译迁移能力等同伪造检测迁移能力。

## 当前文献结论（暂不构造方法）

最稳妥的论文问题不是笼统“跨域泛化”，而是：**在目标语言无任何训练/适配样本时，XLS-R 上由英语伪造标签学到的判别规则，哪些部分能跨语言转移，哪些部分依赖语言与生成器的交互？** 已有研究分别显示语言差异可影响伪造分数、语言信息可被统计消除、不同生成器会改变跨语表现；但这些证据尚未证明本项目失效的具体机制，更不能预先断言“冻结分支负责跨语、微调分支负责伪造”。下一步仍是继续精读跨语言论文与协议，不启动模型训练。
