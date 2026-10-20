# 随机种子不稳定、跨语言泛化与近期研究分析

## 结论先行

当前 M2 不能说明二阶动态残差有效。它只能说明：在 seed 20 对应的优化轨迹上，
二阶残差信号可能有用；但该结构没有把模型稳定地引导到同一类跨语言解。seed 1001
的显著退化使其不满足有效改进的基本条件。

近期研究与我们的现象基本一致：跨域语音伪造检测的主要困难通常不是目标样本“更难”，
而是训练域和目标域存在不同的语言、生成器、说话人和录制捷径。模型训练损失可以接近
零，但不同随机轨迹会选择不同捷径，从而产生完全不同的跨语言性能。

## 当前现象的更具体解释

### 1. 大量等价的训练域最优解，但跨语言行为不同

两个基线种子以及 M2 两个种子都能把训练损失压到很低，但 DE/CH EER 差异很大。
这说明训练域二分类目标不能唯一确定一个跨语言解。模型可以分别依赖：

- ASVspoof 训练集中特定声码器的局部痕迹；
- 训练说话人的身份、声道或语料来源；
- 英语音素和语义相关表示；
- 真伪类别与上述因素之间的偶然相关性。

这些特征都能降低源域损失，但只有一部分能迁移到中文和德语。

### 2. 早期 batch 会决定捷径的形成顺序

不同 shuffle 顺序改变早期梯度。某些批次先强化可迁移伪造痕迹，模型随后可能继续沿
较好的盆地优化；另一些批次先强化生成器、说话人或英语内容捷径，后续即使训练损失
继续下降，也不一定会离开该解。这与目前观察到“早期样本顺序决定后期 DE 上限”一致。

### 3. M2 额外放大了轨迹扰动

M2 的二阶分支在原双头前执行并包含 dropout。即使门控为零，它仍改变两个基线头的
dropout 随机序列。因此 M2 的 seed 1001 不是纯粹的“基线 + 几乎为零的残差”，而是
从第一个 batch 开始走了不同随机路径。

NeurIPS 2022 的数值稳定性研究表明，即使只在浮点数最后一位引入扰动，也可能被深网
优化放大，产生与随机 batch 顺序相当的测试方差。这不是说我们的失败主要来自浮点误差，
而是为“很小的随机路径变化会产生完全不同泛化解”提供了机制证据：
[Surprising Instabilities in Training Deep Networks and a Theoretical Analysis](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf)。

## 近期语音伪造论文如何解释和处理

### A. 不是增加容量，而是避免域特定捷径

Interspeech 2024 将跨域性能差分解为“目标样本更难”和“目标分布不同”，发现主要来源
是后者，并明确指出单纯增大模型容量不能解决泛化问题。这与 M2 的教训一致：添加更复杂
的二阶 Transformer 分支并不会自然得到语言不变表示。

来源：[Harder or Different? Understanding Generalization of Audio Deepfake Detection](https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.html)。

### B. 显式建模低层声学与高层语义之间的不一致

2026 年 Information Fusion 的 LHCC 使用 XLS-R，将隐藏层物理划分为低层声学视图和
高层语义/身份视图，然后用非对称融合模块检查两种视图之间的结构一致性，而不是把所有
层差分无差别地输入同一个分支。论文将跨域失败归因于模型过拟合生成器特定局部痕迹，
其模型在 ITW 上报告 7.1% EER。

来源：[LHCC: Cross-domain audio deepfake detection via consistency-aware fusion of acoustic and semantic representations](https://www.sciencedirect.com/science/article/pii/S156625352600237X)。

这对当前项目最有启发：我们已经使用冻结 XLS-R 的多层输出，但当前 transition 结构没有
明确区分“声学构造层”和“语义层”，二阶差分也没有显式约束二者一致性。模块可能加入了
更多变化信息，却没有规定哪些变化应当跨语言稳定。

### C. 学习攻击无关的公共子空间

Interspeech 2025 的 `Beyond Attacks` 与我们的协议非常接近：只用英文 ASVspoof 2019
训练冻结 XLS-R 前端，在未见的多种印度语言上测试。论文采用两种办法：

1. Attack-Invariant Encoder-Decoder：让不同攻击类型的表示互相重建，迫使瓶颈忽略
   攻击特定细节；
2. Common-Specific Decomposition：把分类器分解为公共部分和攻击特定部分，推理时
   只使用公共分类器。

组合方法在多语言测试上取得 34% 相对 EER 改善。它不需要目标语言训练数据，使用的是
源域已有的攻击/声码器标签。

来源：[Beyond Attacks: Advancing Fake Speech Detection with Attack-Agnostic Methods](https://www.isca-archive.org/interspeech_2025/chandra25b_interspeech.pdf)。

本仓库的数据管线已经提供 `vocoder_label`、`vocoder_n` 和 `spk_id`，只是当前训练循环
没有使用，因此该方向可以落地，不需要重新构造数据标注。

### D. 显式分离风格和语言内容

NeurIPS 2024 的 SLIM 不直接依靠更复杂的二分类头，而是先在真实语音上学习风格—语言
依赖，再利用伪造语音中的风格—语言失配进行检测。它冻结特征编码器，只训练后端；实验
重复三个随机种子并报告均值。作者还发现，简单拼接风格和语言嵌入并不能带来明显改善，
必须显式建模二者依赖关系。

来源：[SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection](https://papers.nips.cc/paper_files/paper/2024/hash/7d6930fd71740eae21224a5ffb70cb8c-Abstract-Conference.html)。

这进一步说明 M2 的问题不是“二阶特征不够强”，而可能是残差相加缺少明确的不变性目标。

### E. 冻结 SSL 前端并限制下游自由度

Interspeech 2024 发现，大型冻结 SSL 表示配合不足 2k 参数的逻辑回归，在八个深伪数据集
上将相对 RawNet2 的 EER 从 30.9% 降到 8.8%。这说明跨域任务中，更复杂的可训练后端
可能反而提供更多学习捷径的自由度。

来源：[Towards generalisable and calibrated audio deepfake detection with self-supervised representations](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html)。

我们已经冻结 XLS-R，但双 transition 后端仍包含两个 Transformer 头、dropout 和关系
一致性训练。该结果提示后续应该检查后端复杂度，而不是继续叠加新分支。

### F. 降低 sharpness，直接减少多种子方差

Interspeech 2025 是与“种子不稳定”最直接的近期工作。论文在 AASIST、Wav2Vec2 和
XLS-R 等多个系统上做三个随机种子，报告 mean±std；研究发现语言、攻击和声道失配都会
提高模型 sharpness，并用 SAM 寻找更平坦的解。多数模型和数据集上，SAM 同时降低 EER
和跨种子标准差。例如 W2V-Base+Linear 在 ADD 上从 `30.47±9.03` 降到
`24.29±3.57`。

来源：[From Sharpness to Better Generalization for Speech Deepfake Detection](https://www.isca-archive.org/interspeech_2025/huang25e_interspeech.pdf)。

但该论文也指出，SAM 对 XLS-R 的收益较小且因数据集而异。SAM 通常需要两次前后向，
训练开销接近翻倍。因此它适合作为确定模型结构后的稳定化步骤，不应作为当前第一个实验。

### G. 后续“哪些样本决定好解”的研究已有数据侧依据

ACL 2026 的数据中心方法发现，训练数据的来源与生成器多样性比单纯扩大数据量更重要；
其 DOSS-Weight 保留样本但降低冗余域权重，比直接删样本更有效。该工作支持后续研究早期
关键样本、生成器覆盖和采样权重，但它属于数据/采样方向，不应与当前模型实验同时修改。

来源：[A Data-Centric Approach to Generalizable Speech Deepfake Detection](https://aclanthology.org/2026.acl-long.796.pdf)。

## 对当前模型的优先级判断

### 第一优先：低层—高层一致性模型，而不是继续堆二阶差分

建议从双 transition 基线回退 M2，保留原始双头，只增加一个**确定性的低—高层一致性
校正模块**：

- 低层 XLS-R 表示作为声学构造视图；
- 高层表示作为语言/语义视图；
- 计算投影后的差值、乘积或非对称门控映射，得到 mismatch 表示；
- 以零初始化的小残差注入现有双头；
- 新模块第一版不使用 dropout，避免再次改变基线头的随机序列；
- 不改损失、优化器、数据和双头关系约束。

这比 M2 更合理，因为它给新特征明确的跨语言含义：检测声学构造和高层内容是否一致，
而不是假设“层间二阶变化天然语言不变”。它也符合一次只改一个模块的实验原则。

在实现前先对冻结 XLS-R 各层做一次离线 CKA/线性 probe，用现有训练数据确定低高层分界。
这是一次只读分析，不需要完整训练，可避免凭经验随意选层。

### 第二优先：公共—声码器特定分解

如果低高一致性仍不稳定，再利用已有 `vocoder_label` 做一个 Common-Specific head：训练时
保留公共分类器和声码器特定偏置，推理只用公共部分。第一版只替换最终分类层，不同时增加
encoder-decoder、对抗头或新的数据增强。

该结构直接限制模型通过某个训练声码器捷径完成分类，理论上比全量隐藏层残差更贴近当前
根因。需要注意：完全去除声码器信息也可能删除真实伪造线索，所以应该“分解并保留公共
部分”，而不是简单做声码器对抗抹除。

### 第三优先：SAM 或权重平均

只有模型侧候选在两个筛选种子上方向一致但仍有较大方差时，再尝试 SAM。它有语音深伪
任务的直接多种子证据，但计算开销高。SWA/EMA 的成本更低，可作为 checkpoint 稳定化
备选，不过它们不能替代一个本身跨语言含义错误的模型结构。

## 建议的低开销实验协议

1. **候选筛选：** seed 20 和 seed 1001，各 10 轮，保留最低训练损失权重。
2. **通过条件：** 两个种子的联合 EER 都不能显著劣于对应基线；平均联合 EER、最差语言
   EER 和两种子差值至少两项改善。不能以一个种子的巨大收益抵消另一个种子的退化。
3. **正式确认：** 只有通过双种子筛选的候选才补第三个种子，并报告 mean±std。
4. **外部验证：** 三种子方向一致后才评估 ITW。
5. **版本保存：** 三种子确认有效后才提交 Git；失败实验只保留配置、CSV、最低训练损失
   权重和分析报告。

## 不建议当前做的事情

- 不继续调 M2 门控上限、hidden size、Transformer 层数；当前实现尚有 RNG 混杂。
- 不靠寻找更好的 seed 或在 CH/DE 上事后挑 epoch。
- 不同时加入低高一致性、声码器分解、SAM 和新损失。
- 不优先使用 prototype 方法；它并未直接针对当前的跨语言路径不稳定，而且会额外引入
  类中心与度量损失超参数。
- 不立即做 ITW；目前连两个已知跨语言测试集上的方向都不一致。

