# THA 项目 VAE 优化与 RFlow 策略备忘录 (VAE Optimization Memo)

## VAE 训练结果

```
--- Summary for pre ---
Epoch:           175
L1:              0.009550 (Best: 0.009550)
MSE:             0.000999
PSNR:            36.25
SSIM:            0.9804
------------------------------
Scale Factor:    2.787793
Global Mean:     0.001770
rFID (Recon):    0.000041  <- 越接近 0 模型越好
iFID (Interp):   0.011456  <- 绝对插值表现
Gap (i - r):     0.011415  <- 插值造成的性能退化量
------------------------------
>> ΔFID < 0.05 隐空间高度平滑连续
   隐空间高度平滑连续，特征分布高度一致。
```


```
--- Summary for metal ---
Epoch:           113
L1:              0.007942 (Best: 0.007942)
MSE:             0.000297
PSNR:            41.34
SSIM:            0.9928
------------------------------
Scale Factor:    7.788557
Global Mean:     -0.000221
rEikonal (Recon):        0.010693  <- 越接近 0 模型越好
iEikonal (Interp):       0.011030  <- 绝对插值表现
Gap (i - r):     0.000337  <- 插值造成的性能退化量
------------------------------
>> ΔEikonal < 0.01 隐空间高度平滑连续
```

---

## 1. 核心理论：重建-生成悖论 (Reconstruction-Generation Dilemma)

在训练用于扩散模型的 VAE 时，**不能仅仅追求完美的重建指标（如 PSNR/L1）**。
* **孤立流形风险**：当 VAE 为了最小化重建误差，倾向于将训练样本在潜在空间中“死记硬背”为互不连通的孤岛（Isolated GMM）。
* **插值幻觉**：在这种破碎的潜在空间中，虽然训练样本能完美解码，但在样本之间的空白区域（插值区）解码会产生严重的物理畸变或噪声。这对需要强力泛化能力（生成未见过的假体或解剖形态）的 RFlow 来说是灾难性的。

---

## 2. VAE 质量评估的“黄金标准”：插值一致性 (Interpolation Consistency)

为了评估 VAE 潜在空间的连续性，我们引入了针对性的插值评估指标（在 `c2_prepare_scale_factor.py` 中实现）：

* **rMetric (Reconstruction Metric)**: 真实样本的重建误差。衡量 VAE 的**记忆能力**。
* **iMetric (Interpolated Metric)**: 潜在空间中最近邻样本的均值插值的解码误差。衡量 VAE 潜在空间的**连续性和泛化能力**。
* **10 倍法则 (The 10x Rule)**: 
  * 若 `iMetric / rMetric < 10`：潜在空间平滑且连续，非常适合 RFlow。
  * 若 `iMetric / rMetric > 50`：陷入“重建-生成悖论”，潜在空间严重碎片化。

### 2.1 针对解剖图像 (Subtask: `pre`)
* **指标**: **iFID (Interpolated FID)**，使用 3D MedicalNet 提取特征。
* **当前状态**: `rFID=0.0003, iFID=0.008, Ratio=22.6x`。重建极好，插值有一定 Gap。
* **策略**: 鉴于 `pre` 在当前架构中仅作为 RFlow 的**条件输入 (Condition)**，而非生成目标，22 倍的 Gap 不会阻碍 RFlow 学习空间映射。

### 2.2 针对几何假体 (Subtask: `metal`)
* **指标**: **iEikonal (Interpolated Eikonal Error)**，衡量生成场偏离物理约束 $|\nabla \phi| = 1/sdf\_t$ 的程度。
* **当前状态**: `rEikonal=0.0026, iEikonal=0.0048, Ratio=1.85x`。
* **策略**: 极其优异。由于引入了程函损失（Eikonal Loss）强正则化，`metal` 的潜在空间构建了完美的连续流形，极大降低了 RFlow 在导航阶段（Navigation Phase）生成假体的“幻觉”概率。

---

## 3. RFlow 架构下的 VAE 缩放法则 (Scaling Laws)

如果未来需要提升 RFlow 的生成质量或支持凭空生成复杂的医学解剖图像，需要权衡以下三个关键参数：

| 调整策略 | 优缺点与权衡 | 对 VAE 的影响 | 对后续 RFlow 的影响 | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- |
| **增大 `kl_weight`** | 强迫特征融合，消除碎片化孤岛。 | **L1 损失上升**（重建模糊），但 iFID 降低（连续性极佳）。 | **无显存影响**。极大地提高了扩散模型在去噪轨迹中的泛化鲁棒性。 | 修复插值生成的假体出现物理断裂时首选。 |
| **增大 VAE 宽度**<br>(增加 `channels`) | 在极强 `kl_weight` 压迫下，凭借强大解码能力维持高频细节。 | **VAE 显存上升**。可通过降低滑窗推理尺寸 (`sw_batch_size`) 来缓解。 | **无显存影响**。RFlow 面对的特征尺寸没有改变。 | 追求“既要流形连续，又要重建锐利”时使用。 |
| **增大潜通道数**<br>(增加 `latent_channels`) | 放宽信息压缩瓶颈，L1 重建轻易达到极低水平。 | VAE 压力减轻，重建完美。 | **RFlow 显存爆炸**。特征序列长度线性增加，在交叉/自注意力机制中导致显存二次方暴增。 | **慎用**。仅在 4 通道彻底无法表达复杂解剖特征，且硬件算力充裕时考虑。 |

---

### 当前架构总结
目前的配置（`latent_channels=4`, `channels=(32, 64, 128)`）在 3D VRAM 预算极其吃紧的情况下，是一套极具性价比的架构。它将有限的算力留给了负责空间推理和分布学习的 RFlow，同时 VAE 凭借物理约束（Eikonal Loss）和对抗训练（PatchGAN）提供了足够鲁棒和连续的潜在空间支撑。

