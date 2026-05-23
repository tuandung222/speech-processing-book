# Modern ASR Architectures

## Tổng quan

Chương này trình bày các kiến trúc ASR hiện đại nhất, từ **Conformer** đến **Zipformer**, **E-Branchformer**, và xu hướng sử dụng **MoE** (Mixture of Experts) cùng **SSM/Mamba** trong speech recognition.

!!! note "Bước tiến từ Transformer đến Conformer"
    Transformer thuần túy (self-attention) không capture tốt local acoustic patterns. Conformer kết hợp **convolution** (local) + **self-attention** (global) tạo ra kiến trúc vượt trội cho speech.


## Conformer

### Kiến trúc

Conformer [^gulati2020conformer] xếp chồng các **Conformer blocks**, mỗi block gồm 4 modules theo thứ tự "macaron-style":

<a id="eq-conformer-ffn1"></a>

$$
\mathbf{x}' = \mathbf{x} + \frac{1}{2}\text{FFN}_1(\mathbf{x})
$$

<a id="eq-conformer-mhsa"></a>

$$
\mathbf{x}'' = \mathbf{x}' + \text{MHSA}(\mathbf{x}')
$$

<a id="eq-conformer-conv"></a>

$$
\mathbf{x}''' = \mathbf{x}'' + \text{Conv}(\mathbf{x}'')
$$

<a id="eq-conformer-ffn2"></a>

$$
\mathbf{y} = \text{LayerNorm}\left(\mathbf{x}''' + \frac{1}{2}\text{FFN}_2(\mathbf{x}''')\right)
$$

**Convolution Module** sử dụng depthwise separable convolution:

<a id="eq-conformer-conv-module"></a>

$$
\text{Conv}(\mathbf{x}) = \text{PointwiseConv} \circ \text{GLU} \circ \text{DepthwiseConv}_{k} \circ \text{PointwiseConv}(\mathbf{x})
$$

với kernel size $k = 31$ (capture ~310ms context ở 10ms frame rate).

```python
#| eval: false
#| code-fold: true
#| code-summary: "Conformer Block Implementation"
import torch
import torch.nn as nn
from torch import Tensor

class ConformerBlock(nn.Module):
    """Single Conformer block with macaron-style FFN sandwich."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForward(d_model, ff_expansion, dropout)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForward(d_model, ff_expansion, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D] - float32
        x = x + 0.5 * self.ffn1(x)           # [B, T, D] - float32
        residual = x
        x_att, _ = self.mhsa(x, x, x)        # [B, T, D] - float32
        x = residual + x_att                   # [B, T, D] - float32
        x = x + self.conv(x)                  # [B, T, D] - float32
        x = x + 0.5 * self.ffn2(x)           # [B, T, D] - float32
        return self.ln(x)                      # [B, T, D] - float32


class FeedForward(nn.Module):
    def __init__(self, d: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * expansion),  # [B, T, D] -> [B, T, D*4]
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d * expansion, d),  # [B, T, D*4] -> [B, T, D]
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [B, T, D] - float32


class ConvolutionModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pointwise1 = nn.Conv1d(d, 2 * d, 1)     # GLU needs 2x
        self.depthwise = nn.Conv1d(
            d, d, kernel, padding=kernel // 2, groups=d
        )
        self.bn = nn.BatchNorm1d(d)
        self.pointwise2 = nn.Conv1d(d, d, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        x = self.ln(x).transpose(1, 2)     # [B, D, T]
        x = self.pointwise1(x)              # [B, 2D, T]
        x = x.chunk(2, dim=1)[0] * torch.sigmoid(
            x.chunk(2, dim=1)[1]
        )                                    # [B, D, T] - GLU
        x = self.depthwise(x)               # [B, D, T]
        x = self.bn(x)                      # [B, D, T]
        x = nn.functional.silu(x)           # [B, D, T]
        x = self.pointwise2(x)              # [B, D, T]
        return self.dropout(x).transpose(1, 2)  # [B, T, D]
```

### Kết quả

| Model | Params | LibriSpeech test-clean | test-other |
|-------|--------|----------------------|------------|
| Transformer | 118M | 2.4% WER | 5.6% |
| **Conformer-L** | 118M | **2.1%** | **4.3%** |
| Conformer-XL | 600M | 1.9% | 3.9% |

: Conformer vs Transformer trên LibriSpeech <a id="tbl-conformer-results"></a>

## E-Branchformer

E-Branchformer [^kim2023ebranchformer] cải tiến Conformer bằng **parallel branches**:

<a id="eq-ebranchformer"></a>

$$
\mathbf{y} = \text{Merge}(\text{GlobalBranch}(\mathbf{x}), \text{LocalBranch}(\mathbf{x})) + \mathbf{x}
$$

- **Global Branch**: Multi-head self-attention (MHSA)
- **Local Branch**: Convolutional gating mechanism (cgMLP)
- **Merge**: Linear projection + concatenation

!!! note "Tại sao E-Branchformer?"
    Conformer xử lý tuần tự: FFN → MHSA → Conv → FFN. E-Branchformer chạy **song song** global và local branches, cho phép tốc độ training nhanh hơn và quality tương đương hoặc tốt hơn.


## Zipformer

Zipformer [^yao2023zipformer] (từ nhóm k2/icefall) cải tiến nhiều khía cạnh:

### Temporal Downsampling

Sử dụng **multi-scale architecture** với different frame rates ở các layers khác nhau:

- Layers đầu: 50 fps (10ms)
- Layers giữa: 25 fps (20ms) - downsample 2x
- Layers cuối: 12.5 fps (40ms) - downsample thêm 2x

<a id="eq-zipformer-fps"></a>

$$
\text{fps}_l = \frac{50}{2^{d_l}}, \quad d_l \in \{0, 1, 2\}
$$

### Swoosh Activation

Thay SiLU bằng **Swoosh** activation:

<a id="eq-swoosh"></a>

$$
\text{Swoosh}(x) = x \cdot \sigma(x - 1)
$$

### BiasNorm thay LayerNorm

<a id="eq-biasnorm"></a>

$$
\text{BiasNorm}(\mathbf{x}) = \frac{\mathbf{x} + \mathbf{b}}{\text{learn\_scale} \cdot \|\mathbf{x} + \mathbf{b}\|_{\text{RMS}}}
$$

| Model | Params | test-clean | test-other | Speed |
|-------|--------|-----------|------------|-------|
| Conformer | 116M | 2.1% | 4.3% | 1.0x |
| E-Branchformer | 116M | 2.1% | 4.2% | 1.1x |
| **Zipformer-L** | 148M | **1.9%** | **4.0%** | **1.3x** |

: So sánh kiến trúc ASR hiện đại <a id="tbl-modern-asr-comparison"></a>

## FastConformer (NVIDIA)

FastConformer [^rekesh2023fast] từ NVIDIA NeMo tối ưu Conformer cho production:

- **8x downsampling** ở CNN frontend (thay vì 4x) - giảm sequence length
- **Multi-blank CTC** - thêm blank symbols cho phép skip nhiều frames
- **Hybrid CTC/RNNT** training

<a id="eq-fastconformer-rtf"></a>

$$
\text{RTF}_{\text{FastConformer}} \approx 0.01 \text{ (trên A100)}
$$

## MoE trong Speech Recognition

### Tại sao MoE cho Speech?

Mixture of Experts (MoE) [^shazeer2017outrageously] cho phép **scale model capacity** mà không tăng compute proportionally:

<a id="eq-moe-speech"></a>

$$
\text{MoE}(\mathbf{x}) = \sum_{i=1}^{N} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})
$$

với $g_i(\mathbf{x})$ là routing weights (chỉ top-$k$ experts được activate).

### Branchformer-MoE

Kết hợp E-Branchformer với MoE routing [^you2024moe]:

- Thay FFN layers bằng MoE layers (8-64 experts, top-2 routing)
- Load balancing loss để tránh expert collapse:

<a id="eq-moe-balance"></a>

$$
\mathcal{L}_{\text{balance}} = N \sum_{i=1}^{N} f_i \cdot P_i
$$

trong đó $f_i$ là fraction of tokens routed to expert $i$, $P_i$ là average routing probability.

## Mamba/SSM trong Speech

### State Space Models cho Audio

S4 [^gu2022efficiently] và Mamba [^gu2023mamba] xử lý sequences dài hiệu quả - rất phù hợp cho audio (16kHz = 16000 samples/sec):

<a id="eq-ssm-speech"></a>

$$
\begin{aligned}
\mathbf{h}_{t+1} &= \bar{\mathbf{A}} \mathbf{h}_t + \bar{\mathbf{B}} x_t \\
y_t &= \mathbf{C} \mathbf{h}_t + D x_t
\end{aligned}
$$

### Hybrid Attention-SSM cho ASR

Jamba-style architectures [^lieber2024jamba] cho speech kết hợp:

- **Attention layers**: Capture global dependencies (cross-utterance context)
- **Mamba layers**: Xử lý long sequences hiệu quả (raw waveform hoặc high-resolution features)

| Kiến trúc | Complexity | Long Audio | Global Context |
|-----------|-----------|------------|----------------|
| Pure Attention | $O(T^2)$ | Kém | Tốt |
| Pure Mamba | $O(T)$ | Tốt | Trung bình |
| **Hybrid** | $O(T)$ | **Tốt** | **Tốt** |

: Attention vs Mamba vs Hybrid cho speech <a id="tbl-hybrid-ssm-speech"></a>

!!! warning "SSM cho Speech còn rất mới"
    Tính đến 2025-2026, research về Mamba/SSM cho speech vẫn đang ở giai đoạn đầu. Conformer và E-Branchformer vẫn là production standard. Hybrid architectures hứa hẹn nhưng chưa có deployment quy mô lớn.


## Tóm tắt

| Kiến trúc | Năm | Đặc điểm chính | Use case |
|-----------|-----|----------------|----------|
| Conformer | 2020 | Conv + Attention (macaron) | Production standard |
| E-Branchformer | 2022 | Parallel branches | ESPnet default |
| Zipformer | 2023 | Multi-scale + Swoosh | k2/icefall |
| FastConformer | 2023 | 8x downsample | NVIDIA NeMo |
| MoE-ASR | 2024 | Sparse expert routing | Scale capacity |
| Hybrid SSM | 2024+ | Attention + Mamba | Long-form audio |

: Tổng hợp kiến trúc ASR hiện đại <a id="tbl-asr-summary"></a>



---

<!-- References (auto-generated from .bib) -->
[^gulati2020conformer]: Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and others, "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech
[^kim2023ebranchformer]: Kim, Kwangyoun and Wu, Felix and Peng, Yifan and others, "E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition", IEEE Spoken Language Technology Workshop
[^yao2023zipformer]: Yao, Zengwei and Guo, Liyong and Yang, Xiaoyu and others, "Zipformer: A Faster and Better Encoder for Automatic Speech Recognition", International Conference on Learning Representations
[^rekesh2023fast]: Rekesh, Dima and Koluguri, Nithin Rao and Kriman, Samuel and others, "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition", IEEE Automatic Speech Recognition and Understanding Workshop
[^shazeer2017outrageously]: Shazeer, Noam and others, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", International Conference on Learning Representations
[^you2024moe]: You, Yifan and others, "E-Branchformer-based MoE Model for Efficient ASR", Interspeech
[^gu2022efficiently]: Gu, Albert and Goel, Karan and R{\'e, "Efficiently Modeling Long Sequences with Structured State Spaces", International Conference on Learning Representations
[^gu2023mamba]: Gu, Albert and Dao, Tri, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv preprint arXiv:2312.00752
[^lieber2024jamba]: Lieber, Opher and others, "Jamba: A Hybrid Transformer-Mamba Language Model", arXiv preprint arXiv:2403.19887
