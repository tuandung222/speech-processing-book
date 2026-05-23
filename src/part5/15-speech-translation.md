# Speech Translation

## Tổng quan

Speech Translation (ST) là bài toán dịch trực tiếp từ **speech sang text** hoặc **speech sang speech** ở ngôn ngữ khác. Đây là một trong những lĩnh vực nghiên cứu và ứng dụng quan trọng nhất, từ hội nghị quốc tế đến giao tiếp xuyên biên giới.

> **📝 Tại sao Speech Translation quan trọng?**
>
> - 7000+ ngôn ngữ trên thế giới, phần lớn không có hệ thống viết
> - Real-time interpretation cho hội nghị, du lịch, y tế
> - Kết nối cộng đồng không cùng ngôn ngữ



## Phân loại

### Các Task chính

| Task | Input | Output | Ký hiệu |
|------|-------|--------|----------|
| Speech-to-Text Translation | Speech (ngôn ngữ A) | Text (ngôn ngữ B) | S2TT |
| Speech-to-Speech Translation | Speech (ngôn ngữ A) | Speech (ngôn ngữ B) | S2ST |
| Simultaneous Translation | Streaming speech A | Streaming text/speech B | SimulST |

: Các task Speech Translation <a id="tbl-st-tasks"></a>

### Cascaded vs End-to-End

**Cascaded pipeline:**

<a id="eq-cascaded-st"></a>

$$
\text{Speech}_A \xrightarrow{\text{ASR}} \text{Text}_A \xrightarrow{\text{MT}} \text{Text}_B \xrightarrow{\text{TTS}} \text{Speech}_B
$$

**End-to-End:**

<a id="eq-e2e-st"></a>

$$
\text{Speech}_A \xrightarrow{\text{E2E Model}} \text{Text}_B \text{ hoặc } \text{Speech}_B
$$

| Tiêu chí | Cascaded | End-to-End |
|----------|---------|------------|
| Accuracy | Cao (mature components) | Đang cải thiện nhanh |
| Latency | Cao (3 stages) | Thấp hơn (1 stage) |
| Error propagation | Có (ASR errors → MT) | Không |
| Data requirement | ASR + MT data riêng | Paired speech-translation data |
| Prosody preservation | Mất qua text | Có thể giữ |
| Deployment | Modular, dễ debug | Đơn giản hơn |

: Cascaded vs End-to-End Speech Translation <a id="tbl-cascaded-vs-e2e"></a>

## Speech-to-Text Translation (S2TT)

### Encoder-Decoder Architecture

<a id="eq-s2tt-arch"></a>

$$
\mathbf{h} = \text{SpeechEncoder}(\mathbf{x}_{\text{source}}), \quad \mathbf{y} = \text{TextDecoder}(\mathbf{h})
$$

### Multi-task Training

Kết hợp ASR và ST objectives:

<a id="eq-s2tt-multitask"></a>

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{ST}} + (1 - \alpha) \mathcal{L}_{\text{ASR}}
$$

trong đó:

- $\mathcal{L}_{\text{ST}}$: Cross-entropy loss cho translated text
- $\mathcal{L}_{\text{ASR}}$: CTC hoặc attention loss cho source transcription

### Pre-training Strategies

1. **Speech encoder pre-training**: Wav2Vec 2.0 / HuBERT trên unlabeled audio
2. **Decoder pre-training**: mBART / mT5 trên parallel text
3. **Curriculum learning**: Train ASR trước → fine-tune cho ST

```python
#| eval: false
#| code-fold: true
#| code-summary: "S2TT Multi-task Training"
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

class S2TTMultitask(nn.Module):
    """Speech-to-Text Translation with multi-task ASR."""

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        src_vocab: int = 5000,   # source language vocab (for ASR)
        tgt_vocab: int = 32000,  # target language vocab (for ST)
    ) -> None:
        super().__init__()
        self.speech_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                encoder_dim, nhead=8, dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=12,
        )
        # ASR head (CTC)
        self.ctc_head = nn.Linear(encoder_dim, src_vocab)

        # ST decoder
        self.st_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                decoder_dim, nhead=8, dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6,
        )
        self.st_head = nn.Linear(decoder_dim, tgt_vocab)

    def forward(
        self,
        speech_features: Tensor,   # [B, T, D] - mel spectrogram
        tgt_tokens: Tensor,        # [B, S] - target language tokens
    ) -> Dict[str, Tensor]:
        # Encode speech
        encoder_out: Tensor = self.speech_encoder(
            speech_features
        )  # [B, T, D] - float32

        # ASR branch (CTC)
        ctc_logits: Tensor = self.ctc_head(
            encoder_out
        )  # [B, T, src_vocab] - float32

        # ST branch (autoregressive)
        tgt_emb: Tensor = self.tgt_embedding(
            tgt_tokens
        )  # [B, S, D] - float32
        decoder_out: Tensor = self.st_decoder(
            tgt_emb, encoder_out
        )  # [B, S, D] - float32
        st_logits: Tensor = self.st_head(
            decoder_out
        )  # [B, S, tgt_vocab] - float32

        return {
            "ctc_logits": ctc_logits,
            "st_logits": st_logits,
        }
```

## Speech-to-Speech Translation (S2ST)

### Direct S2ST Approaches

1. **Translatotron** [^jia2019direct]: Trực tiếp speech→spectrogram
2. **Translatotron 2** [^jia2022translatotron2]: + linguistic decoder
3. **Textless S2ST** [^lee2022direct]: Speech→discrete units→speech

### Unit-based S2ST

<a id="eq-unit-s2st"></a>

$$
\text{Speech}_A \xrightarrow{\text{Encoder}} \text{Units}_A \xrightarrow{\text{Unit-to-Unit}} \text{Units}_B \xrightarrow{\text{Vocoder}} \text{Speech}_B
$$

Discrete units từ HuBERT k-means clustering - không cần text!

> **📝 Textless NLP**
>
> Unit-based S2ST mở ra khả năng dịch cho **ngôn ngữ không có chữ viết** - chỉ cần paired speech data, không cần transcription.



## SeamlessM4T

SeamlessM4T [^communication2023seamlessm4t] từ Meta - model đa nhiệm speech translation:

### Capabilities

- S2TT: Speech-to-Text Translation (100+ languages)
- S2ST: Speech-to-Speech Translation (36 languages)
- T2TT: Text-to-Text Translation
- T2ST: Text-to-Speech Translation
- ASR: Automatic Speech Recognition

### Kiến trúc

SeamlessM4T v2 sử dụng:

1. **w2v-BERT 2.0**: Self-supervised speech encoder (600M params, trained trên 4.5M hours)
2. **Text encoder**: Multilingual text encoder
3. **Shared decoder**: Unified decoder cho text output
4. **Unit decoder**: Cho speech output (HuBERT units → vocoder)

### Seamless Streaming

SeamlessStreaming [^communication2023seamless] hỗ trợ **simultaneous translation**:

- **EMMA** (Efficient Monotonic Multihead Attention): Quyết định khi nào read vs write
- **Monotonic policy**: Chờ đủ context trước khi dịch

<a id="eq-monotonic-attention"></a>

$$
\text{READ/WRITE policy}: \quad p_i^j = \sigma\left(\frac{\mathbf{q}_j \cdot \mathbf{k}_i}{\sqrt{d}}\right)
$$

| Model | S2TT (BLEU) | S2ST (BLEU) | Languages | Streaming |
|-------|-------------|-------------|-----------|-----------|
| Whisper + NLLB | 25.3 | - | 99 | Không |
| SeamlessM4T v1 | 28.7 | 18.2 | 100 | Không |
| **SeamlessM4T v2** | **30.1** | **20.5** | 100+ | Không |
| SeamlessStreaming | 26.8 | 16.3 | 100+ | **Có** |

: So sánh Speech Translation models <a id="tbl-st-models"></a>

## Simultaneous Speech Translation

### Challenge

Simultaneous translation phải bắt đầu dịch **trước khi người nói kết thúc**:

<a id="eq-quality-latency"></a>

$$
\text{Quality} \propto \frac{1}{\text{Latency}}: \quad \text{BLEU} \downarrow \text{ khi } \text{AL} \downarrow
$$

### Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **AL** (Average Lagging) | Trung bình số source tokens chờ trước khi write |
| **AP** (Average Proportion) | Tỷ lệ source consumed trước mỗi write |
| **BLEU** | Translation quality |
| **DAL** (Differentiable AL) | Differentiable version cho training |

: Metrics cho Simultaneous Translation <a id="tbl-simul-metrics"></a>

### Policies

1. **Wait-k** [^ma2019stacl]: Luôn chờ $k$ source tokens trước khi write
2. **Adaptive**: Model tự quyết định READ/WRITE
3. **Monotonic Attention**: Soft attention cho READ/WRITE decisions

## Evaluation Datasets

| Dataset | Pairs | Languages | Hours | Source |
|---------|-------|-----------|-------|--------|
| CoVoST 2 | S2TT | 21→1, 1→15 | 2880h | CommonVoice |
| MuST-C | S2TT | En→{De,Es,Fr,...} | 400h | TED Talks |
| FLEURS | S2TT + LID | 102 | ~12h/lang | Curated |
| VoxPopuli | S2TT | 15 EU langs | 400K h | EU Parliament |

: Speech Translation evaluation datasets <a id="tbl-st-datasets"></a>

## Tóm tắt

1. **Speech Translation** gồm S2TT, S2ST, và Simultaneous Translation
2. **Cascaded** vẫn mạnh cho production, **E2E** đang tiến nhanh
3. **SeamlessM4T** là SOTA open-source cho multilingual speech translation
4. **Unit-based S2ST** cho phép dịch ngôn ngữ không có chữ viết
5. **Simultaneous translation** cần cân bằng quality vs latency
6. Vietnamese speech translation: cần thêm research và data (xem Chương 17)



---

<!-- References (auto-generated from .bib) -->
[^jia2019direct]: Jia, Ye and Weiss, Ron J and others, "Direct Speech-to-Speech Translation with a Sequence-to-Sequence Model", Interspeech
[^jia2022translatotron2]: Jia, Ye and Ramanovich, Michelle Tadmor and Remez, Tal and Pomerantz, Roi, "Translatotron 2: High-quality Direct Speech-to-Speech Translation with Voice Preservation", International Conference on Machine Learning
[^lee2022direct]: Lee, Ann and Chen, Peng-Jen and Wang, Changhan and others, "Direct Speech-to-Speech Translation with Discrete Units", Annual Meeting of the Association for Computational Linguistics
[^communication2023seamlessm4t]: {Seamless Communication}, "SeamlessM4T: Massively Multilingual and Multimodal Machine Translation", arXiv preprint arXiv:2308.11596
[^communication2023seamless]: {Seamless Communication}, "Seamless: Multilingual Expressive and Streaming Speech Translation", arXiv preprint arXiv:2312.05187
[^ma2019stacl]: Ma, Mingbo and Huang, Liang and others, "STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency", Annual Meeting of the Association for Computational Linguistics
