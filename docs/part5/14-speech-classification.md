# Speech Classification

## Tổng quan

Speech classification bao gồm nhiều bài toán quan trọng: **emotion recognition** (SER), **speaker identification/verification** (SID/SV), **language identification** (LID), **audio event detection** (AED), và **keyword spotting** (KS). Chương này trình bày kiến trúc, loss functions, và best practices cho từng bài toán.

## Speech Emotion Recognition (SER)

### Formulation

Cho audio $\mathbf{x}$ và tập emotion labels $\mathcal{Y} = \{\text{neutral}, \text{happy}, \text{sad}, \text{angry}, \text{fear}, \text{surprise}, \text{disgust}\}$:

<a id="eq-ser-classify"></a>

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} p(y | \mathbf{x}) = \arg\max_{y} \text{softmax}(\mathbf{W} \cdot \text{Pool}(f(\mathbf{x})) + \mathbf{b})
$$

### Kiến trúc phổ biến

| Approach | Model | Input | Accuracy (IEMOCAP) |
|----------|-------|-------|---------------------|
| Handcrafted | SVM + eGeMAPS | Acoustic features | ~58% |
| SSL fine-tune | Wav2Vec 2.0 + head | Raw waveform | ~68% |
| SSL fine-tune | HuBERT + head | Raw waveform | ~70% |
| SSL fine-tune | WavLM + head | Raw waveform | **~72%** |
| Multimodal | Text + Audio fusion | Transcript + waveform | ~74% |

: So sánh approaches cho SER trên IEMOCAP <a id="tbl-ser-comparison"></a>

### SSL-based SER Pipeline

```python
#| eval: false
#| code-fold: true
#| code-summary: "SER with HuBERT"
import torch
import torch.nn as nn
from torch import Tensor

class SERModel(nn.Module):
    """Speech Emotion Recognition with SSL backbone."""

    def __init__(
        self,
        d_model: int = 768,     # HuBERT hidden dim
        n_emotions: int = 7,
        pool: str = "attention", # mean, attention, stats
    ) -> None:
        super().__init__()
        self.pool_type = pool
        if pool == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, 128),      # [B, T, 768] -> [B, T, 128]
                nn.Tanh(),
                nn.Linear(128, 1),            # [B, T, 128] -> [B, T, 1]
            )
        elif pool == "stats":
            d_model = d_model * 2  # mean + std
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),       # [B, D] -> [B, 256]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_emotions),    # [B, 256] -> [B, 7]
        )

    def forward(
        self,
        hidden_states: Tensor,  # [B, T, 768] - from HuBERT
    ) -> Tensor:
        if self.pool_type == "mean":
            pooled: Tensor = hidden_states.mean(dim=1)  # [B, 768]
        elif self.pool_type == "attention":
            weights: Tensor = torch.softmax(
                self.attn_pool(hidden_states), dim=1
            )  # [B, T, 1] - float32
            pooled = (hidden_states * weights).sum(dim=1)  # [B, 768]
        elif self.pool_type == "stats":
            mean: Tensor = hidden_states.mean(dim=1)       # [B, 768]
            std: Tensor = hidden_states.std(dim=1)         # [B, 768]
            pooled = torch.cat([mean, std], dim=-1)        # [B, 1536]
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")

        return self.classifier(pooled)  # [B, n_emotions] - float32
```

### Dimensional Emotion (Valence-Arousal-Dominance)

Ngoài categorical emotions, có thể predict continuous values:

<a id="eq-vad-regression"></a>

$$
(\hat{v}, \hat{a}, \hat{d}) = f(\mathbf{x}), \quad v, a, d \in [-1, 1]
$$

- **Valence**: Positive ↔ Negative
- **Arousal**: High energy ↔ Low energy
- **Dominance**: Dominant ↔ Submissive

Loss function: Concordance Correlation Coefficient (CCC):

<a id="eq-ccc"></a>

$$
\text{CCC}(\hat{y}, y) = \frac{2 \rho \sigma_{\hat{y}} \sigma_y}{\sigma_{\hat{y}}^2 + \sigma_y^2 + (\mu_{\hat{y}} - \mu_y)^2}
$$

## Speaker Identification & Verification

### Speaker Identification (Closed-set)

Phân loại speaker từ tập cố định $N$ speakers:

<a id="eq-sid"></a>

$$
\hat{s} = \arg\max_{s \in [N]} p(s | \mathbf{x}) = \arg\max_s \text{softmax}(\mathbf{W}\mathbf{e} + \mathbf{b})
$$

### Speaker Verification (Open-set)

Xác minh 2 audio có cùng speaker không:

<a id="eq-sv-cosine"></a>

$$
\text{score}(\mathbf{x}_1, \mathbf{x}_2) = \cos(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}
$$

<a id="eq-sv-decision"></a>

$$
\text{decision} = \begin{cases} \text{same speaker} & \text{nếu score} > \tau \\ \text{different} & \text{ngược lại} \end{cases}
$$

### Loss Functions cho Speaker Embedding

**AAM-Softmax** (Additive Angular Margin):

<a id="eq-aam-softmax"></a>

$$
\mathcal{L}_{\text{AAM}} = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
$$

với $m$ là angular margin, $s$ là scale factor.

### ECAPA-TDNN

ECAPA-TDNN [^desplanques2020ecapa] là kiến trúc speaker verification phổ biến nhất:

- **Squeeze-Excitation** blocks (channel attention)
- **Multi-scale features** aggregation
- **Attentive Statistics Pooling**

| Model | Params | EER (VoxCeleb1) | minDCF |
|-------|--------|-----------------|--------|
| x-vector | 4.2M | 3.13% | 0.329 |
| ECAPA-TDNN | 6.2M | 0.87% | 0.106 |
| WavLM + ECAPA | 300M+ | **0.56%** | **0.059** |

: So sánh Speaker Verification models <a id="tbl-sv-comparison"></a>

## Language Identification (LID)

Nhận dạng ngôn ngữ từ audio:

<a id="eq-lid"></a>

$$
\hat{l} = \arg\max_{l \in \mathcal{L}} p(l | \mathbf{x})
$$

### Approaches

1. **i-vector + LDA**: Traditional, dùng GMM-based features
2. **x-vector**: TDNN-based, E2E training
3. **SSL + fine-tune**: Wav2Vec 2.0/XLSR fine-tuned cho LID
4. **Whisper-based**: Dùng Whisper's language detection head

!!! note "LID cho Vietnamese"
    Vietnamese LID cần phân biệt 3 phương ngữ (Bắc, Trung, Nam) - khó hơn nhiều so với language-level ID. SSL models (XLSR) cho kết quả tốt nhất.


## Audio Event Detection

### Sound Event Detection (SED)

Phát hiện và phân loại các sự kiện âm thanh (không phải speech):

<a id="eq-sed"></a>

$$
p(e_k | \mathbf{x}_t) = \sigma(\text{Model}(\mathbf{x})_{t,k}), \quad k \in \{1, \ldots, K\}
$$

### Audio Tagging

Gán nhãn cho toàn bộ audio clip (weak labeling):

<a id="eq-audio-tagging"></a>

$$
\hat{\mathbf{y}} = \sigma(\text{Pool}(\text{Model}(\mathbf{x}))) \in [0, 1]^K
$$

### Pretrained Audio Neural Networks (PANNs)

PANNs [^kong2020panns] sử dụng CNN14 trained trên AudioSet:

| Model | Params | mAP (AudioSet) |
|-------|--------|-----------------|
| CNN14 | 80M | 0.431 |
| HTS-AT | 30M | 0.471 |
| BEATs | 90M | **0.498** |
| Audio-MAE | 86M | 0.473 |

: So sánh Audio Classification models <a id="tbl-audio-classification"></a>

## Keyword Spotting

Phát hiện từ khóa cụ thể (e.g., "Hey Siri", "OK Google"):

### Constraints

- **Tiny model**: < 500KB (chạy trên edge device)
- **Always-on**: Chạy liên tục, power < 1mW
- **Low latency**: < 100ms response time

### Kiến trúc

| Model | Params | Accuracy (GSC v2) | Latency |
|-------|--------|-------------------|---------|
| DS-CNN | 26K | 95.4% | 10ms |
| TC-ResNet | 66K | 96.6% | 15ms |
| Keyword Transformer | 180K | 97.7% | 20ms |
| BC-ResNet | 15K | 97.2% | 8ms |

: Keyword Spotting models cho edge deployment <a id="tbl-kws"></a>

## SUPERB Benchmark

SUPERB [^yang2021superb] đánh giá SSL models trên 10 speech tasks:

| Task | Metric | Best SSL Model | Score |
|------|--------|---------------|-------|
| ASR | WER ↓ | WavLM Large | 3.44% |
| KS | Acc ↑ | HuBERT Large | 98.76% |
| IC | Acc ↑ | WavLM Large | 99.55% |
| SID | Acc ↑ | WavLM Large | 94.90% |
| SV | EER ↓ | WavLM Large | 0.56% |
| ER | Acc ↑ | WavLM Large | 72.48% |
| SD | DER ↓ | WavLM Large | 3.28% |
| SE | PESQ ↑ | - | - |

: SUPERB Benchmark results (WavLM dominates) <a id="tbl-superb-results"></a>

!!! note "WavLM là SSL model tốt nhất cho classification tasks"
    Nhờ denoising objective, WavLM học được representations robust cho noise và speaker variations - rất quan trọng cho SV, ER, SD.


## Tóm tắt

1. **SER**: SSL backbone + attention pooling + classifier. WavLM cho best results.
2. **Speaker Verification**: ECAPA-TDNN + AAM-Softmax loss. Cosine similarity scoring.
3. **LID**: SSL fine-tuning (XLSR) hoặc Whisper language detection.
4. **Audio Event Detection**: PANNs/BEATs cho audio tagging, frame-level models cho SED.
5. **Keyword Spotting**: Tiny models (< 500KB) cho edge deployment.
6. **SUPERB**: WavLM Large là SSL model tốt nhất overall cho speech classification tasks.



---

<!-- References (auto-generated from .bib) -->
[^desplanques2020ecapa]: Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech
[^kong2020panns]: Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and others, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing
[^yang2021superb]: Yang, Shu-wen and Chi, Po-Han and others, "SUPERB: Speech Processing Universal PERformance Benchmark", Interspeech
