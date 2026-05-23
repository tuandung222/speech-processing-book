# Speech Pretraining & Representation Learning

## Tổng quan

Trước khi đi sâu vào các bài toán cụ thể như ASR hay TTS, chúng ta cần hiểu **cách biểu diễn speech** (speech representations) - nền tảng của mọi hệ thống speech AI hiện đại. Chương này trình bày ba paradigm chính: **self-supervised learning (SSL)**, **contrastive learning**, và **neural codec tokenization**.

## Self-Supervised Speech Learning

### Từ NLP đến Speech SSL

Trong NLP, BERT [^devlin2019bert] đã cách mạng hóa representation learning bằng masked language modeling. Tương tự, speech SSL học representations từ **unlabeled audio** - nguồn tài nguyên dồi dào hơn labeled data rất nhiều.

!!! note "Core Insight"
    Speech SSL cho phep học representations từ hang tram nghin gio audio khong can transcript - chi can audio thoi.


### Wav2Vec 2.0

Wav2Vec 2.0 [^baevski2020wav2vec] là mô hình SSL tiên phong, kết hợp **contrastive learning** với **masked prediction**:

**Architecture:**

1. **Feature Encoder** (CNN): Raw waveform $\mathbf{x} \in \mathbb{R}^T$ được encode thành latent representations $\mathbf{z}_1, \ldots, \mathbf{z}_L$

<a id="eq-wav2vec-encoder"></a>

$$
\mathbf{z} = f_{\text{CNN}}(\mathbf{x}), \quad \mathbf{z} \in \mathbb{R}^{L \times d}
$$

2. **Quantization Module**: Discretize $\mathbf{z}$ thành $\mathbf{q}$ qua product quantization với $G$ codebook groups, mỗi group có $V$ entries:

<a id="eq-wav2vec-quant"></a>

$$
\mathbf{q}_t = \text{argmin}_{v \in [V], g \in [G]} \| \mathbf{z}_t - \mathbf{e}_{g,v} \|^2
$$

3. **Transformer Encoder**: Masked latent $\tilde{\mathbf{z}}$ được đưa vào Transformer để tạo context representations $\mathbf{c}_t$

**Training Objective - Contrastive Loss:**

<a id="eq-wav2vec-loss"></a>

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{c}_t, \mathbf{q}_t) / \kappa)}{\sum_{\tilde{\mathbf{q}} \in \mathcal{Q}_t} \exp(\text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}}) / \kappa)}
$$

trong đó $\kappa$ là temperature, $\mathcal{Q}_t$ gồm 1 positive và $K$ distractors (negative samples).

**Diversity Loss** để tránh codebook collapse:

<a id="eq-wav2vec-diversity"></a>

$$
\mathcal{L}_{\text{diversity}} = \frac{1}{GV} \sum_{g=1}^{G} H(\bar{p}_g) = -\frac{1}{GV} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar{p}_{g,v} \log \bar{p}_{g,v}
$$

**Total loss**: $\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \alpha \mathcal{L}_{\text{diversity}}$

```python
#| eval: false
#| code-fold: true
#| code-summary: "Wav2Vec 2.0 forward pass demo"
import torch
import torch.nn as nn
from typing import Tuple

class Wav2Vec2ForwardDemo(nn.Module):
    """Simplified Wav2Vec 2.0 forward pass."""

    def __init__(
        self,
        d_model: int = 768,
        n_codebooks: int = 2,
        codebook_size: int = 320,
    ) -> None:
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, 10, stride=5),   # [B,1,T] -> [B,512,L]
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2),  # [B,512,L] -> [B,512,L']
            nn.GELU(),
            nn.Conv1d(512, d_model, 3, stride=2),  # -> [B,d_model,L'']
        )
        self.quantizer = nn.Linear(d_model, n_codebooks * codebook_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=12, dim_feedforward=3072,
                batch_first=True
            ),
            num_layers=12
        )

    def forward(
        self, waveform: torch.Tensor  # [B, T] - torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.feature_encoder(
            waveform.unsqueeze(1)  # [B, 1, T] - torch.float32
        )  # [B, d_model, L] - torch.float32
        z = z.transpose(1, 2)  # [B, L, d_model] - torch.float32
        c = self.transformer(z)  # [B, L, d_model] - torch.float32
        return c, z

# Demo
model = Wav2Vec2ForwardDemo()
x = torch.randn(2, 16000)  # [2, 16000] - 1 sec audio
c, z = model(x)
print(f"Input: {x.shape}")  # [2, 16000]
print(f"Latent z: {z.shape}")  # [2, L, 768]
print(f"Context c: {c.shape}")  # [2, L, 768]
```

### HuBERT

HuBERT [^hsu2021hubert] thay thế contrastive loss bằng **masked prediction** với pseudo-labels:

**Training Procedure (Iterative):**

1. **Iteration 0**: Cluster MFCC features bằng k-means $\rightarrow$ pseudo-labels $\hat{y}_t$
2. **Iteration k**: Train model với masked prediction loss, sau đó cluster hidden features của layer $l$ $\rightarrow$ new pseudo-labels

**Masked Prediction Loss:**

<a id="eq-hubert-loss"></a>

$$
\mathcal{L}_{\text{HuBERT}} = \sum_{t \in \mathcal{M}} -\log p(c_t = \hat{y}_t \mid \tilde{\mathbf{X}})
$$

trong đó $\mathcal{M}$ là tập các masked positions và $\hat{y}_t$ là pseudo-label.

!!! note "HuBERT vs Wav2Vec 2.0"
    | | Wav2Vec 2.0 | HuBERT |
    |---|---|---|
    | **Objective** | Contrastive | Masked prediction |
    | **Target** | Quantized latent | Pseudo-label (k-means) |
    | **Negative samples** | Cần | Không cần |
    | **Iterative** | Không | Có (re-cluster) |
    | **Performance** | Mạnh | **Mạnh hơn** (iteration 2+) |


### WavLM

WavLM [^chen2022wavlm] mở rộng HuBERT với **denoising objective** - train trên cả clean và noisy/overlapping speech:

<a id="eq-wavlm-loss"></a>

$$
\mathcal{L}_{\text{WavLM}} = \mathcal{L}_{\text{masked}} + \lambda \mathcal{L}_{\text{denoising}}
$$

Điều này giúp WavLM học representations **robust với noise** và **hiểu speaker overlap** - quan trọng cho speaker diarization va separation.

### data2vec

data2vec [^baevski2022data2vec] tiến tới **modality-agnostic SSL**:

- **Teacher**: EMA của student model, predict **contextualized representations** (không chỉ tokens)
- **Student**: Predict teacher's output tại masked positions
- **Loss**: Smooth L1 giữa student và teacher representations

<a id="eq-data2vec-loss"></a>

$$
\mathcal{L}_{\text{data2vec}} = \sum_{t \in \mathcal{M}} \text{SmoothL1}(\mathbf{c}_t^{\text{student}}, \mathbf{c}_t^{\text{teacher}})
$$

### BEST-RQ

BEST-RQ [^chiu2022selfsupervised] đơn giản hóa quantization bằng **random projection**:

- Không cần learned codebook
- Project features bằng random matrix, sau đó nearest-neighbor lookup
- Performance tương đương Wav2Vec 2.0 nhưng đơn giản hơn nhiều

## Contrastive Learning for Audio (CLAP)

### CLAP Architecture

CLAP (Contrastive Language-Audio Pretraining) [^elizalde2023clap] là phiên bản audio của CLIP [^radford2021clip]:

**Architecture:**

- **Audio Encoder** $f_a$: CNN hoặc Audio Transformer (HTS-AT)
- **Text Encoder** $f_t$: BERT hoặc RoBERTa
- **Projection**: Linear layers project cả 2 vào chung embedding space

**Contrastive Loss (InfoNCE):**

<a id="eq-clap-loss"></a>

$$
\mathcal{L}_{\text{CLAP}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\mathbf{a}_i^\top \mathbf{t}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{a}_i^\top \mathbf{t}_j / \tau)} + \log \frac{\exp(\mathbf{t}_i^\top \mathbf{a}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{t}_i^\top \mathbf{a}_j / \tau)} \right]
$$

trong đó $\mathbf{a}_i = f_a(\text{audio}_i)$, $\mathbf{t}_i = f_t(\text{text}_i)$, $\tau$ là learnable temperature.

### Ứng dụng của CLAP

| Ứng dụng | Mô tả |
|----------|-------|
| Zero-shot audio classification | Classify audio bằng text prompts |
| Audio retrieval | Tìm audio từ text query |
| Audio captioning | Describe audio content |
| Sound event detection | Detect events với text anchors |
| TTS evaluation | Đánh giá naturalness qua audio-text alignment |

### Audio-Text Datasets cho Contrastive Learning

| Dataset | Số cặp audio-text | Nguồn |
|---------|-------------------|-------|
| AudioCaps [^kim2019audiocaps] | 50K | YouTube |
| Clotho [^drossos2020clotho] | 5K | Freesound |
| WavCaps [^mei2024wavcaps] | 400K | Mixed |
| AudioSet [^gemmeke2017audioset] | 2M | YouTube (weak labels) |
| LAION-Audio-630K [^wu2023large] | 630K | Web crawl |

## Các Objective Pretraining Truyền thống

### Tổng hợp các Paradigm

| Paradigm | Mô hình | Mechanism | Ưu điểm |
|----------|---------|-----------|---------|
| **Masked Prediction** | HuBERT, data2vec | Mask frames, predict target | Học local + global context |
| **Contrastive** | Wav2Vec 2.0, CLAP | Positive/negative pairs | Discriminative representations |
| **Autoregressive** | AudioLM, VALL-E | Predict next token | Natural cho generation |
| **Denoising** | WavLM | Reconstruct clean từ noisy | Robust representations |
| **Multi-task** | UniSpeech-SAT [^chen2022unispeech] | SSL + speaker labels | Task-aware features |

### Autoregressive Pretraining

**AudioLM** [^borsos2023audiolm] và **VALL-E** [^wang2023neural] sử dụng autoregressive modeling trên codec tokens:

<a id="eq-ar-pretrain"></a>

$$
p(\mathbf{c}_{1:T}) = \prod_{t=1}^{T} p(c_t \mid c_1, \ldots, c_{t-1})
$$

Đây chính là **language modeling on audio tokens** - cầu nối trực tiếp giữa NLP và speech.

### Multi-task Pretraining

**UniSpeech-SAT** [^chen2022unispeech] kết hợp SSL với speaker supervision:

<a id="eq-unispeech-loss"></a>

$$
\mathcal{L} = \mathcal{L}_{\text{SSL}} + \alpha \mathcal{L}_{\text{speaker}} + \beta \mathcal{L}_{\text{utterance}}
$$

## So sánh Speech Representation Models

| Model | Params | Pre-train Data | SSL Objective | Downstream |
|-------|--------|----------------|---------------|------------|
| Wav2Vec 2.0 Large | 317M | 60K hrs (LS) | Contrastive | ASR, classification |
| HuBERT Large | 317M | 60K hrs (LS) | Masked prediction | ASR, classification |
| WavLM Large | 317M | 94K hrs (LS+VoxPopuli) | Masked + denoising | ASR, separation, diarization |
| data2vec 2.0 | 317M | 60K hrs (LS) | Teacher-student | ASR, classification |
| CLAP | ~200M | 630K audio-text pairs | Contrastive (audio-text) | Zero-shot classification |
| Whisper Large-v3 | 1.5B | 5M hrs (weak supervised) | Seq2seq (supervised) | ASR, translation |

: So sánh các mô hình speech representation <a id="tbl-ssl-comparison"></a>

## SUPERB Benchmark

**SUPERB** (Speech processing Universal PERformance Benchmark) [^yang2021superb] là benchmark chuẩn để đánh giá speech representations trên 10 tasks:

| Task | Metric | Description |
|------|--------|-------------|
| ASR | WER | Automatic speech recognition |
| KS | Accuracy | Keyword spotting |
| QbE | MTWV | Query by example |
| IC | Accuracy | Intent classification |
| SF | F1 + CER | Slot filling |
| SID | Accuracy | Speaker identification |
| SV | EER | Speaker verification |
| SD | DER | Speaker diarization |
| ER | Accuracy | Emotion recognition |
| SE | PESQ + STOI | Speech enhancement |

: SUPERB benchmark tasks <a id="tbl-superb-tasks"></a>

## Tóm tắt

1. **Self-supervised learning** là nền tảng của speech AI hiện đại - học từ unlabeled audio
2. **Wav2Vec 2.0** dùng contrastive learning, **HuBERT** dung masked prediction với pseudo-labels
3. **WavLM** thêm denoising objective cho robust representations
4. **CLAP** mở rộng contrastive learning sang audio-text pairs (tương tự CLIP)
5. **Autoregressive pretraining** trên codec tokens là cầu nối trực tiếp đến LLM
6. **SUPERB** là benchmark chuẩn để so sánh speech representations

Chương tiếp theo sẽ đi vào **ASR Foundations** - áp dụng các representations này vào bài toán nhận dạng giọng nói.



---

<!-- References (auto-generated from .bib) -->
[^devlin2019bert]: Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", North American Chapter of the Association for Computational Linguistics
[^baevski2020wav2vec]: Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", Advances in Neural Information Processing Systems
[^hsu2021hubert]: Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman, "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units", IEEE/ACM Transactions on Audio, Speech, and Language Processing
[^chen2022wavlm]: Chen, Sanyuan and Wang, Chengyi and others, "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", IEEE Journal of Selected Topics in Signal Processing
[^baevski2022data2vec]: Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael, "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language", International Conference on Machine Learning
[^chiu2022selfsupervised]: Chiu, Chung-Cheng and Qin, James and Zhang, Yu and Yu, Jiahui and Wu, Yonghui, "Self-supervised Learning with Random-projection Quantizer for Speech Recognition", International Conference on Machine Learning
[^elizalde2023clap]: Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming, "CLAP: Learning Audio Concepts from Natural Language Supervision", IEEE International Conference on Acoustics, Speech and Signal Processing
[^radford2021clip]: Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others, "Learning Transferable Visual Models From Natural Language Supervision", International Conference on Machine Learning
[^kim2019audiocaps]: Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee, "AudioCaps: Generating Captions for Audios in the Wild", North American Chapter of the Association for Computational Linguistics
[^drossos2020clotho]: Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas, "Clotho: An Audio Captioning Dataset", IEEE International Conference on Acoustics, Speech and Signal Processing
[^mei2024wavcaps]: Mei, Xinhao and others, "WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset", arXiv preprint arXiv:2303.17395
[^gemmeke2017audioset]: Gemmeke, Jort F and Ellis, Daniel PW and others, "Audio Set: An Ontology and Human-Labeled Dataset for Audio Events", IEEE International Conference on Acoustics, Speech and Signal Processing
[^wu2023large]: Wu, Yusong and Chen, Ke and others, "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation", IEEE International Conference on Acoustics, Speech and Signal Processing
[^chen2022unispeech]: Chen, Sanyuan and Wu, Yu and others, "UniSpeech-SAT: Universal Speech Representation Learning with Speaker Aware Pre-Training", IEEE International Conference on Acoustics, Speech and Signal Processing
[^borsos2023audiolm]: Borsos, Zal{\'a}n and Marinier, Rapha{\"e}l and others, "AudioLM: A Language Modeling Approach to Audio Generation", IEEE/ACM Transactions on Audio, Speech, and Language Processing
[^wang2023neural]: Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and He, Lei and Zhao, Sheng and Wei, Furu, "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers", arXiv preprint arXiv:2301.02111
[^yang2021superb]: Yang, Shu-wen and Chi, Po-Han and others, "SUPERB: Speech Processing Universal PERformance Benchmark", Interspeech
