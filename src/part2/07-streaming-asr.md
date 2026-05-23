# Streaming ASR

## Tổng quan

Streaming ASR (hay Online ASR) là bài toán nhận dạng giọng nói **real-time** - model phải output text **trong khi người dùng đang nói**, không chờ toàn bộ utterance. Đây là yêu cầu bắt buộc cho voice assistants, live captioning, và call center AI.

> **📝 Offline vs Streaming ASR**
>
> - **Offline ASR**: Nhận toàn bộ audio → output transcript. Latency không quan trọng, chỉ cần accuracy.
> - **Streaming ASR**: Nhận audio từng chunk → output **partial results** liên tục. Phải cân bằng **accuracy vs latency**.



## Offline vs Online: Phân tích Chi tiết

### Bảng So sánh

| Tiêu chí | Offline ASR | Streaming ASR |
|----------|------------|---------------|
| Input | Toàn bộ utterance | Từng chunk (10-640ms) |
| Attention | Bidirectional (full context) | Causal / limited context |
| Latency | Không giới hạn | < 200-500ms (user-perceived) |
| WER | Thấp hơn (có full context) | Cao hơn 5-15% relative |
| Use case | Transcription, subtitles | Voice assistant, live caption |
| Decoding | Beam search (offline) | Greedy / streaming beam |

: So sánh Offline vs Streaming ASR <a id="tbl-offline-vs-streaming"></a>

### Latency Budget

Tổng latency của một voice AI system:

<a id="eq-latency-budget"></a>

$$
L_{\text{total}} = L_{\text{audio}} + L_{\text{endpointing}} + L_{\text{ASR}} + L_{\text{NLU}} + L_{\text{TTS}}
$$

Trong đó:

- $L_{\text{audio}}$: Audio buffering (chunk size) - 80-640ms
- $L_{\text{endpointing}}$: Phát hiện người dùng ngừng nói - 200-800ms
- $L_{\text{ASR}}$: Inference time - 50-200ms
- $L_{\text{NLU}}$: Language understanding - 100-500ms
- $L_{\text{TTS}}$: Tổng hợp giọng nói - 100-300ms

> **⚠️ Latency Target**
>
> Để đạt trải nghiệm tự nhiên, tổng latency từ khi người dùng **ngừng nói** đến khi **nghe phản hồi** phải < **1 giây**. Streaming ASR cần latency < 200ms.



## Kiến trúc Streaming

### CTC-based Streaming

CTC tự nhiên hỗ trợ streaming vì output tại mỗi frame **độc lập** (conditional independence):

<a id="eq-ctc-streaming"></a>

$$
p(\mathbf{y} | \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} p(\pi_t | \mathbf{x}_{\leq t})
$$

Chỉ cần **causal encoder** (không nhìn future frames) là có thể streaming.

### RNN-Transducer Streaming

RNN-T [^graves2012sequence] là kiến trúc streaming phổ biến nhất trong production:

<a id="eq-rnnt-streaming"></a>

$$
p(y_u | \mathbf{x}_{\leq t}, y_{<u}) = \text{JointNet}(\text{Encoder}(\mathbf{x}_{\leq t}), \text{Predictor}(y_{<u}))
$$

- **Encoder**: Causal (chỉ nhìn past + current)
- **Predictor**: Autoregressive trên previous tokens
- **Joint Network**: Kết hợp encoder và predictor

### Chunked Attention

Thay vì full bidirectional attention, chunked attention [^zhang2020streaming] chia audio thành chunks:

<a id="eq-chunked-attention"></a>

$$
\text{Attention}(\mathbf{Q}_c, \mathbf{K}_c, \mathbf{V}_c) = \text{softmax}\left(\frac{\mathbf{Q}_c \mathbf{K}_c^T}{\sqrt{d_k}}\right) \mathbf{V}_c
$$

trong đó $c$ là chunk index, và keys/values chỉ từ:

- Current chunk
- Left context (previous $L$ chunks)
- Optional: Right context (future $R$ frames - thêm latency)

### Dynamic Chunk Training

**Unified model** [^zhang2022wenet] train với **random chunk sizes** để một model duy nhất hoạt động ở nhiều latency modes:

<a id="eq-dynamic-chunk"></a>

$$
\text{chunk\_size} \sim \text{Uniform}(\{1, 2, 4, 8, 16, \infty\}) \times \text{subsampling\_factor}
$$

- chunk_size = $\infty$ → offline mode (full attention)
- chunk_size = 1 → streaming mode (minimum latency)
- chunk_size = 4-8 → balanced mode

```python
#| eval: false
#| code-fold: true
#| code-summary: "Dynamic Chunk Attention Mask"
import torch
from torch import Tensor

def create_chunk_mask(
    seq_len: int,
    chunk_size: int,
    left_context: int = -1,  # -1 means unlimited
) -> Tensor:
    """Create attention mask for chunked streaming ASR.

    Args:
        seq_len: Sequence length T
        chunk_size: Size of each chunk in frames
        left_context: Number of left chunks to attend to (-1 = all)

    Returns:
        mask: [T, T] boolean mask, True = can attend
    """
    mask: Tensor = torch.zeros(
        seq_len, seq_len, dtype=torch.bool
    )  # [T, T] - bool

    for i in range(seq_len):
        chunk_idx: int = i // chunk_size
        # Can attend to frames in same chunk
        chunk_start: int = chunk_idx * chunk_size
        chunk_end: int = min((chunk_idx + 1) * chunk_size, seq_len)
        mask[i, chunk_start:chunk_end] = True

        # Can attend to left context
        if left_context == -1:
            mask[i, :chunk_start] = True  # unlimited left context
        else:
            left_start: int = max(
                0, (chunk_idx - left_context) * chunk_size
            )
            mask[i, left_start:chunk_start] = True

    return mask  # [T, T] - bool

# Example: chunk_size=4, left_context=2
mask = create_chunk_mask(seq_len=16, chunk_size=4, left_context=2)
print(f"Mask shape: {mask.shape}")  # [16, 16]
print(f"Sparsity: {(~mask).float().mean():.1%}")
```

## Endpointing

### Tại sao Endpointing quan trọng?

Endpointing (Voice Activity Detection + End-of-Query detection) quyết định **khi nào người dùng ngừng nói** để trigger response generation:

<a id="eq-endpointing"></a>

$$
\text{endpoint} = \begin{cases}
\text{True} & \text{nếu } \text{silence\_duration} > \tau_{\text{threshold}} \\
\text{False} & \text{ngược lại}
\end{cases}
$$

### 3 Chiến lược Endpointing

1. **VAD-based**: Đơn giản, dùng energy/model-based VAD
   - Silero VAD: ONNX, ~1ms inference
   - Threshold: 300-800ms silence

2. **Token-based**: Dựa trên ASR output
   - Nếu ASR output `<eos>` token → endpoint
   - Hoặc: N consecutive blank tokens trong CTC

3. **Neural Endpointer**: Learned model
   - Input: Audio features + ASR state
   - Output: Probability of end-of-query
   - Google's neural endpointer: 200ms latency reduction

## Streaming ASR trong Industry

### NVIDIA Nemotron ASR

Nemotron ASR [^nvidia2024nemotron] (2024-2025):

- **FastConformer** encoder với cache-based streaming
- Hybrid CTC/RNNT decoding
- **Multi-blank CTC**: Skip blank frames, giảm decoding time
- RTF < 0.01 trên A100

### NVIDIA Canary

Canary [^nvidia2024canary] hỗ trợ multilingual streaming:

- Multi-task: ASR + Translation + Language ID
- Streaming via chunked attention
- 100+ languages

### WeNet Unified Framework

WeNet [^zhang2022wenet] cung cấp **unified streaming/non-streaming ASR**:

- Single model, switchable modes (chunk size = 1 → $\infty$)
- Dynamic chunk training
- Production-ready C++ runtime
- ONNX + TensorRT export

### Google USM (Universal Speech Model)

USM [^zhang2023google] - 2B parameter model:

- Pre-trained trên 12M hours unlabeled audio
- Fine-tuned cho 300+ languages
- Streaming via causal attention + look-ahead

| Engine | Kiến trúc | Streaming | Languages | RTF |
|--------|----------|-----------|-----------|-----|
| Whisper | Encoder-Decoder | Không (offline) | 99 | ~0.1 |
| Nemotron | FastConformer-RNNT | Có | English-focus | <0.01 |
| Canary | Conformer-CTC/RNNT | Có | 100+ | <0.02 |
| WeNet | Conformer-CTC/RNNT | Có (unified) | Configurable | <0.05 |
| USM | Conformer-CTC | Có | 300+ | N/A |

: So sánh các Streaming ASR engines <a id="tbl-streaming-asr-engines"></a>

## Metrics đánh giá Streaming ASR

Ngoài WER (Word Error Rate), streaming ASR cần thêm:

| Metric | Công thức | Ý nghĩa |
|--------|----------|---------|
| **RTF** (Real-Time Factor) | $\frac{T_{\text{process}}}{T_{\text{audio}}}$ | < 1.0 = real-time |
| **Latency** | $T_{\text{first\_token}} - T_{\text{speech\_end}}$ | Thời gian chờ kết quả |
| **EL** (Emission Latency) | Average delay per token | Token-level latency |
| **WER** | $\frac{S + D + I}{N}$ | Accuracy |
| **Partial stability** | % partial results không thay đổi | UX quality |

: Metrics cho Streaming ASR <a id="tbl-streaming-metrics"></a>

## Tóm tắt

1. **Streaming ASR** là yêu cầu bắt buộc cho voice AI production
2. **CTC** tự nhiên hỗ trợ streaming, **RNN-T** là kiến trúc phổ biến nhất
3. **Chunked attention** cho phép trade-off latency vs accuracy
4. **Dynamic chunk training** (WeNet) tạo unified offline/streaming model
5. **Endpointing** quyết định user experience - neural endpointer giảm 200ms latency
6. Industry leaders: NVIDIA (Nemotron, Canary), Google (USM), WeNet (open-source)



---

<!-- References (auto-generated from .bib) -->
[^graves2012sequence]: Graves, Alex, "Sequence Transduction with Recurrent Neural Networks", arXiv preprint arXiv:1211.3711
[^zhang2020streaming]: Zhang, Shiliang and others, "Streaming Transformer ASR with Attention-based Chunk Approaches", IEEE International Conference on Acoustics, Speech and Signal Processing
[^zhang2022wenet]: Zhang, Binbin and Wu, Di and Peng, Zhendong and others, "WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit", Interspeech
[^nvidia2024nemotron]: {NVIDIA}, "Nemotron ASR: State-of-the-Art Speech Recognition", NVIDIA Technical Report
[^nvidia2024canary]: {NVIDIA}, "Canary: Multi-Lingual Multi-Task ASR and Translation Model", NVIDIA NeMo Toolkit Documentation
[^zhang2023google]: Zhang, Yu and Han, Wei and Qin, James and others, "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages", arXiv preprint arXiv:2303.01037
