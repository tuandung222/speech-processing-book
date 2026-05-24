# Chương 12: Multimodal Omni Models

## Vì sao chương này quan trọng

Multimodal Omni models là một trong các xu hướng lớn nhất của AI giai đoạn 2024-2026: một model duy nhất xử lý đồng thời **text, audio, image, video**, vừa hiểu vừa sinh ở cả bốn modality. Khác với cascaded pipeline truyền thống (ASR → LLM → TTS) hay multimodal LLM thế hệ đầu (chỉ hỗ trợ một-hai modality), Omni models hợp nhất mọi modality vào một forward pass duy nhất, với latency thấp và khả năng cross-modal reasoning sâu hơn.

Đối với người làm NLP/LLM, Omni model là "GPT-4 nhưng với speech và video native". Đối với engineer voice AI, Omni model thay thế nhiều thành phần của pipeline cascaded bằng một mô hình duy nhất, đơn giản hoá triển khai nhưng đặt ra thách thức mới về cost, observability, và customization.

Chương này phân tích các kiến trúc tiêu biểu theo dòng thời gian: Qwen2.5-Omni (Thinker-Talker pattern), GPT-Realtime (OpenAI, GA tháng 8/2025), Gemini Live (Google), Qwen3-Omni (Alibaba, MoE Thinker-Talker, tháng 10/2025), và Qwen3.5-Omni Plus (tháng 3/2026, được công bố là vượt Gemini 3.1 Pro trên một số benchmark).

> **Cấu trúc chương**
>
> - **Phần 1**: phân loại kiến trúc multimodal: cascaded, early fusion, omni.
> - **Phần 2**: bốn cách encode audio cho LLM (spectrogram + adapter, codec tokens, SSL features, Thinker-Talker).
> - **Phần 3**: Qwen2-Audio và Qwen2.5-Omni, Thinker-Talker pattern.
> - **Phần 4**: GPT-Realtime, Gemini Live, các Omni model closed-source frontier.
> - **Phần 5**: Qwen3-Omni và Qwen3.5-Omni, open-source frontier 2025-2026.
> - **Phần 6**: so sánh tổng hợp và xu hướng 2026-2027.

## Tổng quan

Multimodal Omni models là xu hướng lớn nhất trong AI 2024-2026: một model duy nhất xử lý **text, audio, image, video** đồng thời. Chương này phân tích các kiến trúc tiêu biểu: **Qwen2.5-Omni**, **GPT-Realtime**, **Gemini Live**, **Qwen3-Omni**, và **Qwen3.5-Omni**.

> **📝 Omni = Tất cả trong Một**
>
> Thay vì pipeline riêng biệt (ASR → LLM → TTS), Omni models nhận **bất kỳ modality nào** làm input và sinh ra **bất kỳ modality nào** làm output, trong một forward pass duy nhất.



## Phân loại Kiến trúc Multimodal

### Cascaded vs End-to-End

| Approach | Pipeline | Ưu điểm | Hạn chế |
|----------|---------|---------|---------|
| **Cascaded** | ASR → LLM → TTS | Modular, dễ debug | Latency cao, error propagation |
| **Early Fusion** | Audio + Text → Shared model | Ít error propagation | Training phức tạp |
| **Omni (E2E)** | Any-to-Any trong 1 model | Lowest latency, unified | Cần data đa dạng, khó train |

: Phân loại kiến trúc multimodal <a id="tbl-multimodal-types"></a>

### Các Cách Encode Audio cho LLM

1. **Spectrogram + Audio Encoder**: Whisper encoder → adapter → LLM (Qwen2-Audio)
2. **Discrete Codec Tokens**: EnCodec/Mimi tokens interleave với text tokens (Moshi, AudioLM)
3. **Continuous Audio Features**: SSL features (HuBERT) → projector → LLM (SALMONN)
4. **Thinker-Talker**: Separate thinking và speaking streams (Qwen2.5-Omni)

## Qwen2-Audio

Qwen2-Audio [^chu2024qwen2audio] từ Alibaba:

### Kiến trúc

- **Audio Encoder**: Whisper-large-v3 encoder (frozen hoặc fine-tuned)
- **Audio Adapter**: Linear projection + downsampling
- **LLM**: Qwen2-7B

<a id="eq-qwen2audio-encode"></a>

$$
\mathbf{h}_{\text{audio}} = \text{Adapter}(\text{WhisperEncoder}(\mathbf{x}_{\text{audio}}))
$$

<a id="eq-qwen2audio-concat"></a>

$$
\mathbf{h}_{\text{input}} = [\mathbf{h}_{\text{text}}; \mathbf{h}_{\text{audio}}; \mathbf{h}_{\text{text}}]
$$

### Capabilities

- Audio understanding (ASR, translation, emotion, events)
- Audio-grounded QA
- Multi-turn audio dialogue
- Không generate audio (chỉ text output)

## Qwen2.5-Omni (Thinker-Talker)

Qwen2.5-Omni [^xu2025qwen25omni] là bước tiến lớn với kiến trúc **Thinker-Talker**:

### Thinker Module

- Nhận **tất cả modalities** (text, audio, image, video)
- Xử lý reasoning, understanding
- Output: Hidden states cho Talker

### Talker Module

- Nhận hidden states từ Thinker
- Generate **streaming speech tokens** song song với text tokens
- Sử dụng **dual-track autoregressive generation**:

<a id="eq-thinker-talker"></a>

$$
p(y_t^{\text{text}}, y_t^{\text{speech}} | \mathbf{h}_{\text{thinker}}, y_{<t}) = p(y_t^{\text{text}} | \mathbf{h}_t) \cdot p(y_t^{\text{speech}} | \mathbf{h}_t, y_t^{\text{text}})
$$

### TMRoPE (Time-aligned Multi-Resolution RoPE)

Để đồng bộ audio và video ở different frame rates:

<a id="eq-tmrope"></a>

$$
\text{RoPE}_{\text{TMR}}(\mathbf{x}, t) = \text{RoPE}(\mathbf{x}, t_{\text{aligned}})
$$

trong đó $t_{\text{aligned}}$ được tính dựa trên timestamps thực tế, không phải token positions.

> **📝 Tại sao Thinker-Talker?**
>
> Tách biệt "suy nghĩ" (Thinker) và "nói" (Talker) cho phép:
>
> 1. Thinker có thể reasoning phức tạp mà không bị constrained bởi real-time speech generation
> 2. Talker generate speech **streaming** ngay khi Thinker bắt đầu output
> 3. Giảm latency so với sequential: think → speak



```python
#| eval: false
#| code-fold: true
#| code-summary: "Thinker-Talker Dual Generation (Simplified)"
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class ThinkerTalker(nn.Module):
    """Simplified Thinker-Talker architecture."""

    def __init__(
        self,
        d_model: int = 4096,
        text_vocab: int = 151936,
        speech_codebook: int = 4096,
        n_layers_thinker: int = 32,
        n_layers_talker: int = 8,
    ) -> None:
        super().__init__()
        # Thinker: full multimodal reasoning
        self.thinker = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=32, dim_feedforward=11008,
                batch_first=True
            ),
            num_layers=n_layers_thinker,
        )
        # Talker: lightweight speech generation
        self.talker = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=32, dim_feedforward=11008,
                batch_first=True
            ),
            num_layers=n_layers_talker,
        )
        self.text_head = nn.Linear(d_model, text_vocab)
        self.speech_head = nn.Linear(d_model, speech_codebook)

    def forward(
        self,
        multimodal_input: Tensor,  # [B, T, D] - mixed modalities
    ) -> Tuple[Tensor, Tensor]:
        # Thinker processes everything
        h_think: Tensor = self.thinker(
            multimodal_input
        )  # [B, T, D] - float32

        # Text generation from Thinker
        text_logits: Tensor = self.text_head(
            h_think
        )  # [B, T, text_vocab] - float32

        # Talker generates speech conditioned on Thinker
        h_talk: Tensor = self.talker(
            h_think
        )  # [B, T, D] - float32

        speech_logits: Tensor = self.speech_head(
            h_talk
        )  # [B, T, speech_codebook] - float32

        return text_logits, speech_logits
```

## GPT-Realtime (GA August 2025)

GPT-Realtime là model speech-to-speech tiên tiến nhất của OpenAI tại thời điểm GA tháng 8/2025, kế thừa GPT-4o Voice Mode (2024) và Advanced Voice Mode. Khác với pipeline cascaded, GPT-Realtime là một model duy nhất xử lý audio in và audio out trong cùng forward pass.

### Đặc điểm chính (theo public docs OpenAI tháng 8/2025)

- **Native multimodal**: audio input và output tích hợp trong cùng model, không qua bước transcription rõ ràng.
- **Real-time voice**: target sub-500 ms first-byte latency trên mạng tốt.
- **MCP server support**: tích hợp tool calling qua Model Context Protocol.
- **Image input**: nhận hình ảnh kèm voice trong cuộc hội thoại.
- **SIP phone calling**: cho phép trigger qua telephony stack tiêu chuẩn.
- **Code-switching và multilingual**: chuyển ngôn ngữ tự nhiên trong cùng câu trả lời.

### Pricing (tham khảo public pricing OpenAI tháng 8/2025)

- Audio input: khoảng 0.03 USD per minute.
- Audio output: khoảng 0.06 USD per minute.
- Tổng cho voice agent điển hình: 0.05-0.10 USD per minute conversation, phụ thuộc tỷ lệ user nói và AI nói.

### Hạn chế công khai

- Kiến trúc chi tiết và số tham số không được công bố.
- Không open-source.
- Một số báo cáo cộng đồng cho thấy voice output đôi khi "too expressive" hoặc hallucinate âm thanh không phải lời nói trong audio output.

### Basic Voice Mode retirement

Tháng 9/2025, OpenAI sunset Basic Voice Mode (cascaded ASR + LLM + TTS), thống nhất mọi voice feature dưới Realtime API.

## Gemini Live (Google)

Gemini Live là dòng multimodal omni của Google, kế thừa Gemini multimodal:

- **Gemini 2.0 Live (2025)**: streaming voice và video, tích hợp với Google Workspace và Vertex AI.
- **Gemini 3 Live (2026)**: cải tiến translation, multi-speaker handling, latency thấp hơn.

Đặc điểm chung:

- Native multimodal from scratch (train đồng thời trên text, images, audio, video).
- Long context: xử lý audio và video hàng giờ.
- Closed-source, available qua Vertex AI và Gemini API.

## Qwen3-Omni (October 2025)

Qwen3-Omni là milestone quan trọng của open-source Speech LLM. Đây là Omni model dựa trên **MoE (Mixture-of-Experts) Thinker-Talker**, kế thừa và mở rộng pattern Thinker-Talker của Qwen2.5-Omni với capacity và đa ngôn ngữ lớn hơn nhiều.

### Variants

- **Qwen3-Omni-30B-A3B-Instruct**: text + audio + video input, text + audio output. Tổng 30B parameters với 3B active per forward pass (MoE).
- **Qwen3-Omni-30B-A3B-Thinking**: hỗ trợ chain-of-thought reasoning trên multimodal input.
- **Qwen3-Omni-30B-A3B-Captioner**: chuyên cho audio captioning.

### Capabilities

- 119 ngôn ngữ text input.
- 19 ngôn ngữ speech input.
- 10 ngôn ngữ speech output.
- Apache 2.0 license, fully open-source.

### Performance (theo Qwen3-Omni Technical Report, tháng 10/2025)

- Theo technical report, đạt kết quả dẫn đầu trên 22 trong 36 audio và audio-video benchmarks được báo cáo.
- Theo technical report, đạt kết quả dẫn đầu trong nhóm open-source trên 32 trong 36 benchmark được báo cáo.
- ASR, audio understanding, và voice conversation comparable với Gemini 2.5 Pro trên nhiều benchmark.

### Qwen3-Omni-Flash (December 2025)

Variant optimised cho inference latency, hướng tới production voice agent.

## Qwen3.5-Omni (March 2026)

Qwen3.5-Omni Plus (release tháng 3/2026) là phiên bản kế thừa Qwen3-Omni với capacity lớn hơn và benchmark cao hơn.

### Đặc điểm

- Native multimodal: text, image, audio, video xử lý trong single forward pass.
- Streaming speech output realtime.
- Hỗ trợ chain-of-thought reasoning trên multimodal input.

### Performance (theo Qwen3.5-Omni Technical Report, tháng 3/2026)

- Theo technical report, đạt nhiều kết quả dẫn đầu trên các benchmark audio, audio-video understanding, reasoning và interaction.
- Theo công bố chính thức, Qwen3.5-Omni Plus so sánh thuận lợi với Gemini 3.1 Pro trên một số benchmark general audio understanding, reasoning và translation. Cần đọc kết quả này theo đúng protocol benchmark và thời điểm công bố.

## So sánh tổng hợp (mid-2026)

| Model | Tổ chức | Năm | Audio In | Audio Out | Real-time | Open-source |
|---|---|---|---|---|---|---|
| Qwen2-Audio | Alibaba | 2024 | Có | Không | Không | Có |
| Qwen2.5-Omni | Alibaba | 2025 | Có | Có | Có | Có |
| Qwen3-Omni 30B-A3B | Alibaba | Oct 2025 | Có | Có | Có | Có |
| Qwen3-Omni-Flash | Alibaba | Dec 2025 | Có | Có | Có | Có |
| Qwen3.5-Omni Plus | Alibaba | Mar 2026 | Có | Có | Có | Có |
| GPT-4o Voice Mode | OpenAI | 2024 | Có | Có | Có | Không |
| GPT-Realtime | OpenAI | Aug 2025 | Có | Có | Có | Không |
| Gemini 2.0 Live | Google | 2025 | Có | Có | Có | Không |
| Gemini 3 Live | Google | 2026 | Có | Có | Có | Không |
| SALMONN | ByteDance | 2024 | Có | Không | Không | Có |
| Moshi | Kyutai | Sep 2024 | Có | Có | Có | Có |
| Moshi v2 + MoshiRAG | Kyutai | Apr 2026 | Có | Có | Có | Có |

: So sánh các Multimodal Omni Models <a id="tbl-omni-comparison"></a>

## Xu hướng 2025-2027

Dựa trên các release giữa 2024 và đầu 2026, có thể nhận diện sáu xu hướng chính:

1. **MoE Thinker-Talker** trở thành kiến trúc phổ biến nhất cho Omni model quy mô lớn, cho phép cân bằng giữa capacity và inference cost.
2. **Native multimodal converging**: các bộ encoder riêng cho audio, image, video dần được thay bằng unified transformer xử lý cùng lúc.
3. **Open-source bắt kịp closed-source trên một số benchmark**: Qwen3.5-Omni Plus được công bố so sánh thuận lợi với Gemini 3.1 Pro ở một số benchmark, nhưng vẫn cần đánh giá lại theo workload cụ thể.
4. **Latency target sub-300 ms first byte**: tất cả các Omni model production đều nhắm mức này để đối thoại tự nhiên.
5. **Tool calling tích hợp**: GPT-Realtime hỗ trợ MCP, Qwen3-Omni hỗ trợ function calling natively, cho phép Omni model điều khiển hành động thực tế.
6. **RAG cho speech**: MoshiRAG (tháng 4/2026) cho thấy retrieval-augmented speech LM trở thành paradigm mới, giải quyết hạn chế về kiến thức của Speech LLM thuần.

## Tóm tắt

1. **Omni models** xử lý nhiều modality trong một model duy nhất, từ understanding đến generation.
2. **Thinker-Talker pattern** (Qwen2.5-Omni, Qwen3-Omni) cho phép tách reasoning và speech generation, giảm latency.
3. **GPT-Realtime** (Aug 2025) và **Gemini 3 Live** (2026) đại diện cho nhóm closed-source frontier trong voice agent production.
4. **Qwen3-Omni** và **Qwen3.5-Omni Plus** là nhóm open-source frontier ở thời điểm viết, với nhiều kết quả benchmark công khai mạnh nhưng vẫn cần kiểm chứng theo domain triển khai.
5. **Xu hướng**: MoE, streaming, tool calling, RAG, multilingual.



---

<!-- References (auto-generated from .bib) -->
[^chu2024qwen2audio]: Chu, Yunfei and Xu, Jin and Yang, Qian and others, "Qwen2-Audio Technical Report", arXiv preprint arXiv:2407.10759
[^xu2025qwen25omni]: Xu, Jin and Chu, Yunfei and others, "Qwen2.5-Omni Technical Report", arXiv preprint arXiv:2503.20215
[^openai2024gpt4o]: {OpenAI}, "GPT-4o System Card", OpenAI Technical Report
[^team2024gemini]: {Gemini Team}, "Gemini: A Family of Highly Capable Multimodal Models", arXiv preprint arXiv:2312.11805
