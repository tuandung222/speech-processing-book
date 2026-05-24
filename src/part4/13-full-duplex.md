# Chương 13: Full-Duplex Dialogue

## Vì sao chương này quan trọng

Cuộc đối thoại tự nhiên giữa người với người không phải tuần tự "tôi nói xong, đến lượt bạn". Hai bên thường xuyên overlap: backchannel ngắn (vâng, ừm, à), ngắt lời để bổ sung, đồng thời nghe và phản hồi. Voice AI thế hệ cũ (Siri, Alexa) chỉ hỗ trợ half-duplex: nghe, xử lý, trả lời, không thể bị ngắt giữa chừng. Trải nghiệm này khác xa đối thoại người với người.

Full-duplex dialogue là khả năng **nói và nghe đồng thời**, đặc trưng kỹ thuật khó nhất của Speech AI hiện đại. Moshi (Kyutai, tháng 9/2024) là model open-source đầu tiên hỗ trợ full-duplex thực sự, đạt latency dưới 300 ms với dual-stream architecture. GPT-Realtime (OpenAI, tháng 8/2025) và Gemini Live (Google) cũng hỗ trợ full-duplex, dù kiến trúc chi tiết không được công bố.

Chương này phân tích các thành phần kỹ thuật của full-duplex dialogue: dual-stream tokenization, turn-taking model, interruption handling, backchannel generation, và các metric đánh giá đối thoại.

> **Cấu trúc chương**
>
> - **Phần 1**: half-duplex vs full-duplex, vì sao full-duplex khó.
> - **Phần 2**: Moshi, kiến trúc dual-stream và Depth Transformer.
> - **Phần 3**: turn-taking, interruption, backchannel.
> - **Phần 4**: MoshiRAG (tháng 4/2026), retrieval-augmented full-duplex.
> - **Phần 5**: đánh giá đối thoại, latency, safety, observability.

## Tổng quan

Full-duplex dialogue là khả năng **nói và nghe đồng thời**, giống cuộc trò chuyện tự nhiên giữa người với người. Đây là thách thức lớn nhất trong voice AI, đòi hỏi model phải xử lý **real-time turn-taking**, **interruptions**, và **backchannel signals** (ừ, vâng, à).

> **📝 Half-Duplex vs Full-Duplex**
>
> - **Half-duplex** (Siri, Alexa): Nghe → Xử lý → Nói. Không thể bị interrupt.
> - **Full-duplex** (Moshi, GPT-4o): Nghe VÀ nói đồng thời. Có thể bị interrupt, phản hồi overlap.



## Moshi: Kiến trúc Full-Duplex

Moshi [^defossez2024moshi] từ Kyutai là model full-duplex đầu tiên được open-source:

### Dual-Stream Architecture

Moshi xử lý **hai audio streams song song**:

- **User stream**: Audio từ microphone (input)
- **Model stream**: Audio do model generate (output)

<a id="eq-moshi-dual"></a>

$$
p(\mathbf{a}_t^{\text{model}} | \mathbf{a}_{<t}^{\text{user}}, \mathbf{a}_{<t}^{\text{model}}, \mathbf{c}_{<t})
$$

trong đó $\mathbf{a}_t$ là audio tokens tại time step $t$, $\mathbf{c}_t$ là inner text tokens.

### Inner Monologue

Moshi sử dụng **inner monologue** - text tokens nội bộ giúp model "suy nghĩ":

<a id="eq-moshi-stream"></a>

$$
\text{Stream} = [\underbrace{\mathbf{c}_t}_{\text{inner text}}, \underbrace{\mathbf{a}_t^{(1:Q_{\text{user}})}}_{\text{user audio}}, \underbrace{\mathbf{a}_t^{(1:Q_{\text{model}})}}_{\text{model audio}}]
$$

### Mimi Audio Codec

Moshi sử dụng **Mimi** codec riêng (không dùng EnCodec):

- 12.5 fps (80ms per frame) - thấp hơn EnCodec (75 fps)
- Semantic + acoustic tokens trong 1 codec
- First codebook encode semantic information
- Remaining codebooks encode acoustic details

<a id="eq-mimi-bitrate"></a>

$$
\text{Bitrate}_{\text{Mimi}} = 12.5 \times Q \times \log_2(C) \text{ bps}
$$

với $Q = 8$ codebooks, $C = 2048$ entries → 1.1 kbps.

### Depth Transformer

Moshi sử dụng **Depth Transformer** để predict $Q$ codebook levels:

- **Temporal Transformer**: Process sequence over time (large, autoregressive)
- **Depth Transformer**: Process codebook levels at each timestep (small)

<a id="eq-moshi-depth"></a>

$$
\mathbf{h}_t = \text{TemporalTF}(\mathbf{h}_{<t}), \quad \mathbf{a}_t^{(q)} = \text{DepthTF}(\mathbf{h}_t, \mathbf{a}_t^{(<q)})
$$

```python
#| eval: false
#| code-fold: true
#| code-summary: "Moshi Depth Transformer (Simplified)"
import torch
import torch.nn as nn
from torch import Tensor

class DepthTransformer(nn.Module):
    """Predict Q codebook tokens at each time step."""

    def __init__(
        self,
        d_model: int = 4096,
        n_codebooks: int = 8,
        codebook_size: int = 2048,
        depth_layers: int = 6,
    ) -> None:
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model)
            for _ in range(n_codebooks)
        ])
        self.depth_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=16, dim_feedforward=d_model * 4,
                batch_first=True
            ),
            num_layers=depth_layers,
        )
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in range(n_codebooks)
        ])

    def forward(
        self,
        h_temporal: Tensor,  # [B, D] - from temporal transformer
    ) -> Tensor:
        """Autoregressively predict Q codebook tokens."""
        B = h_temporal.shape[0]
        tokens: list[Tensor] = []
        depth_input: Tensor = h_temporal.unsqueeze(1)  # [B, 1, D]

        for q in range(self.n_codebooks):
            # Run depth transformer
            out: Tensor = self.depth_tf(
                depth_input
            )  # [B, q+1, D] - float32

            # Predict token for codebook q
            logits: Tensor = self.heads[q](
                out[:, -1, :]
            )  # [B, codebook_size] - float32

            token: Tensor = logits.argmax(dim=-1)  # [B] - int64
            tokens.append(token)

            # Add embedding to depth input
            emb: Tensor = self.embeddings[q](
                token
            )  # [B, D] - float32
            depth_input = torch.cat(
                [depth_input, emb.unsqueeze(1)], dim=1
            )  # [B, q+2, D]

        return torch.stack(tokens, dim=1)  # [B, Q] - int64
```

### Training

- Pre-trained Helium 7B LLM trên text
- Fine-tuned trên paired (user audio, model audio) conversations
- **Streaming inference**: 12.5 fps = 80ms per step, với ~200ms tổng latency

## Turn-Taking Models

### Challenges

Turn-taking trong conversation tự nhiên cực kỳ phức tạp:

1. **Overlap**: ~40% turn transitions có overlap (người nói tiếp trước khi người kia ngừng)
2. **Backchannels**: "ừ", "vâng", "uh-huh" - không phải turn-taking mà là acknowledgment
3. **Interruptions**: Cắt ngang chủ đích - model phải ngừng generate
4. **Silence interpretation**: Im lặng 500ms có thể là "đang nghĩ" hoặc "đã nói xong"

### Voice Activity Projection (VAP)

VAP [^ekstedt2022voice] dự đoán ai sẽ nói tiếp:

<a id="eq-vap"></a>

$$
p(\text{speaker}_{\text{next}} | \text{audio}_{t-L:t}) = \sigma(\text{VAP}(\mathbf{x}_{t-L:t}))
$$

## So sánh Full-Duplex Models (mid-2026)

| Model | Latency first byte | Turn-taking | Open-source | Codec | Năm |
|---|---|---|---|---|---|
| Moshi | khoảng 200 ms | Neural dual-stream | Có | Mimi | Sep 2024 |
| GPT-4o Voice | khoảng 320 ms | Neural | Không | Proprietary | 2024 |
| Qwen2.5-Omni | khoảng 300 ms | Thinker-Talker | Có | CosyVoice codec | 2025 |
| Gemini Live | khoảng 250 ms | Neural | Không | Proprietary | 2025 |
| GPT-Realtime | sub-500 ms target | Neural | Không | Proprietary | Aug 2025 |
| Qwen3-Omni | sub-300 ms | MoE Thinker-Talker | Có | Multi-codebook audio | Sep 2025 |
| Moshi v2 + MoshiRAG | khoảng 200 ms | Neural + RAG | Có | Mimi | Apr 2026 |
| Gemini 3 Live | sub-250 ms target | Neural | Không | Proprietary | 2026 |

: So sánh Full-Duplex Dialogue Models <a id="tbl-full-duplex-comparison"></a>

## MoshiRAG (April 2026)

MoshiRAG là phát triển quan trọng của Kyutai vào tháng 4/2026, giải quyết một hạn chế cơ bản của Speech LLM: **kiến thức bị giới hạn bởi training data**.

### Vấn đề

Moshi v1 và các Speech LLM full-duplex đều có vốn kiến thức cố định tại thời điểm training. Khi user hỏi về sự kiện mới, dữ liệu chuyên ngành, hoặc thông tin cá nhân hoá, model không thể trả lời chính xác. Pipeline cascaded có thể bù bằng cách gọi text-LLM cộng RAG, nhưng phá vỡ ưu điểm low-latency của full-duplex.

### Cơ chế asynchronous RAG

MoshiRAG giữ nguyên kiến trúc dual-stream của Moshi nhưng bổ sung **asynchronous knowledge retrieval**: khi model phát hiện câu hỏi cần kiến thức chi tiết, nó gửi truy vấn đến text-LLM cộng RAG system (chạy song song trên server), và inject kết quả vào stream khi sẵn sàng. Cơ chế này giữ được latency thấp cho phần lớn câu trả lời, đồng thời cho phép trả lời các câu hỏi cần kiến thức ngoài.

### Pocket TTS và MoshiVis (cùng giai đoạn)

Kyutai cũng release:

- **Pocket TTS**: TTS khoảng 100M parameters, được báo cáo có chất lượng cạnh tranh với một số model lớn hơn trong thiết lập đánh giá công khai, hỗ trợ voice cloning và streaming realtime.
- **MoshiVis**: Moshi với image input support, giữ nguyên latency realtime.

## Đánh giá full-duplex dialogue

Đánh giá full-duplex khó hơn đánh giá ASR hay TTS đơn lẻ, vì cần đo cả nội dung lẫn động lực đối thoại. Các metric thực dụng:

- **First byte latency**: thời gian từ "user nói xong" đến "AI bắt đầu phát audio". Target: dưới 500 ms cho voice agent production, dưới 300 ms cho trải nghiệm natural.
- **Interruption handling latency**: thời gian AI ngừng phát audio khi user bắt đầu nói chen vào.
- **Backchannel naturalness**: tần suất và độ tự nhiên của backchannel ("vâng", "à", "ừm").
- **Turn-taking accuracy**: tỷ lệ AI nhường lượt đúng lúc, không cắt lời user, không chậm trễ.
- **Conversation MOS**: Mean Opinion Score do người đánh giá cho toàn cuộc đối thoại, không chỉ đoạn audio đơn lẻ.

Hiện chưa có benchmark public chuẩn cho full-duplex evaluation. Các sản phẩm thương mại đều dùng internal eval với human raters.

## Tóm tắt

1. **Full-duplex dialogue** yêu cầu nghe và nói đồng thời, phá vỡ rào cản turn-taking tuần tự của voice AI thế hệ cũ.
2. **Moshi** (Kyutai, tháng 9/2024) tiên phong với dual-stream cộng inner monologue cộng Mimi codec, đạt latency dưới 300 ms.
3. **Depth Transformer** giải quyết multi-codebook generation hiệu quả trong Moshi.
4. **Turn-taking** vẫn là bài toán mở, cần neural models thay vì rule-based đơn giản.
5. **MoshiRAG** (tháng 4/2026) mở rộng full-duplex sang retrieval-augmented, giải quyết hạn chế về kiến thức.
6. **Xu hướng**: GPT-Realtime, Gemini 3 Live, Qwen3-Omni đều hỗ trợ full-duplex. Mỗi nhà cung cấp có cách tiếp cận riêng nhưng tất cả đều nhắm tới sub-300 ms first-byte latency.



---

<!-- References (auto-generated from .bib) -->
[^defossez2024moshi]: D{\'e}fossez, Alexandre and Musicant, Laurent and others, "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue", arXiv preprint arXiv:2410.00037
[^ekstedt2022voice]: Ekstedt, Erik and Skantze, Gabriel, "Voice Activity Projection: Self-supervised Learning of Turn-taking Events", Interspeech
