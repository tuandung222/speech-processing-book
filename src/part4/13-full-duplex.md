# Full-Duplex Dialogue

## Tổng quan

Full-duplex dialogue là khả năng **nói và nghe đồng thời** - giống như cuộc trò chuyện tự nhiên giữa người với người. Đây là thách thức lớn nhất trong voice AI, đòi hỏi model phải xử lý **real-time turn-taking**, **interruptions**, và **backchannel signals** (ừ, vâng, à...).

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

## So sánh Full-Duplex Models

| Model | Latency | Turn-taking | Open-source | Codec |
|-------|---------|------------|-------------|-------|
| Moshi | ~200ms | Neural | Có | Mimi |
| GPT-4o | ~320ms | Neural | Không | Proprietary |
| Qwen2.5-Omni | ~300ms | Neural | Có | CosyVoice codec |
| Gemini Live | ~250ms | Neural | Không | Proprietary |

: So sánh Full-Duplex Dialogue Models <a id="tbl-full-duplex-comparison"></a>

## Tóm tắt

1. **Full-duplex dialogue** yêu cầu nghe và nói đồng thời
2. **Moshi** tiên phong với dual-stream + inner monologue + Mimi codec
3. **Depth Transformer** giải quyết multi-codebook generation hiệu quả
4. **Turn-taking** vẫn là bài toán mở - cần neural models thay vì rule-based
5. Xu hướng: Tất cả major players (OpenAI, Google, Alibaba) đều đang phát triển full-duplex



---

<!-- References (auto-generated from .bib) -->
[^defossez2024moshi]: D{\'e}fossez, Alexandre and Musicant, Laurent and others, "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue", arXiv preprint arXiv:2410.00037
[^ekstedt2022voice]: Ekstedt, Erik and Skantze, Gabriel, "Voice Activity Projection: Self-supervised Learning of Turn-taking Events", Interspeech
