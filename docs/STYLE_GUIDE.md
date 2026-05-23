# Style Guide — S-AI Speech Book

**Audience target**: Data scientist nền tảng **NLP/LLM** (BERT, GPT, Transformer) muốn chuyển sang **Speech AI**.

**Goal**: Viết với chất lượng giảng viên đại học tận tâm, có năng lực sư phạm. KHÔNG phải lecture notes vắn tắt. KHÔNG phải bullet-list dump.

---

## 1. Voice & Tone (Bắt buộc)

### 1.1 Build intuition trước formalism

❌ **BAD** (vắn tắt, formalism-first):
> CTC loss được định nghĩa:
> $$\mathcal{L}_{CTC} = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t | X)$$

✅ **GOOD** (intuition-first):
> Để hiểu CTC, đặt câu hỏi: làm sao một model có thể "khớp" output text với input audio mà KHÔNG có alignment label sẵn?
>
> Trong NLP, bạn quen với seq2seq: encoder đọc input → decoder sinh output token-by-token, mỗi step decoder chọn 1 token. Audio thì khác — input có thể có 1000 frames nhưng output text chỉ 10 ký tự. Không có "1-1 alignment" giữa frame và character.
>
> CTC giải bài toán này bằng cách nói: "OK, tôi sẽ cho model output 1 ký tự HOẶC blank tại mỗi frame, rồi collapse lại". Ví dụ frames = `[h, h, ε, ε, e, e, l, ε, l, o]` → collapse → `"hello"`.
>
> Loss function CTC marginalize trên TẤT CẢ các paths đều collapse về target Y:
> $$\mathcal{L}_{CTC} = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t | X)$$
>
> trong đó $\mathcal{B}^{-1}(Y)$ là tập tất cả paths collapse về Y, $T$ là số frames.

### 1.2 NLP→Speech bridge mỗi concept mới

Mọi concept lạ phải có 1 đoạn "**Đối chiếu với NLP**" hoặc "**Cho audience NLP**".

Ví dụ:
- **Mel spectrogram** → "tương tự BPE tokenization: cả 2 chuyển continuous signal/text thành discrete representation phù hợp neural net"
- **CTC blank token** → "tương tự `<MASK>` của BERT, nhưng cho phép skip thay vì predict"
- **Codec tokens (EnCodec, Mimi)** → "tương tự BPE tokens nhưng cho audio: vocabulary ~1024-4096, dùng VQ-VAE"
- **Wav2Vec self-supervised** → "tương tự BERT MLM nhưng masked frame thay vì masked token"

### 1.3 Anticipate confusion — dừng lại giải thích

Khi gặp concept khó/phản trực giác, ĐỪNG bỏ qua. Pause và viết:

> **🤔 Tại sao điều này lại đúng?**
> Bạn có thể nghĩ X, nhưng thực ra Y vì Z. Lý do là...

Hoặc:

> **⚠️ Lưu ý dễ nhầm**
> Đừng nhầm A với B. A là [định nghĩa A], còn B là [định nghĩa B]. Khác nhau ở chỗ...

### 1.4 Concrete analogies

Mỗi concept abstract nên có **ít nhất 1 analogy** từ NLP, daily life, hoặc CS.

| Analogy class | Ví dụ |
|---|---|
| **NLP analogy** | "STFT giống như sliding window n-gram" |
| **Daily life** | "Sample rate giống frame rate khi quay video" |
| **CS** | "Beam search giống DFS với pruning" |
| **Linear algebra** | "Mel filterbank là ma trận triangular weights" |

### 1.5 GIỌNG VĂN NATURAL — Giảng viên Việt nói chuyện, không robot

**ĐÂY LÀ NGUYÊN TẮC QUAN TRỌNG NHẤT.** Văn phong phải nghe như một giảng viên Việt đang đứng lớp, nhiệt huyết, đôi khi vui đùa, đôi khi nghiêm túc, KHÔNG phải textbook đều đều khô khan.

#### Patterns natural lecturer Việt

✅ **Bắt đầu paragraph với rhetorical question**:

> "Bạn có để ý điều gì lạ ở đây không?"
> "Đến đây bạn chắc đang nghĩ: vậy tại sao không dùng X?"
> "Hãy thử nghĩ thử một chút — nếu bỏ bước này thì sẽ ra sao?"

✅ **Surprise/excitement ở các điểm thú vị**:

> "Và đây mới là chỗ hay nhất:"
> "Bùm! Đó chính là lý do."
> "Cực kỳ elegant phải không?"

✅ **Acknowledge difficulty / pause**:

> "Phần này hơi khó nhằn một chút, mình đi từ từ nhé."
> "Đoạn formula này khó đọc, nhưng đừng lo — mình sẽ dịch ra plain English ngay."
> "Hồi mình mới đọc paper này, cũng phải đọc lại 3 lần mới hiểu."

✅ **Personal touches, dùng "mình" hoặc "tôi"**:

> "Mình thường khuyên junior nên đọc paper này trước."
> "Tôi từng deploy thử và thấy..."

✅ **Conversational connectors**:

> "OK giờ qua phần tiếp theo nhé."
> "Anyway, quay lại câu hỏi cũ..."
> "Alright, đến đây chắc bạn đã nắm cơ bản rồi."
> "Nói thật là..."

✅ **Mix Vietnamese-English natural như dev VN nói chuyện**:

> "Pipeline này có 3 stages: encoder, decoder, vocoder. Mỗi stage có thể train độc lập, swap independently."
> "Cái này gọi là 'in-context learning' — bạn show vài examples, model tự generalize."
> "OK so giờ ta cần handle case khi user nói code-switched."

❌ **TUYỆT ĐỐI TRÁNH**:

> ❌ "Phần này thảo luận về..." (đều đều, textbook)
> ❌ "Chương này tổng hợp..." (impersonal)
> ❌ "Ta sẽ xem xét..." (anonymous voice)
> ❌ "Trong phần này, ta sẽ nghiên cứu..." (cold academic)
> ❌ "Cần lưu ý rằng..." (textbook-y)

#### Compare bad vs good

❌ **BAD** (robotic, even-tone, textbook):
> Phần này thảo luận về CTC loss. CTC loss được định nghĩa là âm log của tổng xác suất trên tất cả các paths có thể collapse về target. Cần lưu ý rằng CTC giả định các output tại các time steps là độc lập có điều kiện.

✅ **GOOD** (natural Vietnamese lecturer):
> OK giờ ta vào phần thú vị nhất của CTC: cái loss function. Nhìn vào công thức bên dưới đừng hoảng nhé:
>
> $$\mathcal{L}_{CTC} = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t | X)$$
>
> Dịch ra plain English: "tổng xác suất trên TẤT CẢ các paths khả dĩ mà collapse về đúng target Y, lấy log âm". Nghe phức tạp, nhưng intuition rất đẹp.
>
> Bạn đã thấy seq2seq dùng cross-entropy đúng không? Mỗi step decoder có 1 target token cụ thể, model học predict đúng cái đó. CTC khác: model không biết "frame nào tương ứng với character nào", nên thay vì pick 1 alignment, nó **marginalize** trên TẤT CẢ alignment khả dĩ. Bùm, đó là toàn bộ idea.
>
> Có một subtle assumption ở đây: CTC giả định các output tại mỗi time step độc lập có điều kiện cho input. Đây là điểm yếu mà RNN-T sau này khắc phục. Mình sẽ quay lại lúc nói về RNN-T.

Thấy sự khác biệt chứ? Bản GOOD có:
- Câu mở rhetorical/casual ("OK giờ ta vào phần thú vị nhất...")
- Acknowledge difficulty ("đừng hoảng nhé")
- Dịch math ra plain English ngay sau công thức
- Bridge sang concept đã quen (seq2seq cross-entropy)
- Excitement đánh dấu insight ("Bùm, đó là toàn bộ idea")
- Personal voice ("Mình sẽ quay lại lúc nói về...")
- Tự nhiên mix Vi-En ("marginalize", "alignment", "subtle assumption", "code-switched", "swap independently")

#### Quy tắc về mix Vi-En

- ✅ Dùng English cho **technical terms ngắn và rõ**: encoder, decoder, pipeline, latency, throughput, fine-tune, deploy, baseline, mainstream, etc.
- ✅ Dùng English cho **acronyms**: ASR, TTS, NLP, LLM, MoE, KV cache, etc.
- ✅ Dùng English cho **modern jargon**: code-switching, hallucination, prompt engineering, function calling, in-context learning, etc.
- ✅ Dùng Vietnamese cho **concepts có translation rõ và phổ biến**: âm thanh, tần số, giọng nói, ngữ nghĩa, tín hiệu.
- ❌ KHÔNG cố ép translate mọi thuật ngữ ("siêu tham số" thay "hyperparameter" → tệ).
- ❌ KHÔNG dùng English chỗ không cần (đừng nói "I think rằng..." khi "Mình nghĩ rằng..." đẹp hơn).

#### Pacing — đừng đều đều

Pacing biến đổi giữa các đoạn:

- **Đoạn nặng kỹ thuật** (formula, math): chậm, careful, giải thích từng dòng.
- **Đoạn intuition**: nhanh, animated, dùng analogy ngay.
- **Đoạn transition**: ngắn gọn, conversational ("OK giờ ta thử thực tế hơn...").
- **Đoạn summary**: structured nhưng vẫn có personality.

Tránh:
- Mọi paragraph cùng độ dài.
- Mọi câu cùng cấu trúc.
- Không bao giờ có short punchy sentence ("Đúng vậy." "Chính xác." "Đó là điểm cốt lõi.").

### 1.6 Flowing prose, không bullet-dump

❌ **BAD** (bullet dump):
> CTC có 3 đặc điểm:
> - Thêm blank token
> - Output mỗi frame
> - Marginalize alignments

✅ **GOOD** (narrative):
> CTC khác với seq2seq ở 3 thiết kế cốt lõi. Đầu tiên, vocabulary được mở rộng thêm một **blank token** (ký hiệu $\langle b \rangle$ hoặc $\epsilon$) — đây không phải padding mà mang ý nghĩa "không phát ra ký tự nào tại frame này". Thứ hai, model output **một xác suất cho mỗi frame** — không có khái niệm "decoder step", mỗi input frame trực tiếp dự đoán một character (có thể là blank). Thứ ba, CTC **marginalize** trên tất cả paths khả dĩ, không chỉ một alignment cố định — đây là điểm tinh tế nhất.

### 1.6 Tables for reference, prose for explanation

Tables dùng tốt cho:
- So sánh tham số (params, perf metrics, vocab sizes)
- Reference data (sample rate standards, model checkpoints)
- API/signature catalog

Tables KHÔNG dùng cho:
- Explanation chính (dùng paragraph)
- Reasoning/derivation steps (dùng paragraph có thứ tự)

---

## 2. Glossary — NLP↔Speech Mapping (Mandatory references)

Khi gặp 1 trong các terms này, LUÔN bridge sang NLP equivalent:

| Speech AI Term | NLP Analogue | Bridge Explanation |
|---|---|---|
| Mel spectrogram | Static word embeddings | Hand-crafted feature representation, deterministic |
| Wav2Vec, HuBERT | BERT MLM | Self-supervised pretrain on raw input, masked prediction |
| EnCodec, Mimi codec tokens | BPE tokens | Discrete vocabulary from continuous input, ~1024-8192 codes |
| CTC loss | seq2seq cross-entropy | Loss for misaligned input/output sequences |
| CTC blank | `<MASK>`/`<PAD>` | Special token for "no emission at this step" |
| RNN-T | Transformer decoder | Stream-friendly seq2seq with joint network |
| Conformer | Transformer encoder | Hybrid CNN + self-attention encoder |
| Whisper | T5/BART | Encoder-decoder, multitask, large pretrained |
| VITS, F5-TTS | GPT/Stable Diffusion | End-to-end generative for audio |
| Phoneme | Sub-word (BPE chunk) | Linguistic unit smaller than word |
| Forced alignment | Token-to-char alignment in seq2seq | Map audio frames to text units |
| Voice Activity Detection (VAD) | Sentence boundary detection | Segment continuous stream |
| Speaker embedding | User/persona embedding | Identity vector for conditional generation |
| Streaming ASR | Causal LM (vs bidirectional) | Online prediction without future context |
| Full-duplex dialogue | Multi-turn chat with overlapping turns | Real-time bidirectional speech |
| Audio LLM (Moshi, Qwen2-Audio) | Multimodal LLM | LLM with audio I/O |

---

## 3. Mathematical Notation Conventions

### 3.1 Variables

| Type | Convention | Example |
|---|---|---|
| Scalars (time, frequency, dimension) | lowercase italic | $t$, $f$, $d$ |
| Vectors | lowercase bold | $\mathbf{x}$, $\mathbf{h}$, $\mathbf{y}$ |
| Matrices | uppercase bold | $\mathbf{W}$, $\mathbf{H}$ |
| Sets | mathcal | $\mathcal{V}$, $\mathcal{L}$ |
| Random variables | uppercase italic | $X$, $Y$ |
| Probability | $P(\cdot)$ or $p(\cdot)$ | $P(y|x)$ |
| Expectation | $\mathbb{E}$ | $\mathbb{E}_{x \sim p}[f(x)]$ |
| Loss | $\mathcal{L}$ | $\mathcal{L}_{\text{CTC}}$ |

### 3.2 Indices

- Time index for audio: $t$ (samples) or $m$ (frames)
- Token/character index: $u$
- Frequency bin: $k$
- Layer index: $\ell$
- Batch index: $b$ or $i$

### 3.3 Functions

- Use `\operatorname{}` for multi-letter functions: $\operatorname{softmax}$, $\operatorname{ReLU}$
- Use `\text{}` for non-italic labels inside math: $\text{frame\_length}$ (NOTE: KaTeX yêu cầu `\_` escape inside `\text{}`)
- Pre-defined macros (xem `theme/katex-macros.txt`): `\R`, `\E`, `\softmax`, `\argmax`, `\argmin`

### 3.4 Display vs inline math

- Inline: `$...$` for short expressions trong câu văn
- Display: `$$...$$` cho công thức quan trọng, definition, theorem
- ALWAYS đặt `$$` trên dòng riêng (không có content khác cùng dòng)

---

## 4. Document Structure (Chapter Template)

Mỗi chapter nên follow structure sau:

```
# Chương N: [Tên]

[Opening paragraph: WHY chapter này quan trọng. Reference NLP background.]

## Bài toán

[Define problem rigorously. Use motivating example.]

## Intuition

[Build intuition trước khi formalism. Analogy, picture, mental model.]

## Formalization

[Mathematical/algorithmic definition. Step-by-step derivation if non-trivial.]

## Code Walkthrough

[PyTorch reference implementation, well-commented. Audience-friendly.]

## So sánh & Đối chiếu

[Compare with alternative approaches. Trade-offs table.]

## Limitations & Open Problems

[Honest discussion of what doesn't work. Modern challenges.]

## Tóm tắt

[5-7 bullet recap of key takeaways. NOT a TL;DR.]

## Tài liệu tham khảo

[Bibliography for further reading.]
```

---

## 5. Length Expectations

Mỗi chapter:
- **Tối thiểu**: 1500 dòng markdown (~8000 từ Vietnamese)
- **Mong đợi**: 2000-3000 dòng (~12000-15000 từ)
- **Tối đa**: 4000 dòng (chia chapter nếu quá dài)

So với hiện tại (~300-500 dòng/chapter), điều này có nghĩa **expand 4-8x**. Đây là expected — chất lượng > số lượng, nhưng chất lượng yêu cầu đủ chỗ để giải thích.

---

## 6. Code Style

### 6.1 PyTorch reference implementations

- Use type hints aggressively (`Tensor`, `Optional[Tensor]`, etc.)
- Annotate tensor shapes inline as comments: `# [B, T, D] - float32`
- Use descriptive variable names (NOT `x`, `h`, but `mel_spec`, `encoder_hidden`)
- Provide minimal runnable examples after the class definition

```python
class Encoder(nn.Module):
    """Conformer encoder for ASR.
    
    Pedagogical note: Conformer = Transformer + ConvNet, combining
    long-range attention with local convolution. Similar to how
    Swin Transformer combines local windows with hierarchical structure.
    """
    def __init__(self, n_mels: int = 80, hidden_dim: int = 512):
        super().__init__()
        # Input projection: mel features -> hidden dim
        self.input_proj = nn.Linear(n_mels, hidden_dim)
        # ... rest

    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: [B, T, n_mels] - float32, mel spectrogram
        Returns:
            hidden: [B, T, hidden_dim] - float32, encoded features
        """
        x = self.input_proj(mel)  # [B, T, hidden_dim]
        # ...
        return x


# Usage example
encoder = Encoder(n_mels=80, hidden_dim=512)
mel = torch.randn(2, 100, 80)  # 2 batch, 100 frames, 80 mel bins
hidden = encoder(mel)
print(hidden.shape)  # torch.Size([2, 100, 512])
```

---

## 7. Forbidden Patterns

### 7.1 Em dashes (—)
NEVER use em dash (`—`) hoặc double-dash (`--`). Dùng colon, parenthesis, hoặc period.

### 7.2 Anonymous voice
KHÔNG dùng "chúng ta sẽ thấy...", "you will see..." chung chung. Cụ thể hơn:
- "Hãy quan sát hiện tượng này:" 
- "Câu hỏi tự nhiên đặt ra là:"

### 7.3 Hand-waving phrases
KHÔNG: "rõ ràng là", "dễ thấy", "obviously", "trivially"
DÙNG: dẫn dắt rõ ràng, show the working.

### 7.4 Bullet-list explanations
Như đã nói ở 1.5: bullet cho catalog, prose cho explanation.

### 7.5 Tautology
KHÔNG: "Mel spectrogram là spectrogram trên thang mel"
DÙNG: "Mel spectrogram chuyển power spectrogram từ thang linear frequency sang thang **mel** — thang phi tuyến mô phỏng cách tai người cảm nhận tần số (logarithmic ở high freq, linear ở low freq)."

---

## 8. Quality Checklist (per chapter)

Trước khi merge 1 chapter, check:

- [ ] Opening paragraph có "tại sao cần đọc chapter này?" với hook cho audience NLP
- [ ] Mỗi concept lạ có ít nhất 1 NLP analogy hoặc daily-life analogy
- [ ] Mỗi công thức quan trọng có **trước** intuition paragraph
- [ ] Có ít nhất 1 "🤔" hoặc "⚠️" callout cho điểm dễ nhầm
- [ ] Code có type hints và shape annotations
- [ ] Code có 1 runnable example phía dưới class definition
- [ ] Có "So sánh & Đối chiếu" với alternative approaches
- [ ] Có "Limitations" — honest về hạn chế
- [ ] Có "Tóm tắt" 5-7 bullets nhưng có context, không phải TL;DR
- [ ] Đã đọc lại 1 lần sau khi viết xong, không có em-dash, không có hand-waving
- [ ] mdBook build local pass (no KaTeX errors)
- [ ] Vietnamese diacritics đúng chính tả (hoặc giảm thiểu)
