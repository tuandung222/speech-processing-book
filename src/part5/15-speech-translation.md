# Chương 15: Speech Translation

## Mở đầu: Tại sao Speech Translation là frontier hot nhất 2024-2026

Năm 2024, Meta công bố SeamlessM4T v2 trên tạp chí **Nature**, một sự công nhận khoa học hiếm hoi cho công trình AI. Cùng năm, Google demo Gemini 2.0 dịch live qua video call. Kyutai phát hành Hibiki có thể dịch song song trong khi người nói còn đang nói. OpenAI bổ sung "realtime translation" vào API tháng 8/2025. Năm 2026, Qwen3.5-Omni vượt Gemini 3.1 Pro trên benchmark dịch tổng quát.

Tại sao Speech Translation đột nhiên trở thành research frontier? Ba lý do hội tụ:

1. **Speech LLMs trưởng thành**: codec tokens (Mimi, EnCodec) + multilingual self-supervised encoder (w2v-BERT 2.0) + LLM scaling cho phép end-to-end speech translation với chất lượng vượt cascaded.
2. **Real-time deployment requirements**: video calling, hội nghị quốc tế, giáo dục online cần dịch <500ms latency, vượt khả năng của cascaded pipeline truyền thống.
3. **Low-resource language coverage**: 7,000+ ngôn ngữ trên thế giới, hơn 3,000 không có chữ viết. Speech-to-speech translation textless là cách duy nhất phục vụ những cộng đồng này.

Cuốn sách trước đây có chương về Speech Translation rất sơ sài. Chương này được rewrite hoàn toàn theo chuẩn mục đích: cho một data scientist NLP/LLM hiểu sâu, từ bài toán cơ bản đến SOTA năm 2026, đến deployment thực tế.

> **📝 Cấu trúc chương**
>
> 12 phần, từ taxonomy bài toán → metrics → datasets → cascaded → end-to-end → simultaneous → SOTA 2024-2026 → production → Vietnamese-specific → limitations → roadmap.
>
> Nếu bạn vội, đọc Phần 1 + Phần 8 (SOTA models) + Phần 9 (production). Nếu bạn nghiên cứu academic, đọc Phần 2 (metrics), Phần 4-6 (architectures), Phần 11 (open problems).

---

## Phần 1 — Phân loại bài toán (Taxonomy)

Speech Translation không phải một bài toán đơn lẻ. Nó là một họ bài toán với input/output khác nhau, latency requirements khác nhau, và data requirements khác nhau. Hiểu rõ taxonomy giúp bạn chọn đúng approach cho usecase cụ thể.

### 1.1 Bốn task chính theo input/output

| Ký hiệu | Input | Output | Use case điển hình |
|---|---|---|---|
| **S2TT** | Speech (ngôn ngữ A) | Text (ngôn ngữ B) | Phụ đề video, transcription đa ngôn ngữ |
| **S2ST** | Speech (ngôn ngữ A) | Speech (ngôn ngữ B) | Hội thoại 1-1 đa ngôn ngữ, dubbing |
| **T2ST** | Text (ngôn ngữ A) | Speech (ngôn ngữ B) | Chatbot đa ngôn ngữ với voice output |
| **T2TT** | Text (ngôn ngữ A) | Text (ngôn ngữ B) | Translation truyền thống (NMT/MT) |

Trong cuốn sách này, ta tập trung vào **S2TT** và **S2ST** vì chúng là speech-centric. T2TT là Machine Translation truyền thống (Google Translate, NLLB), không phải Speech AI. T2ST chính là TTS cho ngôn ngữ khác (đã được cover ở Phần III, Chương 8-10).

### 1.2 Bốn task theo latency mode

Một chiều phân loại trực giao quan trọng: **offline** (chờ user nói xong rồi dịch) vs **streaming/simultaneous** (dịch song song khi user nói).

| Ký hiệu | Latency mode | Mô tả | Use case |
|---|---|---|---|
| **Offline ST** | Sau khi user kết thúc | Đợi hết câu/hết clip, dịch toàn bộ | Subtitle, batch transcription |
| **Streaming ST** | Bắt đầu dịch khi user còn nói | Partial output, update khi có thêm context | Caption live, voice assistant |
| **Simultaneous ST** | Dịch token-by-token, không chờ end-of-sentence | Sub-sentence latency, monotonic decisions | Conference interpretation, real-time meeting |
| **Cascaded streaming** | ASR streaming → MT incremental → TTS streaming | Pipeline streaming end-to-end | Voice agent đa ngôn ngữ |

Phần 7 sẽ đào sâu các kỹ thuật cho simultaneous translation (Wait-k, Adaptive policies, EMMA).

### 1.3 Một chiều thứ ba: text-required vs textless

Đây là chiều phân loại mới (post-2022) và rất quan trọng cho ứng dụng low-resource.

- **Text-required**: pipeline cần transcription ở source và/hoặc target. Áp dụng được cho ~5% ngôn ngữ trên thế giới có hệ thống viết tốt.
- **Textless (unit-based)**: pipeline dùng discrete units (HuBERT k-means clusters, hoặc codec tokens) thay thế text intermediate. Áp dụng cho mọi cặp ngôn ngữ có speech parallel data, kể cả ngôn ngữ không có chữ viết.

Textless S2ST đã được Translatotron 2, Textless S2ST (Lee 2022), và AudioPaLM 2 chứng minh khả thi. Phần 6 sẽ deep dive.

### 1.4 Tóm tắt taxonomy

Ba chiều phân loại trên cho ta một không gian 2D × 3D × 2 = 16 combinations. Trong thực tế, chỉ một số combinations là phổ biến:

> **🎯 Combinations phổ biến**
>
> - **Offline S2TT text-required**: Whisper + NLLB. Production-ready, accuracy cao.
> - **Streaming S2TT text-required**: SeamlessStreaming, Hibiki (Kyutai). Cutting edge 2024-2025.
> - **Offline S2ST text-required**: SeamlessM4T v2. SOTA cho high-resource pairs.
> - **Simultaneous S2ST textless**: AudioPaLM 2, Translatotron 3. Frontier research.
> - **Streaming T2ST text-required**: ElevenLabs Multilingual, voice agent đa ngôn ngữ.

---

## Phần 2 — Metrics: đo lường chất lượng và độ trễ

Speech Translation evaluation là một trong những lĩnh vực nhiều metric nhất trong toàn bộ NLP/Speech. Lý do: bạn vừa đo chất lượng (đo gì?) vừa đo latency (đo thế nào?). Phần này tổng hợp các metric quan trọng nhất, với chú trọng các metric mới (BLASER 2.0, COMET) thay thế BLEU truyền thống.

### 2.1 Quality metrics cho S2TT (text output)

#### 2.1.1 BLEU — gold standard truyền thống

BLEU (Papineni et al., 2002) là n-gram precision với brevity penalty:

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
$$

trong đó $p_n$ là precision của n-gram, $w_n = 1/4$ (uniform weight), BP = brevity penalty để phạt output ngắn quá.

**Ưu điểm**: nhanh tính, no model needed, well-established.

**Nhược điểm nghiêm trọng cho Speech Translation**:

1. **Sensitive to wording**: "I love speech" vs "Speech is loved by me" có cùng meaning nhưng BLEU thấp.
2. **No semantic understanding**: BLEU chỉ đếm overlap, không hiểu paraphrase.
3. **Punctuation và casing artifacts**: ASR output thường không có punctuation, làm BLEU drop giả tạo.
4. **Brittle to ASR errors**: 1 từ sai có thể làm BLEU drop 20-30% vì phá n-gram alignment.

Vì các lý do này, BLEU đã bị thay thế bởi neural metrics ở các paper post-2023.

#### 2.1.2 COMET — neural metric chuẩn mới

COMET (Rei et al., 2020, updated 2024) là một regression model fine-tuned trên human ratings:

$$
\text{COMET}(\hat{y}, y, x) = f_\theta(\text{Encoder}(\hat{y}), \text{Encoder}(y), \text{Encoder}(x))
$$

trong đó $\hat{y}$ là hypothesis, $y$ là reference, $x$ là source.

**Ưu điểm so với BLEU**:

- Hiểu semantic similarity (sentence embeddings).
- Correlates rất tốt với human ratings (Pearson > 0.7 trên WMT direct assessment).
- Calibrated score 0-1 (BLEU score scale gây confusion).

**Phiên bản 2024-2025**: COMET-22 (XLM-R Large backbone), COMET-Kiwi (reference-free), XCOMET (extended quality estimation).

#### 2.1.3 BLASER 2.0 — neural metric cho cả speech và text

BLASER 2.0 (Meta, 2023) là neural metric đặc thù cho Speech Translation. Nó nhận trực tiếp speech embedding (SONAR) thay vì text, giảm dependency vào ASR:

$$
\text{BLASER}(\hat{a}, a, x) = g_\phi(\text{SONAR}(\hat{a}), \text{SONAR}(a), \text{SONAR}(x))
$$

trong đó $\hat{a}, a$ là speech hypothesis và reference (audio), $x$ là source (audio hoặc text).

**Ưu điểm**:

- Đánh giá được S2ST trực tiếp mà không cần ASR transcript của output (loại bỏ một nguồn nhiễu).
- Cross-modal: source có thể là text hoặc speech.
- Strong correlation với human ratings cho speech outputs.

**Nhược điểm**: cần SONAR model (~400M params), không lightweight như BLEU.

#### 2.1.4 chrF, METEOR, BERTScore (legacy)

- **chrF**: character n-gram F-score, robust hơn BLEU với morphologically rich languages.
- **METEOR**: synonym-aware metric, ít dùng cho ST gần đây.
- **BERTScore**: embedding similarity, đã được COMET thay thế.

### 2.2 Quality metrics cho S2ST (speech output)

Đánh giá speech output khó hơn text output vì có thêm chiều **acoustic quality** và **prosody**.

#### 2.2.1 ASR-BLEU (Cascaded eval)

Approach phổ biến nhất: chạy ASR trên speech output, tính BLEU trên transcription:

$$
\text{ASR-BLEU} = \text{BLEU}(\text{ASR}(\hat{a}), y_{\text{text}})
$$

**Nhược điểm**: phụ thuộc vào ASR quality. ASR error rate trên speech output thường > 5%, gây noise đáng kể.

#### 2.2.2 MOS (Mean Opinion Score)

Human-rated, scale 1-5 cho naturalness, similarity với reference voice.

- **MOS-N**: Naturalness (giọng tự nhiên không?).
- **MOS-S**: Similarity (giống voice reference không, cho voice cloning).
- **CMOS**: Comparative MOS, so sánh hai outputs theo cặp.

**Nhược điểm**: tốn kém ($1-5/sample human rating), không scalable.

#### 2.2.3 UTMOS, NISQA, DNSMOS (auto MOS approximations)

Neural model predicts MOS rating từ audio.

- **UTMOS** (Saeki et al., 2022): MOS prediction trained on Voice Conversion Challenge data, correlates ~0.85 với human MOS.
- **NISQA** (TU Berlin): naturalness, noisiness, coloration, discontinuity, loudness sub-scores.
- **DNSMOS** (Microsoft): designed for noise suppression eval, also used for ST.

### 2.3 Latency metrics cho Simultaneous Translation

Đây là lĩnh vực có nhiều metric đặc thù nhất.

#### 2.3.1 Average Lagging (AL)

AL đo trung bình token source đã consumed trước mỗi token target được produced:

$$
\text{AL}(y) = \frac{1}{|y|} \sum_{i=1}^{|y|} d(i) - \tau
$$

trong đó $d(i)$ là số source tokens consumed trước khi emit target token $y_i$, $\tau$ là baseline (thường là source/target length ratio).

**Ý nghĩa**: AL = 0 nghĩa là "translation hoàn hảo simultaneous" (không chờ). AL > 0 nghĩa là phải chờ trung bình $\text{AL}$ tokens trước khi dịch. AL < 0 không khả thi (translation chạy trước nói).

#### 2.3.2 Average Proportion (AP)

$$
\text{AP}(y) = \frac{1}{|x| \cdot |y|} \sum_{i=1}^{|y|} d(i)
$$

Range 0-1. AP = 0.5 nghĩa trung bình consume 50% source trước mỗi target token.

#### 2.3.3 Differentiable Average Lagging (DAL)

Cải tiến của AL cho phép training với policy gradient hoặc soft attention. Cho phép trade-off quality vs latency trong optimization.

#### 2.3.4 Latency-Quality curves (LAAL, ATD)

- **LAAL** (Length-Adaptive AL): normalize AL theo length, dùng cho long-form.
- **ATD** (Average Time Delay): latency tính theo giây thay vì tokens, sát với production hơn.

### 2.4 Tabulation: metric nào dùng cho ai

| Use case | Recommended metrics |
|---|---|
| Academic paper trên S2TT | COMET + BLEU + chrF |
| Academic paper trên S2ST | BLASER 2.0 + ASR-BLEU + UTMOS |
| Academic paper trên SimulST | LAAL + COMET + ATD |
| Production system A/B test | User-side: CSAT, completion rate. Server-side: ASR-BLEU samples + latency p50/p95/p99 |
| Voice cloning eval | MOS-S (similarity) + UTMOS (quality) |

---

## Phần 3 — Training Datasets

Quality của ST model phụ thuộc trực tiếp vào quality và scale của parallel speech-translation data. Phần này tổng hợp các dataset chính, kèm phân tích về domain bias và size limitations.

### 3.1 General-purpose ST datasets

| Dataset | Năm | Task | Hours | Languages | Domain | Source |
|---|---|---|---|---|---|---|
| **MuST-C** | 2019 | S2TT (En→8 langs) | ~400h | En → De, Es, Fr, It, Nl, Pt, Ro, Ru | TED talks | EU-funded |
| **CoVoST 2** | 2020 | S2TT (21→1, 1→15) | 2,880h | 21 source → En, En → 15 | CommonVoice (crowdsourced) | Meta |
| **VoxPopuli** | 2021 | S2TT + ASR | 400,000h (unlabeled), 1,800h (labeled) | 23 EU langs | EU Parliament | Meta |
| **FLEURS** | 2022 | S2TT + LID + ASR | ~12h/lang × 102 | 102 langs | Read speech (devset of FLORES-200) | Google |
| **CVSS** | 2022 | S2ST (21→1) | 1,153h | 21 source → En | Built on CoVoST + TTS-generated target | Google |
| **mTEDx** | 2021 | S2TT (multilingual TED) | 765h | 8 source → 11 target | TED talks | Meta |

### 3.2 Speech-to-Speech datasets

S2ST đặc biệt khó về data vì cần parallel SPEECH ở cả source và target. Hai approach để xây dataset:

1. **Human dubbing**: rất hiếm, expensive. Ví dụ: SE3 Italian-English (~80h).
2. **Synthetic target via TTS**: hầu hết S2ST datasets. CVSS-T dùng VITS-style TTS để generate target speech từ target text của CoVoST.

| Dataset | Pairs | Hours | Method |
|---|---|---|---|
| **CVSS-C** | 21 → En | 1,153h | Single-speaker TTS target |
| **CVSS-T** | 21 → En | 1,153h | Multi-speaker TTS target (xấp xỉ voice của source) |
| **MultilingualLibriSpeech (MLS)** | 8 langs | 50,500h (mostly mono, dùng cho ST hậu xử lý) | LibriVox audiobooks |
| **MUStCv2** | Multi → Multi | ~500h | TED talks v2 |

### 3.3 Streaming/Simultaneous-specific datasets

- **WMT IWSLT Simultaneous Translation track**: annual benchmark từ 2018, có train/dev/test splits với streaming-friendly format.
- **ESPnet-ST**: collection của ST datasets reformatted cho streaming eval.

### 3.4 Vietnamese-specific resources

Phần này quan trọng vì user explicit yêu cầu, sẽ deep dive ở Phần 10.

| Dataset | Hours | Direction | Source |
|---|---|---|---|
| **VLSP 2020 ASR Task 2** | 6h test | Vi audio → Vi text | VLSP Vietnamese Speech Lab |
| **PhoMT** | ~3M sentence pairs | En ↔ Vi (text only, dùng cho cascaded) | VinAI |
| **CoVoST Vi** | ~12h | Vi → En | Subset of CoVoST 2 |
| **FLEURS Vi** | ~12h | Vi → 101 langs, 101 → Vi | Google FLEURS |

> **⚠️ Data scarcity warning**
>
> Vietnamese ST có **rất ít** parallel speech data so với high-resource pairs (En-De có > 10,000h). Hệ luỵ: train end-to-end ST cho Vi rất khó, hầu hết production system Vi dùng cascaded pipeline (PhoWhisper → NLLB → vinai/vibert-tts).

### 3.5 Synthetic data approaches (2023+)

Khi không đủ parallel speech data, các approach modern dùng synthetic data:

1. **Pseudo-labeling**: chạy strong ASR + MT trên unlabeled speech, dùng output làm pseudo-labels.
2. **Back-translation**: train Tgt→Src model trước, dùng nó để generate synthetic Src từ Tgt monolingual data.
3. **TTS-based augmentation**: dùng TTS để generate target speech từ text translations.
4. **Self-training**: iterative pseudo-labeling với confidence filtering.

SeamlessM4T v2 dùng kết hợp tất cả 4 approaches để scale đến 100+ languages.

---

## Phần 4 — Cascaded Pipeline (ASR → MT → TTS): chi tiết kỹ thuật

Cascaded pipeline là approach truyền thống và vẫn dominant ở production năm 2026. Phần này phân tích cụ thể từng bước, error propagation, và optimization strategies.

### 4.1 Pipeline tổng quan

$$
\text{Speech}_{\text{src}} \xrightarrow{\text{ASR}} \text{Text}_{\text{src}} \xrightarrow{\text{MT}} \text{Text}_{\text{tgt}} \xrightarrow{\text{TTS}} \text{Speech}_{\text{tgt}}
$$

Mỗi bước là một mô hình riêng, có thể train độc lập, có thể swap independently.

### 4.2 Phân tích từng stage

#### 4.2.1 ASR stage

Lựa chọn typical:

- **Whisper Large-v3** (OpenAI, 2023): 99 languages, robust to noise, batch inference friendly.
- **Canary** (NVIDIA, 2024): 4 languages (En, De, Es, Fr), better WER hơn Whisper trong domain.
- **AssemblyAI Universal-2** (2024): commercial, low-latency streaming.
- **PhoWhisper** (VinAI, 2024): cho Vietnamese, vượt Whisper Large-v3 trên VLSP.

#### 4.2.2 MT stage

- **NLLB-200** (Meta, 2022): 200 languages, dense + MoE variants. Open source.
- **GPT-4 / GPT-5** (OpenAI): general-purpose, high quality, expensive.
- **NLLB-200 Distilled-600M**: smaller variant cho real-time.
- **Google Translate API**: production-tested cho 100+ languages.

#### 4.2.3 TTS stage

- **VITS** open source, multi-speaker.
- **XTTS v2** (Coqui): zero-shot voice cloning với 3-sec sample.
- **ElevenLabs Multilingual v2**: commercial SOTA cho voice quality.
- **F5-TTS** (2024): flow matching, zero-shot.
- **Cartesia Sonic 2**: streaming TTS với 75ms first byte latency.

### 4.3 Error propagation: tại sao cascaded không phải free lunch

Đây là vấn đề kinh điển của cascaded:

$$
P(\text{error}_{\text{final}}) \approx 1 - (1 - p_{\text{ASR}})(1 - p_{\text{MT}})(1 - p_{\text{TTS}})
$$

Với mỗi stage có 5% error rate:

$$
P(\text{error}_{\text{final}}) \approx 1 - 0.95^3 \approx 14.3\%
$$

Một sai sót ở ASR (mispelled name, missing punctuation) có thể bị MT khuếch đại (translating wrong word context), sau đó TTS đọc sai nghĩa.

### 4.4 Latency budget của cascaded

Ví dụ realistic cho voice agent đa ngôn ngữ:

| Stage | Latency | Note |
|---|---|---|
| Audio capture buffering | 100-300ms | Đợi đủ 1 phrase trước khi process |
| ASR (streaming Whisper) | 200-400ms | First partial result |
| ASR final result | +200ms | Sau khi nói xong, finalize |
| MT (NLLB-200 distilled) | 100-300ms | Sequential decode |
| TTS first byte (Cartesia Sonic) | 75-150ms | Streaming TTS |
| Network overhead | 50-200ms | Round trip |
| **Total perceived** | **725-1550ms** | Bottleneck thường ở ASR final |

Đây là **slow** so với simultaneous interpretation human (~3-5 giây delay accepted). End-to-end approaches có thể đạt 200-500ms total.

### 4.5 Production optimizations cho cascaded

Mặc dù chậm hơn end-to-end về theory, cascaded vẫn là default cho production vì:

1. **Modular**: swap any component, version independently.
2. **Easier debugging**: log mỗi stage riêng, identify root cause của error.
3. **Component reuse**: ASR model dùng cho cả ST và transcription.
4. **Cost effective**: each component có thể optimize riêng (quantization, distillation).
5. **Mature tooling**: ESPnet, NeMo, Triton Inference Server đều support cascaded pipeline.

### 4.6 Streaming cascaded variant

Modern voice agents (LiveKit, Vapi.ai, ElevenLabs Conversational AI) dùng **streaming cascaded**:

- ASR: Deepgram Nova-3 streaming (chunk-by-chunk).
- MT: incremental decoding (e.g., LLM với streaming prompt).
- TTS: streaming TTS (Cartesia, ElevenLabs Flash).

Total latency drop xuống ~500-800ms, vẫn cao hơn end-to-end nhưng dễ deploy hơn.

---

## Phần 5 — End-to-End S2TT Approaches

End-to-end (E2E) S2TT giảm error propagation bằng cách train một mô hình duy nhất ánh xạ trực tiếp speech → target text. Phần này deep dive các architectures.

### 5.1 Encoder-Decoder vanilla architecture

Architecture chuẩn từ 2017-2020:

$$
\mathbf{h} = \text{SpeechEncoder}(\mathbf{x}_{\text{src}}), \quad p(\mathbf{y}_{\text{tgt}} | \mathbf{x}_{\text{src}}) = \prod_{i=1}^{|\mathbf{y}|} p(y_i | y_{<i}, \mathbf{h})
$$

- Speech encoder: Conformer hoặc Transformer trên mel spectrogram.
- Decoder: Transformer decoder, cross-attention vào speech encoder output.
- Vocabulary: BPE/SentencePiece của target language.

### 5.2 Multi-task training: ASR + ST joint

Vấn đề của vanilla E2E: cần parallel speech-translation data, ít hơn ASR data nhiều. Solution: multi-task training.

$$
\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{ST}} + (1 - \alpha) \cdot \mathcal{L}_{\text{ASR}}
$$

ASR loss (CTC) là auxiliary task giúp encoder học representations tốt hơn.

### 5.3 Pre-training strategies

#### 5.3.1 Speech encoder pre-training

Wav2Vec 2.0, HuBERT pre-trained trên unlabeled audio → fine-tune cho ST. Hot trick: pre-train MULTILINGUAL encoder (XLS-R, w2v-BERT 2.0) trước khi fine-tune.

#### 5.3.2 Decoder pre-training

mBART, mT5 pre-trained trên parallel text → init decoder của ST model. Encoder warm-start từ MT decoder.

#### 5.3.3 Curriculum learning

1. Pre-train encoder (Wav2Vec 2.0 trên unlabeled audio).
2. Pre-train decoder (mBART trên parallel text).
3. Fine-tune both on ASR (more data than ST).
4. Fine-tune both on ST (less data, but final task).

Đây là recipe SeamlessM4T sử dụng.

### 5.4 Architecture cụ thể: SeamlessM4T v2 deep dive

SeamlessM4T v2 (Meta, 2023, Nature 2024) là SOTA open-source cho multilingual ST. Architecture:

```
[Speech Input]
      ↓
  w2v-BERT 2.0  (600M params, pretrained on 4.5M hours)
      ↓
  Speech-Text Encoder Adapter
      ↓
  Multilingual Shared Encoder  ──────┐
                                       ↓
                              Length Adapter
                                       ↓
   [Text Input] →  Multilingual Text Encoder  → Shared Embeddings
                                       ↓
                              Shared Decoder (T2U Generator)
                                       ↓
                          [Text Output (any task)]
                                       ↓
                         Unit-to-Speech Decoder
                                       ↓
                              [Speech Output]
```

**Key components**:

- **w2v-BERT 2.0**: speech encoder, pre-trained on 4.5M hours of unlabeled audio.
- **NLLB encoder + decoder**: text encoder và decoder, pre-trained on parallel text.
- **Length Adapter**: bridge speech (continuous, long) → text-like representation (discrete-ish, shorter).
- **T2U Generator**: text → discrete acoustic units (HuBERT clusters).
- **Unit Vocoder**: discrete units → waveform.

**Capabilities**: S2TT (101 → 96), S2ST (101 → 36), T2TT (96), T2ST (96 → 36), ASR (101).

**Benchmarks** (FLEURS test set, BLEU on S2TT into English):

| Model | BLEU |
|---|---|
| Whisper Large-v3 + NLLB-3.3B | 25.3 |
| SeamlessM4T v1 (Large) | 28.7 |
| SeamlessM4T v2 (Large) | **30.1** |

(Numbers from public benchmarks, verify before citing in academic work.)

### 5.5 Architecture cụ thể: AudioPaLM 2 (Google, 2023-2024)

Google approach hơi khác: thay vì encoder-decoder, dùng single decoder-only LM trên audio + text tokens unified.

- Audio → AudioLM-style codec tokens (SoundStream).
- Text → BPE tokens.
- Combined vocabulary: text tokens ∪ audio tokens.
- Decoder-only Transformer như PaLM, train next-token prediction.

Advantages: simpler architecture, scaling laws giống LLM. Cũng support T2T, S2T, S2S trong cùng một model.

### 5.6 Q4 2025 / 2026 SOTA updates

Year 2025-2026 mang lại major improvements:

- **Qwen3-Omni (Oct 2025)**: Open source, MoE Thinker-Talker, comparable to Gemini 2.5 Pro on voice translation.
- **Qwen3.5-Omni (Mar 2026)**: Plus variant outperforms Gemini 3.1 Pro on translation benchmarks.
- **GPT-Realtime (Aug 2025)**: OpenAI Realtime API có realtime translation feature, MCP server support.
- **Gemini 3 multimodal Live**: Google streaming translation trong video call.
- **Hibiki (Kyutai, 2025)**: Simultaneous S2ST, open source, low latency.

So sánh chi tiết ở Phần 8.

---

## Phần 6 — Unit-based / Textless S2ST

Phần này về một trong những đột phá conceptual quan trọng nhất của Speech AI 2022-2024: **bỏ text intermediate**.

### 6.1 Tại sao textless?

Hai lý do chính:

1. **Phục vụ ngôn ngữ không có chữ viết**: hàng nghìn ngôn ngữ thiểu số chỉ tồn tại dưới dạng nói.
2. **Tránh information loss**: text là lossy compression của speech (mất prosody, emotion, speaker info). Textless pipeline có thể preserve tốt hơn.

### 6.2 Discrete units: HuBERT k-means

Idea cốt lõi: dùng k-means clustering trên HuBERT features để tạo "pseudo-vocabulary" cho speech.

$$
\mathbf{u}_t = \arg\min_k \|\text{HuBERT}_t(\mathbf{x}) - \mathbf{c}_k\|^2
$$

trong đó $\mathbf{c}_k$ là centroids của k-means (thường $K = 100$ hoặc $200$).

Output: sequence of integers $\mathbf{u} = (u_1, \ldots, u_T)$, mỗi $u_t \in \{1, \ldots, K\}$. Đây là "tokens" cho speech, không cần text.

### 6.3 Direct unit-to-unit S2ST

$$
\text{Speech}_{\text{src}} \xrightarrow{\text{HuBERT k-means}} \text{Units}_{\text{src}} \xrightarrow{\text{Seq2Seq}} \text{Units}_{\text{tgt}} \xrightarrow{\text{Unit Vocoder}} \text{Speech}_{\text{tgt}}
$$

- **HuBERT k-means**: encode speech thành discrete units.
- **Seq2Seq**: Transformer encoder-decoder mapping source units → target units.
- **Unit Vocoder**: HiFi-GAN variant trained để generate waveform từ unit sequence.

### 6.4 Architecture cụ thể: Translatotron 2 và 3

- **Translatotron 1 (2019)**: direct speech-to-spectrogram, no intermediate. Hard to train.
- **Translatotron 2 (2022)**: + linguistic decoder (predict text as auxiliary), much better stability.
- **Translatotron 3 (2024)**: unsupervised approach using contrastive learning, supports low-resource pairs.

### 6.5 Architecture cụ thể: Lee 2022 (Direct S2ST with Discrete Units)

Facebook AI paper: dùng HuBERT k-means để generate target speech units, eliminate text completely.

Pipeline:

1. Train HuBERT trên target language audio.
2. K-means clustering HuBERT features → target units (K=100).
3. Train Seq2Seq từ source speech (mel) → target units.
4. Train Unit Vocoder cho target language.

Performance trên CVSS:

| Approach | ASR-BLEU |
|---|---|
| Cascade (ASR + MT + TTS) | 24.5 |
| Direct S2ST with discrete units | 22.1 |
| Translatotron 2 | 21.8 |

(Numbers approximate, verify in original paper.)

### 6.6 GSLM và Textless NLP

Generative Spoken Language Model (Facebook, 2021) generalized ý tưởng textless: do everything on discrete speech units.

- Speech encoder → units → Transformer LM (training as LM trên units) → unit decoder → speech.
- GSLM 1.0: prosody chưa tốt, output sounds monotonous.
- pGSLM (prosody-aware GSLM, 2022): added explicit prosody modeling.

### 6.7 Limitations của textless approach

- Khó scale lên hàng trăm ngôn ngữ (mỗi pair cần training data).
- Quality vẫn thấp hơn cascaded cho high-resource pairs.
- Hard to evaluate (BLEU không áp dụng trực tiếp, dùng ASR-BLEU vẫn cần ASR).
- Latent leakage qua units (model có thể "learn" text-like representation trong units).

Despite these, textless là **only viable path** cho low-resource và unwritten languages.

---

## Phần 7 — Simultaneous (Streaming) Translation

Simultaneous Translation là bài toán Speech AI thách thức nhất về mặt engineering. Phải dịch token-by-token, không có lookahead, vẫn đạt accuracy đủ tốt. Phần này deep dive các policies và architectures.

### 7.1 Bài toán

Cho streaming source audio, output stream of target tokens với constraint:

- Mỗi target token $y_i$ chỉ được dùng source prefix $x_{1:d(i)}$, không full source.
- Goal: minimize latency (small $d(i)$) AND maximize quality (high COMET / BLEU).

Trade-off này được phân tích bằng latency-quality curves.

### 7.2 Wait-k policy (Ma et al., 2019)

Đơn giản nhất: luôn chờ $k$ source tokens trước khi emit mỗi target token.

$$
d(i) = i + k - 1
$$

Pros: deterministic, easy to implement.
Cons: rigid, không adaptive với content (một số sentence cần nhiều context hơn).

### 7.3 Adaptive policies

Model tự quyết định READ (chờ thêm source) hoặc WRITE (emit target):

$$
\text{action}_i = \arg\max_{a \in \{\text{READ}, \text{WRITE}\}} \pi_\theta(a | \text{state}_i)
$$

Trained với policy gradient (REINFORCE) hoặc supervised với expert demonstrations.

### 7.4 Monotonic Attention

Soft attention với monotonic constraint. Cho phép differentiable training nhưng vẫn enforce streaming property.

$$
p_i^j = \sigma\left(\frac{\mathbf{q}_j \cdot \mathbf{k}_i}{\sqrt{d}}\right), \quad \alpha_i^j = p_i^j \prod_{k<i}(1 - p_k^j)
$$

Trong đó $p_i^j$ là READ probability tại source position $i$ cho target position $j$, $\alpha$ là attention weight (monotonic).

### 7.5 EMMA (Efficient Monotonic Multihead Attention)

SeamlessStreaming dùng EMMA, một variant cải tiến của Monotonic Attention:

- Multi-head với independent monotonic decisions.
- Efficient computation (không phải compute full attention matrix).
- Trained với expectation training thay vì hard sampling.

EMMA cho phép SeamlessStreaming đạt BLEU 26.8 với AL ~3 (so với offline BLEU 30.1).

### 7.6 Hibiki (Kyutai, 2025)

Latest SOTA cho simultaneous S2ST. Kiến trúc:

- Mimi codec (12.5 Hz) cho both source và target.
- Dual-stream Transformer như Moshi, nhưng cho translation context.
- "Inner monologue" inspired by Moshi: predict text token AS prefix to audio token cho linguistic guidance.
- Open source.

Performance (per public Kyutai claims, verify before citing):

| Direction | BLEU | Latency |
|---|---|---|
| En → Fr | ~25 | ~1 sec |
| Fr → En | ~22 | ~1 sec |
| En → Es | ~24 | ~1 sec |

### 7.7 Production simultaneous: latency budget

Trong production (conference interpretation, live caption), budget thường:

- p50 latency: < 1 sec
- p95 latency: < 2 sec
- Quality: > 70% accuracy (human-rated)

Cách typical đạt được: lookahead 200-500ms + Wait-k với $k$ adaptive theo content density.

---

## Phần 8 — SOTA Models 2024-2026 (Comprehensive Survey)

Phần này tổng hợp các mô hình SOTA mới nhất, với chú trọng Q4 2025 và 2026 releases. Mỗi mô hình include architecture, benchmarks, deployment notes.

### 8.1 SeamlessM4T v2 (Meta, Nov 2023 → ongoing)

Đã deep dive ở 5.4. Vẫn là baseline open-source cho multilingual ST đến 2026.

### 8.2 Whisper + NLLB Cascaded (2022-2023, vẫn dùng rộng rãi)

Baseline cascaded mà bất kỳ Speech ST pipeline nào cũng so sánh với:

- ASR: Whisper Large-v3 (99 languages).
- MT: NLLB-200-3.3B (200 languages).
- Optional: GPT-4 instead of NLLB for higher quality.

Pros: Stable, well-tested, open. Cons: Slow, error propagation.

### 8.3 AudioPaLM 2 (Google, 2023-2024)

Đã thảo luận ở 5.5. Decoder-only LM trên audio + text tokens.

### 8.4 Qwen2-Audio (Alibaba, 2024)

- Audio encoder: Whisper-large encoder.
- LLM: Qwen2-7B.
- Capabilities: ASR, audio understanding, S2TT (en/zh/de/es and more).
- Open source.

### 8.5 Qwen2.5-Omni (Alibaba, early 2025)

- Unified text + audio + image + video.
- Thinker-Talker architecture.
- S2TT for ~10 languages.
- Better translation than Qwen2-Audio.

### 8.6 Qwen3-Omni (Alibaba, Oct 2025) ⭐ Major release

- MoE Thinker-Talker (30B-A3B variant).
- 119 text languages, 19 speech input, 10 speech output.
- ASR + audio understanding comparable to Gemini 2.5 Pro.
- Open source Apache 2.0.
- Available variants: Instruct, Thinking (reasoning), Captioner (audio→text).

### 8.7 Qwen3-Omni-Flash (Dec 2025)

- Optimized for latency.
- Real-time streaming speech output.

### 8.8 Qwen3.5-Omni (Mar 2026) ⭐⭐ Latest SOTA

- **215 SOTA results** trên các benchmark âm thanh, video, reasoning.
- **Outperforms Gemini 3.1 Pro** trên general audio understanding, reasoning, translation.
- Plus variant native multimodal (text+image+audio+video single pass).
- Streaming speech output realtime.

### 8.9 GPT-Realtime (OpenAI, Aug 2025)

- Production-grade speech-to-speech.
- General availability of Realtime API.
- New features: MCP server support, image input, SIP phone calling.
- Realtime translation as feature.
- Closed source, pay per minute.

### 8.10 Gemini Live (Google, ongoing 2024-2026)

- Gemini 2.0 Live (2025): streaming voice + video.
- Gemini 3 Live (2026): improved translation, multi-speaker handling.
- Closed source, available via Vertex AI và Gemini API.

### 8.11 Hibiki (Kyutai, 2025)

- Open source simultaneous S2ST.
- Built on Moshi architecture.
- En ↔ Fr, En ↔ Es support.
- Low latency ~1 sec.

### 8.12 StreamSpeech (2024 academic)

- Direct streaming S2ST.
- Two-pass decoding: monotonic for streaming, non-monotonic for re-ranking.
- Strong on FLEURS Streaming benchmark.

### 8.13 Summary Table

| Model | Type | Year | Languages | Streaming | Open Source | Score (approx) |
|---|---|---|---|---|---|---|
| Whisper-v3 + NLLB | Cascaded | 2022-23 | 99 | No | Yes | BLEU 25.3 |
| SeamlessM4T v2 | E2E | 2023 | 100+ | No | Yes | BLEU 30.1 |
| SeamlessStreaming | E2E Streaming | 2023 | 100+ | Yes | Yes | BLEU 26.8 |
| AudioPaLM 2 | E2E LM | 2023 | 100+ | No | No | Higher than Seamless |
| Qwen2-Audio | Multimodal LM | 2024 | 10+ | No | Yes | Comparable to Seamless |
| Qwen2.5-Omni | Multimodal LM | 2025 | 19 | Yes | Yes | Better than Qwen2-Audio |
| Qwen3-Omni | MoE Multimodal | Oct 2025 | 119/19/10 | Yes | Yes | ~ Gemini 2.5 Pro |
| Qwen3.5-Omni | MoE Multimodal | Mar 2026 | 119+ | Yes | Yes | **> Gemini 3.1 Pro** |
| GPT-Realtime | Closed S2S | Aug 2025 | 50+ | Yes | No | Commercial leader |
| Gemini 3 Live | Closed Streaming | 2026 | 50+ | Yes | No | Commercial leader |
| Hibiki | E2E Streaming | 2025 | 4-6 pairs | Yes | Yes | Best open simultaneous |

---

## Phần 9 — Production Considerations

Biết rõ SOTA paper không đồng nghĩa với build được production system. Phần này về những điểm mà paper không nhắc đến nhưng bạn sẽ đụng phải ngay ngày đầu triển khai: latency budget, cost economics, code-switching, hallucination, observability. Đúc kết từ public docs của ElevenLabs Conversational AI, Vapi.ai, LiveKit, và Cartesia.

### 9.1 Latency budget

Production target tuỳ use case:

| Use case | Total latency target |
|---|---|
| Voice agent đa ngôn ngữ | < 1.5 sec |
| Conference live caption | 2-5 sec acceptable |
| Subtitle for video (offline) | Hours, no constraint |
| Phone call interpretation | < 1 sec ideal |

### 9.2 Streaming pipeline architecture

Modern production system (Vapi, ElevenLabs Conversational AI):

```
[Mic Input]
   ↓
WebRTC streaming
   ↓
VAD (Voice Activity Detection) - 50-100ms
   ↓
Streaming ASR (Deepgram Nova-3, Whisper streaming) - first partial 200-400ms
   ↓
Translation (LLM với streaming prompt, OR dedicated MT model) - 200-500ms incremental
   ↓
Streaming TTS (Cartesia Sonic, ElevenLabs Flash) - first byte 75-200ms
   ↓
Audio output streaming
   ↓
[Speaker]
```

### 9.3 Cost economics

Honest numbers based on public pricing (verify before quoting):

| Component | Provider | Cost (per pricing page tháng 11/2025) |
|---|---|---|
| ASR streaming | Deepgram Nova-3 | 0.0036 USD/min |
| ASR streaming | AssemblyAI Universal-2 | 0.0067 USD/min |
| MT/LLM | GPT-4o-mini | 0.15 USD/M input tokens + 0.60 USD/M output tokens |
| MT dedicated | NLLB-200 self-host | ~0.002 USD/min (compute only, on rented A100) |
| TTS streaming | Cartesia Sonic | 0.022 USD/1k chars |
| TTS streaming | ElevenLabs Flash | 0.18 USD/1k chars |
| Voice agent bundled | Vapi.ai estimated | 0.05-0.15 USD/min total |

### 9.4 Code-switching handling

Real-world speech thường mix nhiều ngôn ngữ ("Tôi sẽ deploy cái này lên production"). Handling strategies:

1. **Language ID per chunk**: detect language change, switch ASR/MT.
2. **Multilingual ASR + multilingual decoder**: Whisper handle naturally, but accuracy degrades.
3. **Code-switching specific models**: train on mixed data (rare for production).

### 9.5 Hallucination & error recovery

ASR hallucinate, MT hallucinate, TTS sometimes too. Production needs:

- **Confidence scoring**: drop low-confidence chunks.
- **Repeat detection**: if output identical to last chunk, likely loop.
- **Profanity filtering**: voice agent shouldn't curse.
- **Fallback responses**: "I didn't catch that, can you repeat?"

### 9.6 Observability

Track production metrics:

- **Latency**: p50, p95, p99 cho each component và end-to-end.
- **WER samples**: random samples manually transcribed, compare with ASR.
- **MT quality samples**: random samples human-evaluated.
- **User-side metrics**: completion rate, CSAT, NPS.

### 9.7 Multi-region deployment

Để giảm network latency:

- Deploy GPU inference servers ở multiple regions (US, EU, APAC).
- Route based on user location (CDN-like).
- Audio streaming through nearest WebRTC TURN/STUN server.

---

## Phần 10 — Speech Translation cho Tiếng Việt

Nếu bạn đang ở Việt Nam hoặc phục vụ user Việt, phần này là quan trọng nhất. Đây là honest assessment về state-of-the-art và industry landscape cho Vietnamese ST năm 2026.

### 10.1 Data scarcity reality

Vietnamese ST có khoảng <100h public parallel speech-translation data (so với En-De: >10,000h). Hệ luỵ:

- Dynamic training end-to-end ST cho Vi cần data augmentation aggressive.
- Hầu hết production system Vi dùng cascaded (PhoWhisper + NLLB).
- Một số nhóm tự build private datasets (VinAI ~500h, ZaloAI ~?).

### 10.2 Code-switching En-Vi: common case

Người Việt thường switch giữa Vi và En, đặc biệt trong tech:

> "Tôi cần build cái dashboard này, nhưng performance hơi slow. Maybe nên cache thêm."

Multilingual Whisper handle khá tốt nhưng có hallucination edge cases. PhoWhisper được fine-tune trên Vi-only data, performance kém hơn cho code-switching.

Solution: train ASR trên code-switched data (CMC challenge có dataset này), hoặc dùng MIVA (multilingual ASR + LM rescoring).

### 10.3 Vietnamese industry landscape

#### 10.3.1 VinAI (VinBigData)

- **PhoWhisper** (2024): Whisper fine-tuned trên 844h Vi data, SOTA trên VLSP.
- **ViBERT**, **PhoGPT**: text LMs cho Vietnamese.
- **ViVi assistant**: in-car voice cho VinFast.
- Open source nhiều mô hình.

#### 10.3.2 ZaloAI

- **KiKi assistant**: voice assistant cho user Zalo, hỗ trợ Vi commands.
- **ZaloPay voice**: voice commands cho ZaloPay app.
- Private models, không công bố benchmarks public.

#### 10.3.3 FPT.AI

- **FPT Speech**: ASR + TTS commercial API.
- Multilingual support (Vi, En primarily).
- Có chatbot infrastructure.

#### 10.3.4 VinFast

- In-car voice assistant cho ô-tô VinFast.
- Vi + En multilingual.
- Tích hợp với navigation, climate, music.

#### 10.3.5 Trusting Social

- Voice biometrics cho KYC verification.
- Multimodal identity (voice + face + ID document).
- Production cho banks at scale in Vi.

### 10.4 Realistic assessment of Vietnamese ST quality

Honest pinion based on public benchmarks và personal testing:

- **Vi → En S2TT**: PhoWhisper + NLLB đạt ~25-28 BLEU trên CMV-Vi test (estimate). Adequate cho most use cases.
- **En → Vi S2TT**: Whisper + NLLB → Vi text, quality decent nhưng có issues với:
  - Vietnamese tones not captured properly trong some MT outputs.
  - Foreign name transliteration (tên người nước ngoài chuyển sang Vi inconsistent).
  - Vietnamese accent variations (Bắc/Trung/Nam) gây WER variation.
- **Vi → En S2ST**: chưa có public model SOTA, hầu hết cascaded với TTS Vi → MT En → TTS En.

### 10.5 Khuyến nghị cho Vietnamese ST production

1. **Cascaded baseline**: PhoWhisper (ASR Vi) + NLLB-200 hoặc GPT-4o-mini (MT) + ElevenLabs Multilingual (TTS En).
2. **Streaming**: ASR streaming với Deepgram Nova hoặc Whisper streaming, MT incremental, TTS Cartesia.
3. **Voice cloning Vi**: XTTS v2 fine-tuned trên Vi data.
4. **Multimodal Vi-aware**: Qwen3-Omni đã support Vi text input, audio output gần đủ.

---

## Phần 11 — Limitations & Open Problems

Không có hệ thống AI nào hoàn hảo. Phần này thẳng thắn về những điểm mà Speech Translation vẫn chưa giải được tốt vào năm 2026. Hiểu rõ limitations giúp bạn set expectation đúng khi build product, và định hướng research nếu bạn là academic.

### 11.1 Low-resource pairs

Direct E2E ST cho cặp low-resource (vd Hindi → Swahili) vẫn rất khó. Best approach hiện tại: pivot through English. Quality degraded đáng kể.

### 11.2 Voice preservation in S2ST

Khi dịch speech sang ngôn ngữ khác, giữ giọng người nói gốc là challenge. Translatotron 2 thử nhưng quality thấp. SeamlessExpressive (Meta 2024) cải tiến nhưng vẫn far from human dubbing quality.

### 11.3 Prosody transfer

Speech mang nhiều thông tin prosody (emphasis, intonation, pause). Hầu hết ST systems lose prosody, output sounds monotonous.

### 11.4 Long-form translation

Hầu hết models được train trên utterances 5-30 sec. Long form (lectures, audiobooks, hour-long videos) cần segmentation + context tracking, model hiện tại weak ở point này.

### 11.5 Hallucination

Mọi neural translation system đều hallucinate. Whisper notoriously hallucinate trong silence periods. NLLB occasionally invent facts. Production system cần guardrails (confidence threshold, repetition detection).

### 11.6 Domain shift

ST trained on TED talks fails on legal proceedings, medical conversations, technical jargon. Domain adaptation cần fine-tuning data which doesn't exist for many domains.

### 11.7 Real-time TTS cho mọi ngôn ngữ

TTS streaming SOTA (Cartesia, ElevenLabs) hỗ trợ ~50-100 languages. Còn 7,000+ ngôn ngữ chưa có streaming TTS. Bottleneck cho ST end-to-end ở low-resource.

---

## Phần 12 — Tóm tắt & Roadmap

### 12.1 Những bài học chính

1. **Speech Translation là gia đình bài toán, không phải bài toán đơn**: S2TT, S2ST, SimulST với latency modes khác nhau, kỹ thuật khác nhau.

2. **BLEU đã obsolete cho ST**: dùng COMET, BLASER 2.0 cho academic. Production dùng human eval + latency p95.

3. **Cascaded (Whisper + NLLB + TTS) vẫn dominant production 2026** vì modularity, debuggability, mature tooling. End-to-end win về quality nhưng harder to deploy.

4. **SeamlessM4T v2** vẫn là baseline open-source. **Qwen3.5-Omni (Mar 2026)** là current SOTA, vượt cả Gemini 3.1 Pro trên benchmarks.

5. **Simultaneous translation** là hardest sub-problem, requires architectural innovation (EMMA, monotonic attention). Best open: **Hibiki** (Kyutai 2025).

6. **Textless / unit-based S2ST** mở khả năng phục vụ ngôn ngữ không có chữ viết, đặc biệt quan trọng cho social impact.

7. **Vietnamese ST** vẫn dependent on cascaded (PhoWhisper + NLLB). Major opportunity cho industry research.

### 12.2 Recommended reading

- SeamlessM4T v2 paper (Nature 2024).
- Hibiki technical report (Kyutai 2025).
- Qwen3-Omni technical report (Alibaba, Oct 2025).
- AudioPaLM 2 (Google).
- GPT-Realtime API docs (OpenAI).

### 12.3 Chương tiếp theo

Chương 16 (Vietnamese Speech Processing) và Chương 17 (Vietnamese Datasets) sẽ deep dive thêm Vietnamese-specific concerns. Chương 20 (Production Speech Systems) sẽ thảo luận production stack chi tiết hơn.

---

## Tài liệu tham khảo

1. Communication, S. et al. (2023). SeamlessM4T: Massively Multilingual & Multimodal Machine Translation. arXiv:2308.11596.
2. Communication, S. et al. (2024). Joint speech and text machine translation for up to 100 languages. Nature.
3. Communication, S. et al. (2023). Seamless: Multilingual Expressive and Streaming Speech Translation. arXiv:2312.05187.
4. Lee, A. et al. (2022). Direct Speech-to-Speech Translation with Discrete Units. ACL.
5. Jia, Y. et al. (2019). Direct Speech-to-Speech Translation with a Sequence-to-Sequence Model. Interspeech.
6. Jia, Y. et al. (2022). Translatotron 2: High-quality Direct Speech-to-Speech Translation with Voice Preservation. ICML.
7. Ma, M. et al. (2019). STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency. ACL.
8. Rei, R. et al. (2020). COMET: A Neural Framework for MT Evaluation. EMNLP.
9. Saeki, T. et al. (2022). UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022. Interspeech.
10. Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). OpenAI.
11. Le, T. et al. (2024). PhoWhisper: Automatic Speech Recognition for Vietnamese. ICLR Tiny Papers.
12. Qwen Team (Oct 2025). Qwen3-Omni Technical Report. Alibaba.
13. OpenAI (Aug 2025). Introducing gpt-realtime and Realtime API updates for production voice agents.
14. Borsos, Z. et al. (2023). AudioLM: A Language Modeling Approach to Audio Generation. IEEE/ACM TASLP.
15. Wang, C. et al. (2023). VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. arXiv:2301.02111.
