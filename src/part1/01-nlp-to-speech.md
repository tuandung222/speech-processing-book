# Chương 1: Từ NLP đến Speech — Cầu nối Khái niệm

## Mở đầu: Một câu hỏi cho data scientist NLP

Hãy bắt đầu bằng một câu hỏi đơn giản: **vào năm 2026, tại sao một data scientist nền tảng NLP/LLM lại cần học Speech AI?**

Câu trả lời ngắn gọn: vì ranh giới giữa "text intelligence" và "audio intelligence" đang biến mất với tốc độ chóng mặt. Hãy nhìn vào bức tranh hiện tại.

GPT-4o của OpenAI có thể nghe người dùng nói, ngắt lời lịch sự, đổi giọng theo cảm xúc, và phản hồi trong vòng 320 mili-giây. Đó là độ trễ tương đương một cuộc trò chuyện thực giữa con người. Moshi của Kyutai Labs đi xa hơn: nó có thể nói và nghe **đồng thời**, không phải kiểu "đợi user kết thúc rồi mới response" như các chatbot truyền thống. Qwen2.5-Omni của Alibaba thống nhất bốn modality (text, audio, image, video) trong một transformer duy nhất. Gemini 2.0 streaming hỗ trợ video gọi thời gian thực.

Điểm chung của tất cả những hệ thống này: chúng đều xây dựng trên cùng một kiến trúc transformer mà bạn đã quen thuộc khi training BERT, fine-tune GPT, hay deploy LLaMA. Khác biệt duy nhất nằm ở **cách input và output được biểu diễn**.

Đây chính là điều cuốn sách này muốn giải thích. Và chương đầu tiên, cụ thể, sẽ xây dựng cây cầu khái niệm cho phép bạn ánh xạ từng kiến thức NLP đã có vào thế giới speech một cách trực giác. Sau khi đọc xong chương này, bạn sẽ thấy:

> Speech AI = NLP + hai bước biến đổi (continuous → discrete) + các kỹ thuật xử lý tín hiệu.

Ba phần còn lại của Part I (Chương 2-3) sẽ đi sâu vào hai bước biến đổi đó. Phần II về sau là ASR. Phần III là TTS. Phần IV về Speech LLMs là điểm mà tất cả những gì bạn biết về Transformer thực sự được áp dụng trực tiếp lên audio. Cuốn sách được thiết kế để mỗi chương đều bắc cầu sang NLP, không bao giờ để bạn cảm thấy bị lạc giữa các thuật ngữ xa lạ.

> **📝 Lưu ý về phong cách trình bày**
>
> Cuốn sách này được viết cho người đã thành thạo Transformer, đã từng pretrain hoặc fine-tune ít nhất một LLM, đã hiểu attention mechanism ở mức implementation. Nếu bạn chưa có những kiến thức đó, hãy đọc bài "The Illustrated Transformer" của Jay Alammar trước, sau đó quay lại đây.
>
> Mỗi khi gặp một khái niệm Speech mới, chương sẽ dừng lại và bắc cầu sang NLP equivalent. Người đọc sẽ thấy rằng "thư viện trực giác" được xây dựng từ NLP không hề bị vứt bỏ, nó chỉ cần được mở rộng.

## Phần 1 — Vấn đề cốt lõi: Discrete vs Continuous

### 1.1 Thế giới NLP: bạn đã quen thuộc với gì?

Trước khi nói về speech, hãy ôn lại pipeline NLP cơ bản mà bạn đã thực hiện hàng nghìn lần. Khi đưa một câu văn vào model, các bước xảy ra:

1. **Tokenization**: chuỗi ký tự `"Tôi yêu Speech AI"` được tách thành các token. Với BPE/SentencePiece, kết quả có thể là `["Tôi", " yêu", " Speech", " AI"]` hoặc tinh hơn `["Tôi", " y", "êu", " Speech", " AI"]` tuỳ tokenizer.

2. **Lookup vocabulary**: mỗi token được map sang một integer ID. Vocabulary của LLaMA-2 có 32,000 entries, GPT-4 có khoảng 100K, các tokenizer multilingual có thể đến 250K.

3. **Embedding**: mỗi token ID được tra cứu trong embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$, trả về vector $d$ chiều (thường $d = 768$ cho BERT-base, $4096$ cho LLaMA-7B).

4. **Positional encoding**: thêm thông tin vị trí (sinusoidal hoặc learned hoặc RoPE).

5. **Transformer layers**: stack of self-attention + feedforward, mỗi layer biến đổi sequence of embeddings thành sequence of contextualized representations.

6. **Output**: tuỳ task (classification head, language modeling head, etc.).

Pipeline này có vài đặc điểm đáng để gọi tên rõ ràng:

- **Input là discrete và có vocabulary cố định**. Khi bạn build tokenizer, bạn quyết định trước rằng vocabulary có 32K entries (hoặc 50K, hoặc 100K). Mỗi token bạn nhìn thấy CHẮC CHẮN phải nằm trong tập này (out-of-vocabulary đã được giải quyết bằng BPE/SentencePiece chia thành sub-units).
- **Mỗi token có nghĩa độc lập (gần đúng)**. Token `"Speech"` mang nghĩa "speech" dù ở câu nào. Tokenization là deterministic.
- **Sequence length tương đối ngắn**. Một câu văn dài ~20 tokens, một paragraph ~100, một đoạn văn dài ~500. Context windows hiện đại (Gemini 1.5: 2M, Claude 3.5: 200K) đã rất dài nhưng vẫn finite.
- **Information density cao**. Mỗi token mang nhiều thông tin ngữ nghĩa, không có token nào "redundant" theo nghĩa lý thuyết thông tin.

Hãy giữ những đặc điểm này trong đầu khi ta nhìn sang thế giới speech.

### 1.2 Thế giới Speech: vấn đề là gì?

Bây giờ tưởng tượng input là âm thanh thay vì text. Cụ thể, một file `.wav` ghi giọng người nói câu "Tôi yêu Speech AI" trong 1.5 giây. Bạn cần đưa file này vào một neural network. Bước đầu tiên là gì?

Câu trả lời không hề hiển nhiên, và sự không-hiển-nhiên đó chính là nguồn cơn của toàn bộ ngành Speech AI.

Hãy nhìn vào file đó dưới ống kính kỹ thuật. Microphone bắt sóng âm, một sóng cơ học truyền qua không khí, và biến đổi thành tín hiệu điện. Tín hiệu điện này được số hoá qua **sampling** (lấy mẫu) và **quantization** (lượng tử hoá). Với chuẩn 16 kHz / 16-bit (thông số mặc định cho speech), 1 giây audio = 16,000 mẫu, mỗi mẫu là một số nguyên signed 16-bit (-32768 đến 32767).

Vậy file 1.5 giây của bạn = 24,000 mẫu. Một mảng `int16` có shape `(24000,)`.

> **🤔 Câu hỏi tự nhiên đầu tiên**
>
> "Vậy mỗi mẫu là một token, đúng không? Vocabulary size = $2^{16}$ = 65,536, tương tự BPE token. Ta cứ embedding rồi đưa vào Transformer như NLP thôi mà?"
>
> Câu trả lời là **KHÔNG**, và để hiểu tại sao không, ta cần thấy một điều: **mỗi audio sample không mang ngữ nghĩa độc lập**.

Hãy nghĩ thử: số `0` trong mảng audio có nghĩa gì? Đó là một điểm trên sóng âm tại thời điểm đó, biên độ bằng không (zero crossing). Không cho ta biết âm thanh đó là "Tôi" hay "yêu" hay "AI". Số `1000` thì sao? Cũng vậy, chỉ là một giá trị tại một điểm, không mang ngữ nghĩa.

So sánh với NLP: token `"yêu"` (ID 12345 trong vocab) độc lập với context vẫn mang nghĩa "yêu". Embedding của nó trong $\mathbf{E}$ chứa thông tin ngữ nghĩa được học từ pretraining.

Audio sample thì khác hoàn toàn. Một âm vị (phoneme) như /a/ trong "Tôi" trải dài khoảng 80-120 mili-giây, tương đương **1,280 đến 1,920 samples** ở 16 kHz. Nói cách khác, để biết bạn đang nói âm /a/, model cần "nhìn" khoảng 1500 samples liên tiếp, KHÔNG phải 1 sample.

Đây là khác biệt cốt lõi giữa NLP và Speech: **information density của audio thấp hơn text vài bậc**. Một token text xấp xỉ 1 phoneme, nhưng 1 token text = 1 unit trong sequence, còn 1 phoneme = ~1500 samples trong sequence audio.

Hậu quả là:

- **Sequence length của audio cực dài**. 10 giây audio = 160,000 samples. So với 10 giây speech được transcribe ra text khoảng 40 tokens. Tỉ lệ 4000:1.
- **Self-attention $O(L^2)$ trở thành thảm hoạ**. Nếu cứ vô tư embedding 160K samples rồi cho qua transformer, FLOPs sẽ vượt cả LLM training. Đây là lý do mọi kiến trúc Speech AI đều có bước **downsampling** trước transformer.
- **Mỗi sample không có ngữ nghĩa riêng**. Không thể "embedding lookup" giống NLP vì không có vocabulary cố định.

### 1.3 Câu hỏi trung tâm của Speech AI

Từ các quan sát ở 1.1 và 1.2, ta có thể phát biểu vấn đề cốt lõi của Speech AI:

> **Cho một dòng audio liên tục (continuous signal) với mật độ thông tin thấp và sequence length cực dài, làm thế nào để biến nó thành một sequence các đơn vị discrete + có ngữ nghĩa, đủ ngắn để Transformer xử lý hiệu quả?**

Nếu giải được bài toán này, mọi kỹ thuật bạn đã biết về Transformer (attention, masking, autoregressive generation, fine-tuning, RLHF, ...) đều áp dụng được trực tiếp lên audio. Đó là lý do ngày nay ta có Moshi, GPT-4o voice, Qwen2-Audio: chúng đều là LLM tiêu chuẩn, chỉ khác ở "tokenizer" của chúng tinh vi hơn BPE.

Ba "tokenizer cho audio" sẽ được thảo luận chi tiết ở Phần 2 của chương này.

> **💡 Bài học trừu tượng**
>
> Mọi modality không phải text (image, audio, video, point cloud) đều phải trải qua một bước "tokenization" trước khi vào LLM. Khác biệt giữa các modality nằm ở chỗ tokenizer được thiết kế thế nào.
>
> - **Text**: BPE/SentencePiece (statistical, deterministic, lossless).
> - **Image**: ViT patches + linear projection (deterministic, lossy if low resolution), hoặc VQ-VAE tokens (learned, lossy, discrete).
> - **Audio**: mel spectrogram (handcrafted), Wav2Vec/HuBERT features (learned, continuous), EnCodec/Mimi codec tokens (learned, discrete).
>
> Nhận thức này quan trọng: nó cho thấy Speech AI **không phải một ngành riêng biệt**. Nó là một ứng dụng của paradigm "tokenize-then-transformer" với tokenizer được đặc-thù-hoá cho audio.

## Phần 2 — Ba con đường biểu diễn audio

Ba con đường này giải bài toán trung tâm theo ba cách khác nhau, mỗi cách có trade-off riêng. Chúng được sắp xếp theo thứ tự lịch sử (handcrafted → learned continuous → learned discrete), và cũng là thứ tự tăng dần độ "tương đồng với NLP".

### 2.1 Con đường 1 — Spectrogram (handcrafted feature)

Đây là cách tiếp cận lâu đời nhất, có gốc rễ từ kỹ thuật xử lý tín hiệu (DSP) từ thế kỷ 20, và vẫn còn được dùng rộng rãi đến năm 2026 (Whisper, Tacotron, FastSpeech đều dùng).

**Ý tưởng cốt lõi**: thay vì xử lý waveform thô (160K samples cho 10 giây), ta biến đổi nó thành ma trận 2D giống như một bức ảnh thấp tầng. Mỗi cột của ma trận đại diện cho một "frame" (khung thời gian ngắn, ~25 ms), và mỗi hàng đại diện cho một dải tần số.

**Pipeline**:

$$
\text{Waveform} \xrightarrow{\text{STFT}} \text{Complex spectrum} \xrightarrow{|\cdot|^2} \text{Power spectrum} \xrightarrow{\text{Mel filterbank}} \text{Mel spectrogram} \xrightarrow{\log} \text{Log-mel spectrogram}
$$

Chi tiết kỹ thuật của từng bước sẽ được giải thích kỹ ở Chương 2. Ở đây, ta chỉ cần hiểu kết quả cuối cùng:

**Output**: ma trận có shape $(n_{\text{mels}}, T_{\text{frames}})$. Ví dụ với Whisper: $(80, 3000)$ cho 30 giây audio = 80 mel bands × 3000 frames @ 100 fps.

Mỗi giá trị trong ma trận là một số thực (float32), đại diện cho năng lượng ở dải tần số đó tại thời điểm đó.

> **🔄 Analogy với NLP / Image**
>
> Mel spectrogram giống nhất với **image patches của ViT**:
>
> - Ảnh raw (224 × 224 × 3 = 150K pixels) → patches (14×14 grid of 16×16 patches) → embedding sequence (196 tokens).
> - Audio raw (160K samples) → mel spec (80 × 1000) → có thể flatten thành sequence of 1000 vectors mỗi vector 80-dim.
>
> Cả hai đều là "deterministic, handcrafted/fixed feature extraction" để rút gọn sequence length trước khi vào Transformer.

**Ưu điểm**:

- Deterministic, không cần training.
- Tương đối "physically grounded": mel scale phản ánh cách tai người cảm nhận tần số.
- Được hiểu rõ, debug dễ, audio engineer kinh nghiệm có thể "đọc" spectrogram bằng mắt.
- Sequence length giảm 160x so với raw (160000 → 1000 frames cho 10 giây).

**Nhược điểm**:

- Mất thông tin phase (chỉ giữ magnitude), cần vocoder (HiFi-GAN, BigVGAN) để tái tạo waveform.
- Không "học được": feature fixed, không thể thích ứng với task cụ thể.
- Vẫn là continuous (float values), không phải discrete tokens, không thể dùng cho autoregressive LM trực tiếp.

**Khi nào dùng**: encoder side của ASR (Whisper, Conformer-based ASR), encoder side của TTS (Tacotron 2 nhận text, sinh ra mel; vocoder biến mel thành wave). Hầu hết các pipeline production trước 2023 đều dùng mel spectrogram làm interface chuẩn.

**Code minh hoạ** (sẽ giải thích chi tiết ở Chương 2):

```python
import torch
import torchaudio
from torch import Tensor

def waveform_to_mel(
    waveform: Tensor,       # [1, T_samples] - float32, range [-1, 1]
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,       # 25 ms window
    hop_length: int = 160,  # 10 ms hop
) -> Tensor:
    """Biến waveform thành log-mel spectrogram.

    Returns:
        log_mel: [1, n_mels, T_frames] - float32
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel: Tensor = mel_transform(waveform)
    log_mel: Tensor = torch.log(mel + 1e-10)
    return log_mel


# Ví dụ: 1 giây audio random
waveform = torch.randn(1, 16000)
log_mel = waveform_to_mel(waveform)
print(log_mel.shape)  # torch.Size([1, 80, 100])
```

Pipeline này chuyển 16,000 floats thành 8,000 floats (80 × 100). Compression ratio 2:1 nếu tính theo số values, nhưng quan trọng hơn là sequence length giảm 160x (160 samples per frame).

### 2.2 Con đường 2 — Self-supervised continuous representations (Wav2Vec 2.0, HuBERT)

Năm 2020, Wav2Vec 2.0 từ Meta AI đề xuất một ý tưởng đột phá: thay vì dùng spectrogram handcrafted, hãy **học** một biểu diễn audio tốt hơn bằng self-supervised pretraining trên hàng nghìn giờ audio không có nhãn.

**Ý tưởng cốt lõi**: train một transformer encoder trên một pretext task không cần label, ép nó phải hiểu cấu trúc temporal và phonetic của speech. Sau pretraining, các activation của encoder là **contextualized speech representations**, tương tự BERT embeddings cho text.

**Pipeline**:

$$
\text{Waveform} \xrightarrow{\text{CNN feature extractor}} \text{Latent } \mathbf{Z} \in \mathbb{R}^{T' \times d} \xrightarrow{\text{Transformer}} \text{Contextualized } \mathbf{C} \in \mathbb{R}^{T' \times d}
$$

Bước 1 (CNN feature extractor): giảm sequence length từ 160K samples xuống 500 latent vectors (downsampling 320x), mỗi vector 512-dim. Đây là bước downsample mạnh nhất.

Bước 2 (Transformer encoder): xử lý 500 vectors qua 12-24 layers self-attention, output 500 contextualized vectors 768-dim (Wav2Vec 2.0 Base) hoặc 1024-dim (Large).

**Pretext task** (cách self-supervised pretrain): mask một số positions trong latent space, ép model predict masked positions từ context xung quanh. Loss là contrastive (chọn đúng vector từ tập distractors).

> **🔄 Analogy mạnh với BERT MLM**
>
> Wav2Vec 2.0 chính là **BERT cho speech**:
>
> | | BERT (text) | Wav2Vec 2.0 (speech) |
> |---|---|---|
> | Input | Raw text → BPE tokens | Raw waveform → CNN latents |
> | Pretext task | Mask 15% tokens, predict | Mask 6.5% latent steps, predict via contrastive |
> | Loss | Cross-entropy | InfoNCE (contrastive) |
> | Pretrain corpus | BookCorpus + Wikipedia (~16GB) | LibriSpeech 960h + Libri-Light 60K h unlabeled |
> | Output | 768-dim contextualized embeddings | 768-dim contextualized speech reps |
> | Downstream | Fine-tune cho classification, NER, QA | Fine-tune cho ASR, speaker ID, emotion |
> | Vocabulary | 30K BPE tokens | KHÔNG có vocabulary explicit (continuous) |
>
> Khác biệt duy nhất ở mức conceptual: Wav2Vec 2.0 output là continuous vectors (không phải logits over vocabulary), nên không thể dùng trực tiếp cho LM. Để dùng cho LM, ta cần một bước nữa (codec quantization, sẽ thảo luận ở 2.3).

**Output**: sequence $(T', d)$ ở khoảng 50 fps. Với 10 giây audio, ta có ma trận $(500, 768)$.

**Ưu điểm**:

- Học được features tốt hơn handcrafted spectrogram (đo bằng downstream task performance).
- Tận dụng được khối lượng unlabeled audio khổng lồ (LibriLight có 60,000 giờ, VoxPopuli có 400,000 giờ).
- Một backbone, nhiều downstream tasks (giống BERT): ASR, speaker ID, emotion recognition, language ID.

**Nhược điểm**:

- Vẫn là continuous (768-dim float vectors per frame), không discrete, nên không LM-friendly.
- Cần GPU mạnh để pretrain (Wav2Vec 2.0 Large dùng 128 V100 trong nhiều ngày).
- Domain gap có thể là vấn đề (pretrain trên LibriSpeech tiếng Anh, fine-tune cho tiếng Việt cần adaptation).

**Khi nào dùng**: encoder side của ASR khi muốn replace mel spectrogram bằng learned features. PhoWhisper cho tiếng Việt fine-tune từ Whisper, nhưng nhiều system commercial dùng Wav2Vec 2.0 cho low-resource languages.

### 2.3 Con đường 3 — Neural codec tokens (EnCodec, Mimi)

Đây là con đường mới nhất (2022-2024) và là cầu nối trực tiếp đến Speech LLMs. Nó giải vấn đề cuối cùng của con đường 2: làm sao biến audio thành **discrete tokens** giống BPE để dùng autoregressive LM?

**Ý tưởng cốt lõi**: train một autoencoder neural với vector quantization (VQ-VAE variant) ở bottleneck. Encoder nén audio thành chuỗi codebook indices (integer IDs), decoder reconstruct lại waveform từ indices. Khi inference, ta dùng indices như "tokens" giống BPE.

Cụ thể, EnCodec (Meta, 2022) và Mimi (Kyutai, 2024) đều dùng **Residual Vector Quantization (RVQ)**: thay vì 1 codebook, dùng nhiều codebooks xếp tầng, mỗi codebook lượng tử hoá residual của codebook trước.

**Pipeline**:

$$
\text{Waveform} \xrightarrow{\text{Encoder (CNN+conv1d)}} \text{Latent } \mathbf{z} \xrightarrow{\text{RVQ}} \text{Token codes } \{q_1, q_2, \ldots, q_Q\} \in \{0, \ldots, C-1\}^Q
$$

trong đó $Q$ là số codebooks (8 với EnCodec, 8 với Mimi), $C$ là codebook size (1024).

**Output**: ma trận integer $(Q, T'')$. EnCodec ở 75 fps: 10 giây audio → $(8, 750)$. Mimi ở 12.5 fps: 10 giây → $(8, 125)$, gọn hơn nhiều, đây là một trong các đột phá của Moshi.

> **🔄 Analogy quan trọng nhất của cuốn sách: BPE ↔ Codec tokens**
>
> Nếu bạn chỉ nhớ một analogy duy nhất từ chương này, hãy nhớ analogy này.
>
> | Khía cạnh | BPE / SentencePiece | Neural Codec (EnCodec, Mimi) |
> |---|---|---|
> | Input | Raw text string | Raw audio waveform |
> | Output | Sequence of integer token IDs | Sequence of integer codebook indices |
> | Algorithm | Statistical merge (Byte-Pair Encoding) | Neural autoencoder + VQ |
> | Vocabulary | Fixed size (32K-128K typical) | Fixed codebook size × layers (1024 × 8 = 8192 effective) |
> | Reversible | Lossless (token IDs → exact text) | Near-lossless (perceptual, có degradation nhẹ) |
> | Compositional | Subwords compose chars | Codebooks compose acoustic features |
> | LM compatible | Yes (GPT, BERT, T5) | **Yes (VALL-E, AudioLM, Moshi)** |
> | Training data | Text corpus | Audio corpus (CommonVoice, AudioSet) |
> | Token rate | ~1-2 tokens/word ≈ 4 tokens/sec speech | 75-150 codec tokens/sec audio |
>
> Cốt lõi: **codec tokens biến audio thành language modeling problem**. Một khi audio đã thành chuỗi integer, mọi thứ về Transformer LM (next-token prediction, beam search, KV cache, RLHF, ...) đều áp dụng được như NLP.

**Ưu điểm**:

- Discrete tokens, dùng được autoregressive LM (GPT-style).
- Compression ratio cực cao (10x-30x so với waveform).
- Reconstructable waveform với chất lượng cao (Mimi 1.1 kbps gần như chất lượng telephone codec ở 24 kbps).
- Unified interface cho speech LLMs (Moshi, AudioLM, VALL-E).

**Nhược điểm**:

- Lossy (có chút degradation perceptual).
- Cần train codec trên domain phù hợp (codec train trên tiếng Anh có thể bị artifact trên tiếng Việt).
- Số tokens nhân với số codebooks → sequence length effective vẫn dài (8 codebooks × 750 frames = 6000 effective tokens cho 10 giây).
- Phụ thuộc vào quality của decoder để reconstruct.

**Khi nào dùng**: bất kỳ system Speech LLM nào (Moshi, AudioLM, VALL-E, Qwen2.5-Omni). Cũng dùng cho streaming codec cho low-bitrate transmission (Mimi 1.1 kbps).

### 2.4 So sánh ba con đường

Bảng dưới đây tổng hợp ba con đường để bạn quyết định khi nào dùng cái nào:

| Tiêu chí | Mel spectrogram | Wav2Vec 2.0 / HuBERT | Neural codec (EnCodec, Mimi) |
|---|---|---|---|
| **Output type** | Continuous matrix 2D | Continuous sequence 1D | Discrete integers 2D |
| **Shape (10s audio)** | (80, 1000) float32 | (500, 768) float32 | (8, 750) int32 |
| **Data per sec** | 32 KB | 154 KB | 6 KB |
| **Frame rate (fps)** | 100 | 50 | 75 (EnCodec) hoặc 12.5 (Mimi) |
| **Training needed** | No (handcrafted) | Yes (60K+ hours unlabeled) | Yes (audio + reconstruction loss) |
| **Reversible to wave** | Lossy (qua HiFi-GAN vocoder) | No (continuous, không có decoder explicit) | Yes (built-in decoder) |
| **LM compatible** | No (continuous) | No (continuous) | **Yes** (discrete) |
| **Used by** | Whisper, Tacotron 2 | XLS-R, multilingual ASR | VALL-E, AudioLM, Moshi |
| **NLP analogy** | ViT image patches | BERT embeddings | BPE tokens |

> **🎯 Quy tắc lựa chọn (rule of thumb)**
>
> - Build ASR truyền thống → **mel spectrogram** (Whisper-style).
> - Build ASR low-resource hoặc cross-lingual → **Wav2Vec 2.0 / XLS-R** (better transfer).
> - Build Speech LLM (text + audio, conversational, generative) → **neural codec tokens** (Moshi-style).
> - Build TTS modern → **codec tokens + autoregressive transformer** (VALL-E) hoặc **mel + diffusion** (NaturalSpeech 3, F5-TTS).

Một xu hướng quan trọng năm 2024-2026: nhiều system "lai" dùng cả 3. Ví dụ AudioLM dùng:

- Semantic tokens (từ Wav2Vec/HuBERT) cho "what is being said".
- Acoustic tokens (từ EnCodec/Mimi) cho "how it is being said" (giọng, nhịp).
- Mel spectrogram cho training vocoder side.

Hiểu rõ ưu nhược điểm của từng con đường giúp bạn đọc paper modern một cách thông thái.

## Phần 3 — NLP↔Speech Concept Mapping (Big picture)

Phần này là bảng tham chiếu trung tâm của cuốn sách. Mỗi khái niệm NLP quen thuộc đều có đối ứng trong Speech. Bạn nên đánh dấu phần này để quay lại tra cứu khi đọc các chương sau.

### 3.1 Bảng mapping đầy đủ

| NLP Concept | Speech Equivalent | Ghi chú nhanh |
|---|---|---|
| **Token** | Audio frame / codec token | 1 token ≈ 10-80 ms audio |
| **BPE tokenizer** | Mel spectrogram + neural codec | Spectrogram handcrafted, codec learned |
| **Embedding layer** | Audio encoder (CNN+Linear) | Maps raw signal → latent |
| **Vocabulary $V$** | Codebook size $C$ × codebooks $Q$ | $1024 \times 8 = 8192$ effective |
| **Vocabulary lookup** | RVQ nearest-neighbor search | Tìm code gần nhất trong codebook |
| **BERT pretraining** | Wav2Vec 2.0 / HuBERT | Masked prediction self-supervised |
| **GPT (causal LM)** | AudioLM / VALL-E / Moshi | Next-token prediction on codec tokens |
| **T5 / BART (seq2seq)** | Whisper (ASR) / VITS (TTS) | Encoder-decoder |
| **Sequence classification** | Audio classification | Speaker ID, emotion, language ID |
| **Token classification** | CTC frame classification | Mỗi frame → 1 character |
| **Cross-attention** | Encoder-decoder attention | Text conditions speech (TTS) |
| **Causal mask** | Streaming mask | Limited look-ahead trong ASR streaming |
| **Bidirectional attention** | Offline ASR encoder | Toàn bộ audio đã có sẵn |
| **Positional encoding** | Time encoding (RoPE, sinusoidal) | Pretty much same techniques |
| **KV cache** | KV cache (literally same) | Autoregressive speech gen |
| **Beam search** | Beam search (literally same) | ASR/TTS decoding |
| **Top-k / Top-p sampling** | Top-k / Top-p sampling | Speech generation diversity |
| **Temperature** | Temperature | Same |
| **Perplexity** | Perplexity on codec tokens | LM eval |
| **BLEU / ROUGE** | WER (ASR), MOS (TTS) | Task-specific metrics |
| **Accuracy** | WER inverted | ASR success rate |
| **F1 score** | F1 for keyword spotting | Same idea |
| **Fine-tuning** | Fine-tuning (literally same) | LoRA, adapter work on speech |
| **PEFT / LoRA** | PEFT / LoRA | Same techniques |
| **RLHF / DPO** | Speech RLHF (mới năm 2024) | Áp dụng cho TTS quality, voice cloning |
| **Distillation** | Distil-Whisper, NeMo distill | Same technique |
| **Quantization (INT8)** | Quantization (INT8) | Same techniques |
| **vLLM, TGI inference** | vLLM-audio, Triton ASR | Inference engines extended for audio |
| **Long context (RoPE, ALiBi)** | Streaming + chunked attention | Different solution (causality) |
| **Multilingual** | Multilingual (Whisper, MMS) | More acute due to acoustic variation |
| **Out-of-vocabulary** | Codebook collapse / unused codes | Different failure mode |
| **Prompt engineering** | Prompt engineering for Speech LLM | Same idea, Moshi accepts prompts |
| **In-context learning** | Voice cloning from 3-sec sample | VALL-E demonstrates this |
| **RAG** | Retrieval-augmented ASR | Less mature, emerging area |
| **Chain-of-thought** | Speech CoT (chain-of-speech) | Experimental |
| **Function calling** | Tool use in Speech agents | GPT-4o, Moshi support |

### 3.2 Đào sâu các mapping quan trọng

Một số mapping ở 3.1 là nông cạn (literally same), nhưng một số mapping cần đào sâu vì có nuance riêng. Hãy đi qua những cái quan trọng nhất.

#### 3.2.1 Token ≈ Audio frame: cùng concept, khác đơn vị

Ở NLP, token là đơn vị cơ bản của input. Vocabulary cố định, mỗi token có ID riêng. Sequence length đo bằng số tokens.

Ở Speech, đơn vị tương đương phụ thuộc representation:

- Mel spectrogram: "frame" là 1 cột của ma trận, đại diện 10-25 ms audio.
- Wav2Vec 2.0: "frame" là 1 latent vector ở 50 fps = 20 ms/frame.
- EnCodec 75 fps: "frame" = 1 codebook tuple ở 75 fps = 13.3 ms/frame.

Một câu nói "I love speech" có khoảng:

- 3 từ → 3-4 BPE tokens.
- 12 phonemes → 12 phoneme tokens (nếu dùng phoneme system).
- 1.5 giây audio → 150 mel frames, 75 Wav2Vec frames, 112 EnCodec frames, 19 Mimi frames.

Tỉ lệ frame/token là quan trọng: nếu bạn đang dùng codec ở 75 fps để feed vào LLM, cùng một câu sẽ chiếm gấp ~25 lần "tokens" so với BPE. Đây là lý do tại sao Mimi (12.5 fps) là đột phá: gần như tương đương rate text token.

#### 3.2.2 BERT pretraining ≈ Wav2Vec 2.0: chi tiết khác biệt

Cả hai đều là self-supervised, masked prediction. Nhưng:

- BERT mask **discrete tokens** (15% of input tokens replaced với `[MASK]`). Loss là cross-entropy over vocabulary (~30K classes).
- Wav2Vec 2.0 mask **continuous latent vectors** (6.5% of latent steps). Vector ground truth là 768-dim float, không thể cross-entropy. Thay vào đó dùng **contrastive loss (InfoNCE)**: chọn đúng vector từ tập distractors.

Tại sao khác biệt? Vì continuous space không có "label" tự nhiên, không như text có vocabulary cố định. Contrastive learning là cách thông minh để giả lập classification loss trong continuous space.

HuBERT (2021) đề xuất một workaround: trước pretraining, dùng k-means để clustering Wav2Vec features thành ~100 clusters, dùng cluster ID làm "label". Bây giờ pretext task là cross-entropy classification giống BERT, đơn giản hơn và performance cao hơn. Đây là HuBERT.

#### 3.2.3 Causal mask ≈ Streaming mask: nuance khác nhau

NLP causal mask: token tại vị trí $t$ chỉ nhìn được tokens $1, \ldots, t-1$. Đây là requirement của autoregressive LM.

Speech streaming mask: phức tạp hơn vì có hai lý do gắn streaming:

1. **Real-time constraint**: ASR streaming phải output partial transcript khi user còn đang nói. Frame $t$ chỉ có thể nhìn frames $1, \ldots, t$ (giống causal NLP).
2. **Latency constraint**: thực tế, có thể chấp nhận "lookahead" nhỏ (say 100-200 ms) để cải thiện accuracy. Frame $t$ nhìn frames $1, \ldots, t + L$ với $L$ là lookahead frames.

Còn có **chunked attention** (used in streaming Conformer): chia audio thành chunks, mỗi chunk attention bidirectional bên trong nhưng causal giữa chunks. Đây là compromise giữa accuracy và latency.

Trong NLP, ít khi gặp các pattern này vì text input thường đã có sẵn. Chỉ có streaming text generation (chatbot) mới có concern tương tự, và họ giải bằng cách generate token-by-token đơn giản.

#### 3.2.4 KV cache: thật sự giống nhau

KV cache cho speech LM (VALL-E, Moshi) hoạt động chính xác như cho NLP LM. Mỗi step generation, lưu lại K, V của tất cả tokens đã sinh, reuse khi sinh token tiếp theo. Đây là một trong những "free wins" khi bạn quen với NLP và làm Speech LLM: code KV cache không cần viết lại.

Một chi tiết kỹ thuật đáng lưu ý: speech LM thường có nhiều codebook layers (Moshi có 8). KV cache cho mỗi codebook nên tách biệt hay chia sẻ? Moshi chọn phương án dùng "depth transformer", một transformer nhỏ xử lý codebook layers độc lập với KV cache riêng. Kỹ thuật này sẽ được trình bày kỹ ở Chương 11.

#### 3.2.5 In-context learning ↔ Voice cloning

NLP in-context learning: cho model 3-5 examples trong prompt, nó generalize được task mới mà không cần fine-tune.

Speech equivalent: cho model 3 giây audio sample của một speaker, nó có thể clone giọng đó để đọc text bất kỳ. Đây chính là VALL-E (2023) và F5-TTS (2024).

Cơ chế: VALL-E là một LM trên codec tokens. Prompt = audio prefix (codec tokens của 3-sec sample). Model tiếp tục sinh codec tokens "theo phong cách" của prefix. Decode về waveform → voice clone.

Đây là một ví dụ đẹp cho thấy **paradigm Transformer chung là đủ mạnh để khái niệm hoá hoàn toàn**: in-context learning trong text và voice cloning là cùng một cơ chế chỉ khác token type.

## Phần 4 — Scale Difference: Tại sao Speech khó hơn

Phần 1-3 đã cho ta thấy cấu trúc tương đồng giữa NLP và Speech. Nhưng có một yếu tố khiến Speech khó hơn về mặt kỹ thuật: **scale của dữ liệu**.

### 4.1 Data rate: cách nhau hàng nghìn lần

So sánh raw data rate:

$$
\text{Data rate}_{\text{text}} \approx 4 \text{ tokens/sec speech} \quad \text{vs} \quad \text{Data rate}_{\text{audio}} \approx 16{,}000 \text{ samples/sec}
$$

Đó là tỉ lệ 4000:1. Cụ thể với 10 giây speech:

| Metric | Text | Audio (16kHz, 16-bit) |
|---|---|---|
| Raw data | ~20 bytes | 32,000 bytes (320,000 byte cho 10s) |
| Unit count | ~40 tokens | 160,000 samples (raw) |
| Sequence length sau "tokenization" | 40 | 1000 mel frames, 500 Wav2Vec, 750 EnCodec, 125 Mimi |

Ngay sau khi tokenize, audio vẫn dài hơn text ~3-25 lần (tuỳ representation chọn). Đây là consequence rõ ràng nhất của information density thấp.

### 4.2 Hệ luỵ kiến trúc

Sequence dài hơn → self-attention $O(L^2)$ tốn kém hơn → cần tối ưu kiến trúc.

Các giải pháp chính:

**4.2.1 Strong downsampling ở encoder**

Whisper dùng 2 convolutional layers stride-2 ở đầu encoder → giảm sequence length 4x trước khi vào Transformer. Conformer dùng subsampling 4x tương tự.

Wav2Vec 2.0 thì cực đoan hơn: CNN feature extractor giảm 320x (16K samples/sec → 50 fps).

Tại sao có thể downsample mạnh vậy mà không mất accuracy? Vì information density của audio thấp: nhiều samples liền kề mang gần như cùng thông tin, có thể nén lại không mất mát đáng kể.

**4.2.2 Local attention / chunked attention**

Conformer thay self-attention global bằng convolution local + self-attention. Local convolution xử lý dependencies ngắn ($\sim$ 10 frames = 100ms), self-attention xử lý long-range.

Streaming Conformer dùng chunked attention: chia thành chunks 16-32 frames, attention bidirectional bên trong chunk, causal giữa chunks.

**4.2.3 Causal CNN trong codec**

EnCodec, Mimi đều dùng causal CNN trong encoder và decoder. Causal CNN giữ được streaming property: encoder có thể process audio chunk-by-chunk, decoder có thể generate audio chunk-by-chunk.

Đây là kỹ thuật mà NLP ít gặp vì text thường không streaming. Speech AI có nhiều "kỹ thuật xử lý tín hiệu" mà NLP không có.

### 4.3 GPU memory implications

Cùng một transformer 1B params, nhưng Speech LM cần xử lý sequence dài hơn:

- LLM text với 4K context: KV cache $\sim$ 4K × 2 × num_layers × hidden_dim × 2 bytes = $\sim$ 2 GB cho LLaMA-7B.
- Speech LM Moshi với 30 giây audio: sequence length $\sim$ 375 frames × 8 codebooks = 3000 effective tokens, gần bằng LLM.

Trên thực tế, Moshi 7B param có inference memory tương đương với LLaMA-7B-Instruct. Nhưng nếu dùng EnCodec ở 75 fps thay vì Mimi 12.5 fps, sequence length sẽ 6x dài hơn, memory bùng nổ.

Đây là một lý do quan trọng tại sao Mimi (12.5 fps) là một đột phá engineering: nó cho phép Speech LLM có memory tương đương LLM thay vì 6-10 lần hơn.

### 4.4 Training compute

Whisper Large-v3 trained trên 680,000 giờ audio (~78 năm dữ liệu). Mỗi giờ = 100 fps × 3600 frames = 360,000 mel frames. Tổng dataset: ~244 tỷ mel frames.

So sánh: LLaMA-3 8B trained trên ~15 trillion tokens.

Speech dataset nhỏ hơn nhưng mỗi sample "đắt" hơn để xử lý (Mel preprocessing, longer sequences). Tổng compute training Whisper-Large-v3 và LLaMA-3 8B tương đương nhau, khoảng 1-2 triệu H100 GPU-hours.

> **⚠️ Latency warning quan trọng**
>
> Audio sequence dài hơn text ~25× ở mel frame level. Self-attention $O(L^2)$ trở thành bottleneck nghiêm trọng. Đây là lý do Conformer kết hợp convolution (local) với attention (global), và tại sao streaming ASR cần kiến trúc đặc biệt.
>
> Nếu bạn đang debug performance của Speech model và thấy chậm hơn LM cùng kích thước, đó là consequence trực tiếp của data rate gap. Các optimization như flash attention, KV cache compression, codec ở fps thấp đều quan trọng hơn ở Speech so với NLP.

## Phần 5 — Architecture Taxonomy

Để giúp bạn map các kiến trúc cụ thể vào bức tranh lớn, đây là taxonomy của Speech AI architectures, sắp xếp theo thời kỳ phát triển.

### 5.1 Era 1: Hybrid (HMM + DNN) — lịch sử

Trước 2015, ASR chủ yếu dùng kiến trúc "hybrid":

1. **GMM-HMM** modeling phonemes và alignment.
2. **DNN** (3-7 layers feedforward) classify frames into phonemes/senones.
3. **Decoder** (WFST-based) tìm transcription tối ưu.

Đây là pipeline phức tạp, nhiều bộ phận riêng biệt (acoustic model, language model, lexicon, decoder). Cần nhiều domain expertise.

Ngày nay, kiến trúc này hầu như đã bị thay thế bởi end-to-end. Chỉ còn dùng trong một số domain niche (legacy system, very low-resource languages).

> **NLP analogy**: tương tự kiến trúc "feature engineering + linear model + heuristic" thời kỳ trước Transformer. Hybrid speech = feature-engineered speech.

### 5.2 Era 2: End-to-End (CTC, Attention, RNN-T) — 2015-2020

End-to-end approach: một neural network duy nhất biến audio → text, train end-to-end với một loss function.

Ba paradigm chính:

**5.2.1 CTC (Connectionist Temporal Classification)**

Output mỗi frame một character (có thể là blank). Loss marginalize over alignments. Sẽ giải thích kỹ ở Chương 4.

- Pros: simple, parallelizable, streaming-friendly.
- Cons: assume conditional independence between outputs (no LM bias built-in).

**5.2.2 Attention-based encoder-decoder (LAS, Listen-Attend-Spell)**

Encoder process audio, decoder attention vào encoder, autoregressive generate transcript. Tương đương seq2seq cho ASR.

- Pros: strong LM bias (decoder is itself a LM).
- Cons: not streaming-friendly (full attention global), exposure bias.

**5.2.3 RNN-T (RNN-Transducer)**

Combine CTC's per-frame output với attention's autoregressive nature. Three components: encoder (audio), predictor (history of outputs), joiner (combine). Streaming-friendly + LM bias.

- Pros: best of both worlds, streaming, strong accuracy.
- Cons: complex training, slower convergence.

Sẽ giải thích kỹ cả ba ở Chương 4.

> **NLP analogy**: 
> - CTC ≈ token-level classification (NER-style).
> - Attention encoder-decoder ≈ T5/BART.
> - RNN-T ≈ kiến trúc đặc thù speech (không có analogy hoàn hảo trong NLP).

### 5.3 Era 3: Self-supervised pretrain + fine-tune (2020-2023)

Wav2Vec 2.0 (2020), HuBERT (2021), WavLM (2022): pretrain backbone trên unlabeled audio, fine-tune cho downstream task.

Pipeline:

1. Pretrain Wav2Vec 2.0 trên 60K giờ unlabeled audio.
2. Fine-tune với CTC loss trên 100 giờ labeled (LibriSpeech).
3. Đạt WER rất cạnh tranh trên LibriSpeech trong thiết lập được công bố, dù dùng ít labeled data hơn nhiều so với các hệ thống supervised quy mô lớn.

Đây là minh chứng rằng **self-supervised + ít labeled data > nhiều labeled data**, ít nhất với English.

Whisper (2022) là counterexample: train hoàn toàn supervised trên 680K giờ. Performance rất mạnh, đặc biệt zero-shot multilingual. Cho thấy "data scaling" cũng có giá trị riêng.

Ngày nay (2026), production system thường lai cả hai: pretrained backbone + fine-tune trên large labeled corpus.

> **NLP analogy**: era 3 song song với era BERT + fine-tune (2018-2020). Cùng paradigm shift.

### 5.4 Era 4: Speech LLMs (2023+)

Năm 2023-2024 là thời điểm Speech AI gặp LLM. Ba paradigm chính:

**5.4.1 Codec-based Speech LLM (AudioLM, VALL-E, Moshi)**

Codec tokens (EnCodec, Mimi) làm "tokenizer", một Transformer LM (GPT-style) làm engine. Audio → codec tokens → LM → codec tokens → audio.

- VALL-E (2023): TTS với 3-sec voice cloning.
- AudioLM (2023): audio continuation (cho 3 giây speech, sinh 30 giây tiếp).
- Moshi (2024): full-duplex dialogue (nghe + nói đồng thời).

**5.4.2 Embedding-based Multimodal LLM (Qwen2-Audio, Qwen2.5-Omni, Gemini)**

Audio encoder (Wav2Vec, Whisper encoder, hoặc dedicated) → continuous embeddings → adapter (linear) → LLM (frozen hoặc fine-tune). Output là text.

- Qwen2-Audio (2024): understanding audio (caption, QA, classification).
- Qwen2.5-Omni (2025): unified text + audio + image + video in single transformer.
- Gemini 2.0 (2025): native multimodal, streaming.

**5.4.3 Hybrid (Whisper + LLM)**

Whisper transcribe → text → LLM xử lý → text output → TTS. Pipeline classical, nhưng vẫn dùng rộng rãi cho production (giảm cost, dễ debug, mỗi component riêng biệt).

- ElevenLabs voice agents.
- LiveKit + GPT + Cartesia.
- Vapi.ai.

> **NLP analogy**: era 4 song song với era LLM. Speech LLM = "LLM has gone multimodal".

### 5.5 Cây taxonomy tổng hợp

```
Speech AI Models
│
├── Era 1: Hybrid (HMM-DNN)        [legacy, 2010-2015]
│   ├── Kaldi-style
│   └── DNN-HMM hybrid
│
├── Era 2: End-to-End                [2015-2020]
│   ├── CTC (DeepSpeech 2, Jasper)
│   ├── Attention (LAS)
│   └── RNN-T (Google Streaming ASR)
│
├── Era 3: Self-supervised           [2020-2023]
│   ├── Wav2Vec 2.0
│   ├── HuBERT
│   ├── WavLM
│   └── XLS-R (multilingual)
│
└── Era 4: Speech LLMs               [2023+]
    ├── Codec-based
    │   ├── AudioLM
    │   ├── VALL-E
    │   ├── Moshi
    │   └── Qwen2.5-Omni
    ├── Embedding-based
    │   ├── Qwen2-Audio
    │   ├── Gemini 2.0
    │   └── GPT-4o (partial speculation)
    └── Hybrid pipeline
        ├── Whisper + LLM + TTS
        └── ASR + Function calling agents
```

Cuốn sách này sẽ đi qua tất cả các era, nhưng nhấn mạnh Era 4 (Speech LLMs) vì đó là frontier hiện tại và cũng là điểm mà audience NLP của bạn quan tâm nhất.

## Phần 6 — Code: So sánh ba biểu diễn cụ thể

Phần này là một code walkthrough chi tiết để bạn cảm nhận sự khác biệt giữa ba con đường biểu diễn audio đã thảo luận ở Phần 2. Bạn nên đọc kỹ comments và chạy thử trên Colab hoặc Jupyter để hiểu rõ.

### 6.1 Bài toán đơn giản

Cho một waveform 1 giây ở 16 kHz (chuẩn speech), so sánh kích thước và rate của:

1. Raw waveform.
2. Mel spectrogram.
3. Wav2Vec 2.0 features (giả lập).
4. EnCodec tokens.
5. Mimi tokens (giả lập).

### 6.2 Implementation

```python
import torch
from torch import Tensor


def compare_representations(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
) -> dict[str, dict]:
    """So sánh kích thước năm biểu diễn audio cho 1 đoạn duration_sec giây.

    Args:
        duration_sec: Độ dài audio (giây)
        sample_rate: Tần số lấy mẫu (Hz)

    Returns:
        Dictionary chứa shape, total values, bytes, fps của từng biểu diễn
    """
    n_samples: int = int(duration_sec * sample_rate)

    # 1. Raw waveform (int16): chuẩn để lưu trữ
    raw_shape = (n_samples,)
    raw_size_bytes = n_samples * 2  # int16 = 2 bytes
    raw_fps = sample_rate

    # 2. Mel spectrogram (float32)
    n_mels: int = 80
    hop_length: int = 160  # 10 ms frames
    n_frames: int = n_samples // hop_length
    mel_shape = (n_mels, n_frames)
    mel_size_bytes = n_mels * n_frames * 4
    mel_fps = sample_rate // hop_length

    # 3. Wav2Vec 2.0 features (float32, giả lập)
    ssl_fps: int = 50  # 50 frames/sec (320x downsampling)
    ssl_dim: int = 768  # hidden dimension
    ssl_frames: int = int(duration_sec * ssl_fps)
    ssl_shape = (ssl_frames, ssl_dim)
    ssl_size_bytes = ssl_frames * ssl_dim * 4

    # 4. EnCodec 75 fps × 8 codebooks (int32)
    encodec_fps: int = 75
    n_codebooks: int = 8
    encodec_frames: int = int(duration_sec * encodec_fps)
    encodec_shape = (n_codebooks, encodec_frames)
    encodec_size_bytes = n_codebooks * encodec_frames * 4  # int32

    # 5. Mimi 12.5 fps × 8 codebooks (int32)
    mimi_fps: int = 12  # round down for cleanness
    mimi_frames: int = int(duration_sec * mimi_fps)
    mimi_shape = (n_codebooks, mimi_frames)
    mimi_size_bytes = n_codebooks * mimi_frames * 4

    return {
        "raw_waveform": {
            "shape": raw_shape,
            "total_values": n_samples,
            "size_bytes": raw_size_bytes,
            "fps": raw_fps,
            "lm_friendly": False,
        },
        "mel_spectrogram": {
            "shape": mel_shape,
            "total_values": n_mels * n_frames,
            "size_bytes": mel_size_bytes,
            "fps": mel_fps,
            "lm_friendly": False,
        },
        "wav2vec2_features": {
            "shape": ssl_shape,
            "total_values": ssl_frames * ssl_dim,
            "size_bytes": ssl_size_bytes,
            "fps": ssl_fps,
            "lm_friendly": False,
        },
        "encodec_tokens": {
            "shape": encodec_shape,
            "total_values": n_codebooks * encodec_frames,
            "size_bytes": encodec_size_bytes,
            "fps": encodec_fps,
            "lm_friendly": True,
        },
        "mimi_tokens": {
            "shape": mimi_shape,
            "total_values": n_codebooks * mimi_frames,
            "size_bytes": mimi_size_bytes,
            "fps": mimi_fps,
            "lm_friendly": True,
        },
    }


# So sánh cho 1 giây audio
result = compare_representations(duration_sec=1.0)

print(f"{'Representation':<20} {'Shape':<20} {'Values':>10} {'Bytes':>10} {'FPS':>6} {'LM?':>5}")
print("-" * 75)
for name, info in result.items():
    shape_str = str(info['shape'])
    print(f"{name:<20} {shape_str:<20} "
          f"{info['total_values']:>10,} "
          f"{info['size_bytes']:>10,} "
          f"{info['fps']:>6} "
          f"{str(info['lm_friendly']):>5}")
```

Output mong đợi (cho 1 giây audio ở 16 kHz):

```
Representation       Shape                    Values      Bytes    FPS   LM?
---------------------------------------------------------------------------
raw_waveform         (16000,)                 16,000     32,000  16000  False
mel_spectrogram      (80, 100)                 8,000     32,000    100  False
wav2vec2_features    (50, 768)                38,400    153,600     50  False
encodec_tokens       (8, 75)                     600      2,400     75   True
mimi_tokens          (8, 12)                      96        384     12   True
```

### 6.3 Phân tích output

Quan sát quan trọng:

1. **Mel spectrogram nhỏ hơn raw waveform về số values** (8000 vs 16000) nhưng cùng size về bytes (do float32 vs int16, factor 2x).

2. **Wav2Vec 2.0 features có MORE total values và bytes hơn cả raw waveform**. Trông như "tốn hơn" nhưng đây là features có ngữ nghĩa cao hơn nhiều, dùng cho downstream tasks.

3. **EnCodec compress mạnh nhất về bytes**: chỉ 2.4 KB cho 1 giây so với 32 KB raw. Compression 13x.

4. **Mimi compress còn mạnh hơn**: 384 bytes cho 1 giây. Compression 83x. Đây là chìa khoá cho speech LLM hiệu quả.

5. **LM-friendly**: chỉ codec tokens là discrete và dùng được LM trực tiếp. Mel và Wav2Vec features cần thêm bước quantization (mất chất lượng).

### 6.4 Mở rộng: ánh xạ sang LLM-token rate

Nếu coi mỗi "tuple" của codec (8 codebooks tại một time step) là một "token", ta có:

- EnCodec ở 75 fps: 75 tokens/sec → 10 giây = 750 tokens. So với text 10s speech ≈ 30-40 tokens, gấp ~20x.
- Mimi ở 12.5 fps: 12.5 tokens/sec → 10 giây = 125 tokens. Gấp ~3-4x text rate. Đây gần như tỉ lệ "lý tưởng" để LLM xử lý audio hiệu quả.

Bạn có thể bắt đầu thấy tại sao kỹ thuật engineering của Mimi (rate thấp + quality cao) là một trong các đột phá quan trọng nhất của 2024.

## Phần 7 — Limitations & Open Problems

Mọi tổng quan đều cần một phần thẳng thắn về giới hạn của công nghệ hiện tại. Phần này phân tích các điểm yếu quan trọng của Speech AI ở thời điểm 2026.

### 7.1 Codec tokens chưa hoàn toàn lossless

EnCodec ở 6 kbps và Mimi ở 1.1 kbps đều có chút artifact perceptual. Tai người nghe kỹ vẫn nhận ra "tổng hợp" thay vì "tự nhiên". Đây là vấn đề khó vì lossless audio compression (FLAC) cần bitrate cao hơn nhiều (~700 kbps cho 16 kHz speech).

Hệ luỵ: Speech LLM output qua codec decoder không bao giờ đạt fidelity của recording gốc. Đối với một số ứng dụng (audiobook, dialogue chatbot) đây không phải vấn đề lớn. Đối với ứng dụng yêu cầu cao (music, professional voice acting) cần kỹ thuật khác (diffusion-based vocoder, neural vocoder cao chất lượng như BigVGAN).

### 7.2 Streaming vs offline tradeoff

Offline ASR (Whisper) có thể nhìn toàn bộ audio trước khi transcribe → accuracy cao, latency tệ.

Streaming ASR (RNN-T với chunked attention) chỉ nhìn được 100-300 ms tương lai → latency thấp, accuracy thấp hơn 5-10% WER.

Trade-off này chưa có giải pháp lý tưởng. Mọi production system phải chọn một mode. Đây là một active research area (Mamba state-space model có thể là answer, đang được explore).

### 7.3 Vietnamese-specific challenges

Tiếng Việt có một số đặc thù khiến Speech AI khó hơn English:

**7.3.1 Thanh điệu (tonal)**

Tiếng Việt có 6 thanh điệu (ngang, huyền, sắc, hỏi, ngã, nặng). Cùng phụ âm và nguyên âm, khác thanh → khác nghĩa hoàn toàn ("ma", "má", "mà", "mả", "mã", "mạ").

Model trained chủ yếu trên English (Whisper, Wav2Vec 2.0) thường không capture tốt thanh điệu. PhoWhisper fine-tune Whisper cho Vietnamese, kết quả cải thiện ~30% WER trên VLSP test set.

**7.3.2 Low resource**

Compared with English (Whisper trained on 680K giờ tiếng Anh), Vietnamese chỉ có khoảng 5-10K giờ public corpus (CommonVoice, VLSP, VivosCorpus). Ít hơn 50x.

Hệ luỵ: cần techniques như semi-supervised pseudo-labeling, multilingual transfer (XLS-R), hoặc data augmentation tích cực. Chương 16-17 sẽ đào sâu.

**7.3.3 Code-switching**

Người Việt thường code-switch giữa Việt và Anh ("Tôi sẽ build cái dashboard này"). Model train chỉ trên Vietnamese clean text thường thất bại với code-switched speech. Đây là open problem chưa có giải pháp tốt.

### 7.4 Multimodal alignment

Speech LLMs (Qwen2-Audio, GPT-4o) học audio + text trong cùng model, nhưng alignment giữa hai modality vẫn imperfect. Ví dụ:

- Model có thể hallucinate transcription (như LLM hallucinate facts).
- Voice cloning có thể "tốt một câu, lỗi câu tiếp theo".
- Long-form audio (>5 phút) thường mất coherence.

Đây không phải vấn đề kỹ thuật cụ thể có thể giải bằng kiến trúc mới, mà là consequence của training dynamics chưa hoàn hảo. Nhiều research đang focus vào RLHF cho speech, DPO cho TTS, etc.

### 7.5 Compute / cost

Một số hệ thống streaming TTS frontier năm 2026 công bố latency khoảng vài trăm mili-giây, nhưng chi phí theo ký tự hoặc theo phút vẫn có thể lớn khi nhân lên quy mô call center. Với 1 triệu cuộc hội thoại, khác biệt nhỏ ở cost per turn có thể chuyển thành hàng nghìn USD mỗi ngày.

So sánh với text LLM (khoảng 0.0001 USD per token, generate vài trăm token per response), audio interaction đắt hơn khoảng 10-100 lần. Đây là rào cản kinh tế cho widespread deployment.

Mimi 1.1 kbps cộng Moshi giúp giảm đáng kể (toàn bộ codec inference cho 1 giây chi phí khoảng 0.00001 USD). Nhưng vẫn cần thêm improvements để cạnh tranh với text-only chatbots về cost.

## Phần 8 — Tóm tắt & Hướng đi tiếp theo

### 8.1 Những bài học chính

Chương 1 đã thiết lập 5 ý tưởng cốt lõi mà bạn cần mang theo khi đọc tiếp:

1. **Speech AI = NLP + tokenizer cho audio**. Mọi kiến thức Transformer của bạn áp dụng được. Khác biệt nằm ở bước biến đổi continuous signal → discrete-semantic units.

2. **Có ba con đường biểu diễn audio**, mỗi cái với trade-off riêng:
   - Mel spectrogram (handcrafted, deterministic, không LM-friendly).
   - Wav2Vec 2.0 features (learned, continuous, không LM-friendly).
   - Codec tokens (learned, discrete, LM-FRIENDLY).

3. **Codec tokens (EnCodec, Mimi) là cầu nối quan trọng nhất** đến speech LLMs. Đây là analogy đẹp nhất với BPE: cùng concept "biến input thô thành integer tokens", chỉ khác tokenizer được trained.

4. **Scale của audio cao hơn text nhiều bậc** (~4000× raw samples/sec). Hệ luỵ kiến trúc: cần downsampling mạnh, cần local/chunked attention, cần kỹ thuật streaming.

5. **Speech AI đang ở Era 4 (Speech LLMs)** với các paradigms: codec-based (Moshi, VALL-E), embedding-based (Qwen2-Audio), hybrid pipeline (Whisper + LLM + TTS). Bạn cần biết mỗi paradigm khi đọc paper hoặc design system.

### 8.2 NLP↔Speech Concept Bridge — final words

Quay lại câu hỏi mở đầu: tại sao một data scientist NLP cần học Speech AI?

Bây giờ bạn có thể trả lời cụ thể hơn:

- Vì cùng paradigm Transformer, kiến thức không bị vứt bỏ.
- Vì các kỹ thuật của bạn (fine-tuning, RLHF, LoRA, quantization) áp dụng trực tiếp.
- Vì Speech LLM là frontier hiện tại, và career-wise đây là skill cao giá trị.
- Vì các sản phẩm voice-first (Moshi, GPT-4o) đang là wave mới của UX.

Mọi chương tiếp theo của cuốn sách sẽ đào sâu chi tiết cụ thể, nhưng bạn đã có "map" để định hướng. Bookmark Bảng 3.1 — đó là tham chiếu chính khi bạn đọc các chương sau và gặp thuật ngữ lạ.

### 8.3 Lộ trình đọc tiếp

Tuỳ background và mục tiêu của bạn:

**Nếu bạn từ LLM/GPT research (audio LM, multimodal)**:

- Bỏ qua Chương 2-3 (foundation signal processing). Đọc nhanh để hiểu vocabulary.
- Đào sâu Chương 11-13 (Speech LLMs, Multimodal, Full-Duplex).
- Quay lại Chương 8-10 (TTS) nếu cần generate audio.

**Nếu bạn từ NLP/BERT research (transfer learning, fine-tuning)**:

- Đọc Chương 2-3 để có foundation.
- Đào sâu Chương 4-5 (ASR foundations + modern ASR).
- Tham khảo Chương 6 (Whisper case study).

**Nếu bạn từ CV/Vision (cần multimodal)**:

- Đọc Chương 2 kỹ (audio fundamentals tương tự image processing).
- Skim Chương 3 (representations).
- Đào sâu Chương 12 (Multimodal Omni — Qwen2.5-Omni, Gemini).

**Nếu bạn cần triển khai production ngay**:

- Đọc Chương 6 (Whisper) + Chương 7 (Streaming ASR).
- Đào sâu Chương 18-20 (Training frameworks, Inference engines, Production systems).
- Bookmark Chương 16-17 nếu deploy cho thị trường Việt Nam.

**Đọc toàn bộ cuốn sách (comprehensive)**:

- Đi tuần tự Chương 0 đến Chương 21.
- Mỗi chương dành 2-3 giờ đọc kỹ và chạy code.
- Tổng thời gian: khoảng 60-80 giờ cho exposure đầy đủ.

### 8.4 Mở đầu cho Chương 2

Chương 1 đã build the "big picture map". Chương 2 sẽ là chương đầu tiên đi sâu vào kỹ thuật cụ thể: **xử lý tín hiệu âm thanh fundamentals** (DFT, STFT, Mel filterbank, MFCC).

Đây là chương nặng về toán và DSP nhất của cuốn sách. Nếu bạn đến từ NLP, có thể bạn chưa quen với spectral analysis. Chương 2 sẽ giải thích từ ground up, kèm analogy NLP và CV mỗi khi có thể.

Mục tiêu Chương 2: sau khi đọc xong, bạn có thể nhìn vào một mel spectrogram và "đọc" được nó (cũng như cách bạn đọc word embedding cosine similarity visualization). Đây là một skill cơ bản cho mọi speech engineer.

---

## Tài liệu tham khảo

1. Défossez, A. et al. (2024). "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue". arXiv:2410.00037.
2. Chu, Y. et al. (2023). "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models". arXiv:2311.07919.
3. Xu, J. et al. (2025). "Qwen2.5-Omni Technical Report". arXiv:2503.20215.
4. Baevski, A. et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations". NeurIPS 2020.
5. Hsu, W. et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units". IEEE/ACM TASLP.
6. Wang, C. et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers". arXiv:2301.02111 (VALL-E).
7. Borsos, Z. et al. (2023). "AudioLM: A Language Modeling Approach to Audio Generation". IEEE/ACM TASLP.
8. Défossez, A. et al. (2022). "High Fidelity Neural Audio Compression". arXiv:2210.13438 (EnCodec).
9. Gulati, A. et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition". Interspeech 2020.
10. Radford, A. et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision". OpenAI Whisper.
11. Nguyen, T. et al. (2024). "PhoWhisper: Fine-tuning Whisper for Vietnamese Automatic Speech Recognition". arXiv preprint.
