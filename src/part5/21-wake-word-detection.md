# Chương 21: Wake-Word Detection (Keyword Spotting)

## Mở đầu: "Hey Siri" — bài toán nghe tưởng đơn giản

Có một bài toán Speech AI hiện diện trong cuộc sống hàng ngày của hàng tỉ người mà ít ai chú ý: mỗi lần bạn nói "Hey Siri", "OK Google", "Alexa", hay "Hey Cortana", có một mô hình neural network đã liên tục lắng nghe microphone của bạn trong nhiều tháng, chờ đợi đúng cụm từ đó.

Bài toán này nghe có vẻ đơn giản: detect 1 cụm từ cụ thể trong audio stream. Nhưng implementation thực tế đòi hỏi vượt qua các trade-off cực đoan:

- **Always-on**: chạy 24/7, không tắt khi pin yếu.
- **Ultra-low power**: tiêu thụ <10 mW để không drain pin trong 1 ngày.
- **Sub-100ms latency**: detect ngay khi user kết thúc nói wake-word.
- **Low false-accept rate (FAR)**: không thức dậy khi user không gọi (1 lần/24h là tệ, 1 lần/tuần là acceptable).
- **Low false-reject rate (FRR)**: thức dậy mỗi khi user thực sự gọi (>95% là chấp nhận được).
- **Privacy-preserving**: không upload audio liên tục lên cloud.

Cân bằng tất cả constraints này khiến Wake-Word Detection trở thành một trong những bài toán engineering thú vị nhất trong Speech AI, và là nền tảng cho mọi voice assistant.

Chương này deep dive về Wake-Word Detection (còn gọi là Keyword Spotting hoặc KWS): bài toán, mathematical formulation, classical vs modern approaches, industry implementations, và đặc biệt là cách deploy trên edge devices với constraints khắc nghiệt.

> **📝 Cấu trúc chương**
>
> Phần 1-2: bài toán và mathematical formulation.
> Phần 3: production constraints (power, latency, FAR/FRR).
> Phần 4: classical approaches (HMM/GMM, DTW) — historical context.
> Phần 5: modern DL approaches (small CNN, Conformer-KWS, streaming RNN-T).
> Phần 6: industry implementations deep dive (Apple, Google, Amazon, Microsoft).
> Phần 7: training data và augmentation strategies.
> Phần 8: deployment trên edge (TFLite Micro, ONNX Runtime Mobile).
> Phần 9: personalization (user-specific wake-words).
> Phần 10: Vietnamese wake-words.
> Phần 11: tóm tắt.

---

## Phần 1 — Bài toán và terminology

### 1.1 Định nghĩa

**Wake-word detection** (WWD) là bài toán detect một cụm từ cụ thể (wake-word, hot-word, trigger phrase) trong continuous audio stream. Khi wake-word được detect, system "wake up" và bắt đầu xử lý audio sau wake-word như command/query.

Cụm từ tương đương: **Keyword Spotting (KWS)** — detect một hoặc nhiều keywords trong audio. Wake-word detection là special case của KWS với vocabulary chỉ 1 keyword.

### 1.2 Các terminology liên quan

| Term | Definition |
|---|---|
| **Wake-word** | Cụm từ duy nhất kích hoạt voice assistant ("Hey Siri") |
| **Hot-word / Hotword** | Đồng nghĩa với wake-word, dùng nhiều bởi Google docs |
| **Trigger phrase** | Đồng nghĩa, dùng bởi Microsoft Cortana docs |
| **Wake-up word** | Đồng nghĩa, dùng cho automotive context |
| **Keyword spotting (KWS)** | General: detect một tập keywords (10-100) trong audio |
| **Open-vocabulary KWS** | Detect bất kỳ keyword nào user define, không pre-train |
| **Speech command recognition** | Variation: classify single utterance into N commands |
| **Voice activity detection (VAD)** | Different problem: detect speech vs silence (không phân biệt nội dung) |

### 1.3 Khác biệt với ASR

Wake-word detection KHÔNG phải ASR:

- ASR: transcribe arbitrary speech → text. Vocabulary mở.
- WWD: binary decision "wake-word present" vs "not present". Vocabulary cố định 1 phrase.

Tại sao không dùng ASR cho wake-word detection? Lý do là **power budget**:

- ASR (Whisper-tiny): ~50 MB model, ~50-100 mW inference, ~10 USD GPU per server-hour.
- WWD (typical): ~50 KB model, <10 mW inference, gần như 0 USD (runs on device DSP).

Nói cách khác, WWD là một specialized binary classifier optimized cực mạnh cho 1 phrase, trong khi ASR là general transcription.

---

## Phần 2 — Mathematical Formulation

Formal definition cho phép ta phân tích trade-offs và design models.

### 2.1 Binary classification per audio window

Given continuous audio stream $x(t)$, WWD output a sequence of binary decisions $y_n \in \{0, 1\}$ tại các time windows $t_n$:

$$
y_n = \mathbb{1}\left[ \text{score}(x[t_n - L : t_n]) > \tau \right]
$$

trong đó:

- $L$ là window length (typical 500-1500 ms, dài đủ chứa 1 wake-word phrase).
- $\text{score}(\cdot)$ là model output (posterior probability hoặc logit).
- $\tau$ là decision threshold.
- $\mathbb{1}[\cdot]$ là indicator function.

Window stride $s$ (typical 10-100 ms) quyết định decision frequency. Tăng $s$ giảm compute nhưng tăng latency.

### 2.2 Posterior smoothing

Single-window decision quá noisy. Smooth across multiple consecutive windows:

$$
\bar{p}_n = \frac{1}{W} \sum_{i=0}^{W-1} \text{score}(x[t_{n-i} - L : t_{n-i}])
$$

Trigger khi $\bar{p}_n > \tau_{\text{smooth}}$. Đây giảm false positives đáng kể.

### 2.3 Decision threshold tuning

Threshold $\tau$ là knob quan trọng nhất:

- $\tau$ cao → ít false positives (FP), nhiều false negatives (FN). Cụ thể FAR thấp, FRR cao.
- $\tau$ thấp → ngược lại.

Operating point được chọn dựa trên use case:

| Use case | Recommended threshold |
|---|---|
| Smart speaker (privacy-sensitive) | High $\tau$, FAR <0.1/hour acceptable, FRR ~5-10% |
| In-car voice assistant | Medium $\tau$, balance safety và usability |
| Wearable (constant battery drain concern) | Very high $\tau$, FAR <0.01/hour |
| Industrial control | Very low $\tau$, FRR <1% critical (don't miss commands) |

### 2.4 Performance metrics

Standard metrics for WWD:

#### 2.4.1 FAR (False Accept Rate)

Số lần system wake nhầm khi user không gọi:

$$
\text{FAR} = \frac{\text{False positives}}{\text{Total non-keyword time}} \quad \text{(per unit time)}
$$

Unit thường là FA/hour hoặc FA/day. Industry standard: <1 FA/hour cho consumer products.

#### 2.4.2 FRR (False Reject Rate)

Tỷ lệ system không thức khi user thực sự gọi:

$$
\text{FRR} = \frac{\text{False negatives}}{\text{Total keyword utterances}}
$$

Industry standard: FRR <5-10% cho clean speech, <20% cho noisy environments.

#### 2.4.3 DET curve

Tradeoff FAR vs FRR plotted trên DET (Detection Error Tradeoff) curve. Operating point = chọn một điểm trên curve.

Equal Error Rate (EER): điểm where FAR = FRR. Useful single-number metric nhưng không reflect real-world tuning.

#### 2.4.4 Latency

Time from "user finishes saying wake-word" to "system signals wake":

$$
\text{Latency} = t_{\text{detection}} - t_{\text{end of wake-word}}
$$

Target: <100-200 ms. Vượt mức này, user perceive lag.

---

## Phần 3 — Constraints Unique to Wake-Word Systems

WWD có constraints khắc nghiệt hơn nhiều speech tasks khác. Hiểu rõ giúp đánh giá đúng các architecture choices.

### 3.1 Always-on power budget

Wake-word model chạy liên tục 24/7. Power budget extremely tight:

- Phone: target <10 mW (so với LED screen ~500 mW, GPS ~50 mW).
- Smart speaker (AC powered): less tight, có thể 100-500 mW.
- Wearable (smartwatch): target <5 mW, often even lower.
- TWS earbuds (true wireless stereo): target <2 mW per ear.

Bao nhiêu battery drain? 10 mW × 24h = 240 mWh = ~5% pin của một phone (5000 mAh × 3.7V = 18500 mWh).

Achievable bằng:

1. Small model (50 KB - 500 KB).
2. INT8 hoặc INT4 quantization.
3. Specialized hardware (DSP chip với MAC accelerator).
4. Aggressive batching (process windows in batches).

### 3.2 Memory budget

Phần lớn wake-word models chạy trên microcontrollers:

- ARM Cortex-M4: 256 KB RAM, 1 MB Flash.
- ESP32: 520 KB RAM, 4 MB Flash.
- Apple Neural Engine: dedicated, MB-scale.

Model size constraint: <500 KB cho most embedded, <100 KB cho ultra-low-power MCUs.

### 3.3 Privacy considerations

Wake-word model là first line of privacy defense:

- Pre-wake: audio xử lý on-device, không upload cloud.
- Post-wake: audio sau wake-word có thể upload (depends on platform).

Nếu wake-word model có false positive, có thể accidentally upload private audio. Đây là lý do FAR < 1/hour là critical, không chỉ UX.

### 3.4 Acoustic environments

Wake-word phải work trong:

- Quiet bedroom (clean, easy).
- Noisy kitchen (background TV, dishwasher).
- Car (engine noise, music, road noise).
- Outdoor (wind, traffic).
- Multiple speakers nearby (cocktail party).

Training data MUST include diverse environments hoặc model fails trong real deployment.

### 3.5 Speaker variations

Same wake-word ("Hey Siri") nói bởi:

- Adult male vs female vs child.
- Native English vs accented English.
- Whispered vs shouted.
- Fast vs slow.
- Recovered from illness (raspy voice).

Robust to all = data augmentation aggressive trong training.

---

## Phần 4 — Classical Approaches (Pre-Deep Learning)

Trước 2014, WWD primarily dùng HMM-GMM hoặc DTW. Phần này quick overview cho historical context.

### 4.1 HMM-GMM keyword/garbage models

Idea: train một HMM-GMM cho keyword (e.g., "OK Google"), một cho "garbage" (everything else). Compare likelihood:

$$
\text{decision} = \log P(x | \text{keyword HMM}) - \log P(x | \text{garbage HMM})
$$

Pros: principled, can be trained với small data.
Cons: GMMs limited capacity, không capture complex acoustic patterns. EER thường 5-15% cho clean speech, much worse noisy.

### 4.2 Sliding window DTW

Dynamic Time Warping computes optimal alignment between template wake-word recording và sliding window of input audio.

$$
d(i, j) = c(i, j) + \min\{d(i-1, j), d(i, j-1), d(i-1, j-1)\}
$$

trong đó $c(i, j)$ là local distance giữa frame $i$ của template và frame $j$ của input.

Trigger khi DTW distance < threshold.

Pros: simple, no training needed.
Cons: scales poorly, sensitive to speaker variations, slow inference.

Modern relevance: DTW vẫn dùng cho speaker verification / user-personalized wake-word (cần ít data).

### 4.3 Why classical approaches failed

Đến khoảng năm 2015, khoảng cách accuracy giữa phương pháp cổ điển và deep learning trở nên đủ lớn để các hệ thống mới chuyển dần sang neural approaches:

- Capture complex acoustic patterns automatically.
- Robust to speaker, environment variations với augmentation.
- EER drops from 5-15% to 1-3% on standard benchmarks.

---

## Phần 5 — Modern Deep Learning Approaches

Modern WWD dùng small CNNs hoặc Conformer-based architectures. Phần này deep dive từng approach.

### 5.1 Small CNN baseline

Architecture đơn giản (dùng bởi Google Speech Commands paper 2017):

```python
import torch.nn as nn

class SmallCNN_KWS(nn.Module):
    def __init__(self, num_keywords: int = 12):
        super().__init__()
        # Input: mel spectrogram [B, 1, 40, 98] (40 mels, 98 frames = ~1 sec)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(20, 8), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global average pooling -> classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_keywords)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
```

Stats: ~150K params, ~100 KB INT8, ~5 mW inference on Cortex-M4.

Performance: ~92-94% accuracy trên Google Speech Commands V2 (35 keywords).

### 5.2 DS-CNN (Depthwise Separable CNN)

Improvement: replace standard conv với depthwise separable (factorize spatial + channel-wise convs):

$$
\text{Standard Conv}: H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot K^2 \quad\text{FLOPs}
$$

$$
\text{Depthwise Separable}: H \cdot W \cdot C_{\text{in}} \cdot K^2 + H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \quad\text{FLOPs}
$$

For typical $K = 3$, $C = 64$, reduction factor ~8x. Same accuracy, much fewer FLOPs.

Used by: Google "OK Google" v3-v5 (around 2018-2020).

### 5.3 TC-ResNet (Temporal Convolution ResNet)

Use 1D temporal convolutions instead of 2D spatial. Optimized for time-frequency features:

- Input: mel spectrogram treated as 1D sequence with $C = n_{\text{mels}}$ channels.
- Architecture: stack of 1D Conv-BN-ReLU blocks with residual connections.

Pros: faster than 2D CNN, similar accuracy.

### 5.4 Conformer-KWS (Google, 2022)

Apply Conformer architecture (originally for ASR) to WWD:

- Conformer block = convolution + self-attention.
- Captures both local (convolution) and global (attention) patterns.
- Better accuracy than CNNs ở same parameter budget.

Architecture details:

```python
class ConformerKWS(nn.Module):
    def __init__(self, num_keywords, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(80, hidden_dim)  # 80 mels
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads=4, ffn_dim=128)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_keywords)
    
    def forward(self, mel):  # [B, T, 80]
        x = self.input_proj(mel)
        for block in self.conformer_blocks:
            x = block(x)
        x = x.mean(dim=1)  # temporal pooling
        return self.classifier(x)
```

Performance: 96-97% on Google Speech Commands V2, ~200K params.

### 5.5 Streaming RNN-T cho continuous KWS

Cho continuous audio stream với multiple potential keywords, RNN-T architecture is natural fit:

- Encoder process audio streaming.
- Predictor maintains language model state.
- Joint network decides per frame to emit keyword token hoặc blank.

This is overkill cho single wake-word but useful cho open-vocabulary KWS hoặc multi-keyword scenarios.

### 5.6 Two-stage architecture (industry pattern)

Stage 1: very small always-on DNN (<10 KB) detects "candidates", FAR ~10/hour.

Stage 2: larger verification DNN (~500 KB) confirms candidates, drops FAR to <0.1/hour.

Used by Apple ("Hey Siri" original architecture), allows aggressive power saving (Stage 2 only activates when Stage 1 triggers).

---

## Phần 6 — Industry Implementations

Đào sâu từng industry implementation. Tất cả info từ public papers, blog posts, conference talks.

### 6.1 Apple "Hey Siri"

Apple publish "Hey Siri" architecture trong machine learning journal (2017):

**Two-stage architecture**:

1. **Stage 1 — Always-on DNN**: ~50 KB, runs on AOP (Always-On Processor) co-processor. ~5 mW. Outputs binary "candidate" / "no candidate".
2. **Stage 2 — Verification DNN**: larger (~MB scale), runs on Apple Neural Engine when Stage 1 triggers. Verifies if real "Hey Siri" vs sound-alike. Outputs probability.

**Personalization**: After user enrolls "Hey Siri" với their voice (3 times), system fine-tunes Stage 2 với user's voice samples. Reduces FAR for "Hey Siri" said by non-owner.

**Languages**: ~20 languages supported (Hey Siri in Vietnamese is "Hey Siri" — không localize).

### 6.2 Google "OK Google" / "Hey Google"

Multiple papers documenting evolution:

- **2014**: Original DNN-based keyword spotting (Chen et al., ICASSP 2014).
- **2017**: Convolutional and Recurrent Neural Networks for KWS (Sainath & Parada).
- **2020-2022**: Conformer-based KWS with improved accuracy.

Architecture today (per public docs):

1. **Always-on small DNN** trên Pixel chip (Tensor): ~100 KB, ~2 mW.
2. **Verification on AP** (Application Processor) khi triggered.

**Personalization**: Voice Match feature trains user-specific classifier.

### 6.3 Amazon Alexa

Alexa wake-word detection runs on Echo device locally:

- **Architecture**: small CNN với mel features.
- **Power**: device always plugged in (AC powered), less constrained than phone.
- **Verification**: cloud verification cho high-confidence false positive reduction.

Amazon publishes Alexa wake-word datasets (Hey Snips contributed by Snips, later acquired by Sonos).

### 6.4 Microsoft Cortana

Cortana wake-word architecture similar to Apple/Google: small DNN on-device + verification.

Notable: Microsoft maintained Cortana for years for enterprise but consumer adoption was limited compared to Siri/Alexa.

### 6.5 Open-source alternatives

#### 6.5.1 Porcupine (Picovoice)

- Commercial company với open-source SDKs.
- Pre-trained wake-word models cho popular phrases.
- Custom wake-word training service.
- Runs on Cortex-M, Cortex-A, x86, mobile.

#### 6.5.2 Snowboy (deprecated)

- Originally open-source by KITT.ai.
- Acquired by Baidu, then deprecated in 2020.
- Still used in legacy projects.

#### 6.5.3 Mycroft Precise

- Open-source wake-word for Mycroft AI voice assistant.
- TensorFlow Lite-based.
- Community-trained models cho "Hey Mycroft" và custom.

#### 6.5.4 Howl (CMU)

- Academic open-source toolkit for KWS research.
- Allows quick prototyping new models.

---

## Phần 7 — Training Data + Augmentation Strategies

Quality của WWD model phụ thuộc heavily vào data.

### 7.1 Standard datasets

#### 7.1.1 Google Speech Commands V2 (2018)

- 35 keywords (~3700 speakers).
- 105,000 utterances.
- 1-second clips at 16 kHz.
- License: Creative Commons.
- Used cho most academic WWD benchmarks.

#### 7.1.2 Hey Snips (2018)

- "Hey Snips" wake-word data.
- Collected by Snips (later acquired by Sonos).
- Open source.

#### 7.1.3 MLCommons Multilingual Spoken Words Corpus (2021)

- 350,000+ words across 50 languages.
- 6,400+ hours of audio.
- Released by ML Commons.

### 7.2 Augmentation strategies

WWD models benefit hugely from aggressive augmentation:

#### 7.2.1 Noise augmentation

Mix wake-word recordings với background noise:

```python
def augment_with_noise(clean_audio, noise_audio, snr_db):
    """Mix clean and noise at specified SNR."""
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # Scale noise to achieve target SNR
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (noise_power * snr_linear))
    
    return clean_audio + scale * noise_audio
```

SNR ranges: -5 dB to +20 dB common. Sub-zero SNR teach model to handle very noisy.

#### 7.2.2 Reverberation augmentation

Convolve với room impulse responses (RIRs):

```python
import scipy.signal

def add_reverb(audio, rir):
    """Convolve audio with room impulse response."""
    return scipy.signal.fftconvolve(audio, rir, mode='same')
```

RIR datasets: BUT Reverb DB, OpenSLR REVERB challenge.

#### 7.2.3 Speed perturbation

Speed up / slow down audio without changing pitch (or changing both):

```python
import librosa

def speed_perturb(audio, sr, speed):
    """Speed perturbation."""
    return librosa.effects.time_stretch(audio, rate=speed)
```

Typical: 0.9x, 1.0x, 1.1x. 3-way speed augmentation triple data effectively.

#### 7.2.4 Pitch shifting

Shift pitch up/down:

```python
def pitch_shift(audio, sr, semitones):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
```

Helps generalize across speakers (low pitch male, high pitch female/child).

#### 7.2.5 SpecAugment

On mel spectrogram (post-feature extraction):

- Frequency masking: zero out random frequency bands.
- Time masking: zero out random time slices.

#### 7.2.6 Hard negative mining

Most important: train on "sounds like wake-word but isn't":

- "Hey Sarah" (similar to "Hey Siri").
- "OK people" (similar to "OK Google").

Hard negatives dramatically reduce FAR.

### 7.3 Data collection for new wake-word

If you want custom wake-word (e.g., "Hey Vinfast"):

1. **Collect positive samples**: 1000-5000 utterances from diverse speakers.
2. **Collect negative samples**: hours of conversational speech (background) + hard negatives.
3. **Augment positives** với noise, reverb, speed (10-100x effective data).
4. **Train binary classifier**: typically Conformer-small or DS-CNN.
5. **Tune threshold** on validation set.

Realistic budget: 1-3 months engineering, 10,000-50,000 USD if outsourcing data collection.

---

## Phần 8 — Edge Deployment

WWD đặc thù về deployment vì power/memory constraints.

### 8.1 TFLite Micro

TensorFlow Lite for Microcontrollers (TFLite Micro): library cho ML on ARM Cortex-M and similar:

```python
# Train PyTorch model
model = train_kws_model()

# Convert to ONNX
torch.onnx.export(model, dummy_input, "kws.onnx")

# Convert ONNX -> TF -> TFLite
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("kws.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("kws_tf")

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("kws_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()

with open("kws.tflite", "wb") as f:
    f.write(tflite_model)
```

TFLite Micro flash usage: ~50-200 KB model + ~50 KB runtime.

### 8.2 ONNX Runtime Mobile

Alternative: ONNX Runtime với mobile-optimized binary:

- iOS: ONNX Runtime iOS framework, ~5 MB binary.
- Android: ONNX Runtime AAR, ~10 MB binary.

Larger than TFLite but supports wider model architectures.

### 8.3 Core ML (Apple)

Apple ecosystem: convert to Core ML for Apple Neural Engine acceleration:

```python
import coremltools as ct

# Convert PyTorch model
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="mel", shape=(1, 1, 40, 98))],
    compute_precision=ct.precision.FLOAT16,
)

mlmodel.save("kws.mlmodel")
```

Performance: ~2-5 mW on iPhone Neural Engine, sub-50ms inference.

### 8.4 MediaPipe (Google)

Google's framework for on-device ML, supports KWS pipeline:

- Audio preprocessing on CPU.
- Model inference on GPU/NPU.
- Post-processing for threshold tuning.

---

## Phần 9 — Personalization (User-Specific Wake-Words)

Generic wake-word models trigger on anyone's voice. Personalization improves FAR by 5-10x.

### 9.1 Enrollment process

1. User says wake-word 3-5 times in quiet environment.
2. Extract speaker embedding from each utterance (using pre-trained speaker encoder).
3. Average embeddings → user's "voice signature".

### 9.2 Inference with personalization

When wake-word candidate detected:

1. Extract speaker embedding from candidate utterance.
2. Compute cosine similarity với user's signature.
3. Accept only if similarity > threshold.

```python
def personalized_wake_word(audio, base_score, user_embedding, speaker_encoder):
    """Combine base wake-word score with speaker verification."""
    # Speaker embedding from candidate audio
    candidate_emb = speaker_encoder(audio)
    
    # Cosine similarity với user
    similarity = F.cosine_similarity(candidate_emb, user_embedding, dim=-1)
    
    # Combined score
    combined = base_score * similarity.sigmoid()
    return combined > threshold
```

### 9.3 Limitations of personalization

- Single voice only (won't work for family with shared device).
- Voice changes over time (illness, age).
- Need re-enrollment periodically.

Multi-user systems use multiple voice profiles (e.g., Amazon Alexa "Voice ID").

---

## Phần 10 — Vietnamese Wake-Words

Trong context Vietnam, các wake-word notable:

### 10.1 Wake-words trên thị trường Vietnam

- **"Ok Vinfast"** — VinFast in-car voice assistant.
- **"Chào Zalo"** / **"Hey Kiki"** — ZaloAI assistants.
- **"Hey Siri"** / **"OK Google"** — Apple/Google products (English wake-word vẫn dùng cho user Vi).

### 10.2 Tonal language considerations

Tiếng Việt có 6 thanh điệu. Wake-word "Ok Vinfast" sai thanh điệu (e.g., "Ok Vĩnfast") không nên trigger.

Solution:

1. Tone-aware features: extract F0 (pitch) contour, include trong model input.
2. Augment với tone-shifted versions cho hard negatives.
3. Use character-level model thay vì BPE (preserve tonal distinctions).

### 10.3 Code-switching wake-words

Wake-word có thể là mixed Vi-En: "Hey Vinfast", "Chào Google" (hypothetical).

Train với code-switched data ensure model handles correctly.

### 10.4 Recommended approach cho Vietnamese custom wake-word

1. **Data collection**: 5,000+ utterances, diverse speakers (Bắc/Trung/Nam dialects).
2. **Hard negatives**: include "Hey VinHomes", "Hey VinGroup" (similar Vinfast).
3. **Augmentation**: in-car noise (engine, music, road).
4. **Model**: small Conformer-KWS, ~200K params, INT8 quantized.
5. **Deployment**: ONNX Runtime Mobile cho Android automotive.
6. **Personalization**: tied to user phone connection (optional).

---

## Phần 11 — Tóm tắt

### 11.1 Key takeaways

1. **WWD là binary classification specialized cho 1 phrase**, not ASR. Different constraints, different architectures.
2. **Power budget < 10 mW** là constraint nhất, requires small models (50-500 KB) + DSP/MCU deployment.
3. **FAR < 1/hour và FRR < 5-10%** là industry standard. Threshold tuning is key knob.
4. **Modern WWD**: Conformer-KWS / DS-CNN với INT8 quantization. ~96-97% accuracy on Google Speech Commands V2.
5. **Two-stage architecture** (Apple "Hey Siri") allows aggressive power saving: small always-on detector + larger verification when triggered.
6. **Data augmentation aggressive**: noise mixing, reverb, speed/pitch perturbation, hard negative mining.
7. **Personalization**: speaker embedding-based verification reduces FAR 5-10x.
8. **Edge deployment**: TFLite Micro for MCU, Core ML for Apple, ONNX Runtime Mobile for cross-platform.
9. **Vietnamese wake-words**: tone-aware features, dialect coverage, in-car noise augmentation critical.

### 11.2 References

1. Chen, G. et al. (2014). Small-footprint Keyword Spotting Using Deep Neural Networks. ICASSP.
2. Sainath, T. & Parada, C. (2015). Convolutional Neural Networks for Small-footprint Keyword Spotting. Interspeech.
3. Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209.
4. Coucke, A. et al. (2019). Efficient Keyword Spotting using Dilated Convolutions and Gating. ICASSP.
5. Apple ML Journal (2017). Hey Siri: An On-device DNN-powered Voice Trigger for Apple's Personal Assistant.
6. Picovoice (2024). Porcupine Wake Word Engine documentation.
7. Google AI Blog (2017). Improving Voice Search for People with Speech Impairments.
8. Mukhopadhyay, A. et al. (2024). Conformer-based Streaming Keyword Spotting for Edge Devices.
