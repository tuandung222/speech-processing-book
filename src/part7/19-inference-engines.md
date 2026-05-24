# Chương 19: Inference Engines và Tooling

## Vì sao chương này quan trọng

Sau khi train được mô hình Speech AI chất lượng, bài toán tiếp theo là **deploy nó vào production với chi phí và latency hợp lý**. Đây là lúc inference engine, optimization toolkit, và data preprocessing/augmentation library trở nên quyết định. Khác biệt giữa naive PyTorch inference và pipeline tối ưu (TensorRT, vLLM, Triton) có thể là 5-10x throughput hoặc 3-5x giảm cost server.

Chương này khảo sát hệ sinh thái inference cho Speech AI ở ba mức:

- **Inference engines core**: TensorRT, ONNX Runtime, OpenVINO, CTranslate2 (faster-whisper), vLLM cho Speech LLM.
- **Serving frameworks**: Triton Inference Server, BentoML, Ray Serve cho production pipeline phức tạp.
- **Supporting libraries**: torchaudio, librosa, soundfile, audiomentations cho preprocessing và augmentation.

Đặc biệt, chương dành một deep dive cho **WeNet** (kế thừa Phần 18 nhưng từ góc nhìn inference), trình bày cách export ONNX và LibTorch cho deployment.

> **Cấu trúc chương**
>
> - **Phần 1**: tổng quan inference engines, tiêu chí lựa chọn (latency, throughput, cost, cross-platform).
> - **Phần 2**: TensorRT, ONNX Runtime, OpenVINO cho ASR và TTS.
> - **Phần 3**: CTranslate2 và faster-whisper, optimization chuyên cho Whisper.
> - **Phần 4**: vLLM cho Speech LLM (Qwen2-Audio, Qwen3-Omni, Moshi self-host).
> - **Phần 5**: Triton, BentoML, Ray Serve cho serving pipeline.
> - **Phần 6**: speech processing libraries ecosystem (preprocessing, augmentation, pretrained loader, quantization).
> - **Phần 7**: WeNet deep guide cho production deployment.

## Tổng quan

Chương này trình bày chi tiết các inference engines phổ biến nhất để **deploy speech processing models** trong production. Từ **TensorRT + Triton** (NVIDIA stack) đến **ONNX Runtime**, **CTranslate2**, và các alternatives khác.

> **📝 Tại sao cần Inference Engine riêng?**
>
> PyTorch inference **chậm** và **tốn memory** cho production:
>
> - Không có graph optimization
> - Dynamic shapes gây overhead
> - Không tận dụng hardware-specific optimizations (TensorCores, INT8)
>
> Inference engines tối ưu: **2-10x faster**, **50-70% less memory** so với raw PyTorch.



## So sánh Tổng quan

| Engine | Tổ chức | Hardware | Quantization | Streaming | Speech-specific |
|--------|---------|----------|-------------|-----------|----------------|
| **TensorRT** | NVIDIA | GPU (NVIDIA) | INT8, FP16 | Có | Không |
| **Triton** | NVIDIA | GPU/CPU | Via backends | Có | Không |
| **ONNX Runtime** | Microsoft | GPU/CPU/Edge | INT8, QDQ | Có | Không |
| **CTranslate2** | OpenNMT | GPU/CPU | INT8, FP16 | Có | **Whisper** |
| **faster-whisper** | Community | GPU/CPU | INT8, FP16 | Có | **Whisper** |
| **vLLM** | UC Berkeley | GPU | AWQ, GPTQ | Có | Speech LLM |
| **OpenVINO** | Intel | CPU/iGPU | INT8 | Có | Không |
| **BentoML** | BentoML | Any | Via backends | Có | Không |
| **Ray Serve** | Anyscale | Any | Via backends | Có | Không |
| **WhisperX** | Community | GPU | Via CTranslate2 | Có | **Whisper+** |

: So sánh Inference Engines cho Speech <a id="tbl-inference-engines"></a>

## TensorRT + Triton Inference Server

### TensorRT

TensorRT [^nvidia2024tensorrt] là inference optimizer của NVIDIA:

**Optimization Pipeline:**

<a id="eq-trt-pipeline"></a>

$$
\text{PyTorch/ONNX} \xrightarrow{\text{TRT Builder}} \text{TRT Engine} \xrightarrow{\text{TRT Runtime}} \text{Inference}
$$

**Các tối ưu chính:**

1. **Layer Fusion**: Combine multiple operations thành 1 kernel
2. **Precision Calibration**: FP32 → FP16/INT8 với calibration data
3. **Kernel Auto-tuning**: Chọn kernel tối ưu cho hardware cụ thể
4. **Dynamic Shapes**: Support variable-length audio input

```python
#| eval: false
#| code-fold: true
#| code-summary: "TensorRT Conversion cho ASR Model"
# import tensorrt as trt
# import torch
# from torch import Tensor

# Step 1: Export PyTorch to ONNX
# def export_to_onnx(
#     model: torch.nn.Module,
#     dummy_input: Tensor,  # [1, T, 80] - mel spectrogram
#     onnx_path: str = "model.onnx",
# ) -> None:
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=["audio_features"],
#         output_names=["logits"],
#         dynamic_axes={
#             "audio_features": {0: "batch", 1: "time"},
#             "logits": {0: "batch", 1: "time"},
#         },
#         opset_version=17,
#     )

# Step 2: Build TensorRT Engine
# def build_trt_engine(
#     onnx_path: str,
#     engine_path: str,
#     precision: str = "fp16",  # fp32, fp16, int8
#     max_batch: int = 8,
#     max_seq_len: int = 3000,
# ) -> None:
#     logger = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(logger)
#     network = builder.create_network(
#         1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     )
#     parser = trt.OnnxParser(network, logger)
#
#     with open(onnx_path, "rb") as f:
#         parser.parse(f.read())
#
#     config = builder.create_builder_config()
#     config.set_memory_pool_limit(
#         trt.MemoryPoolType.WORKSPACE, 4 << 30  # 4GB
#     )
#
#     if precision == "fp16":
#         config.set_flag(trt.BuilderFlag.FP16)
#     elif precision == "int8":
#         config.set_flag(trt.BuilderFlag.INT8)
#         # Need calibrator for INT8
#
#     # Dynamic shapes
#     profile = builder.create_optimization_profile()
#     profile.set_shape(
#         "audio_features",
#         min=(1, 1, 80),
#         opt=(4, 1500, 80),
#         max=(max_batch, max_seq_len, 80),
#     )
#     config.add_optimization_profile(profile)
#
#     engine = builder.build_serialized_network(network, config)
#     with open(engine_path, "wb") as f:
#         f.write(engine)
```

### Triton Inference Server

Triton [^nvidia2024triton] là serving framework hỗ trợ multiple backends:

**Đặc điểm chính:**

- **Multi-model serving**: Serve nhiều models trên 1 server
- **Dynamic batching**: Tự động batch requests
- **Model pipelines**: Chain multiple models (e.g., VAD → ASR → NLU)
- **Multiple backends**: TensorRT, ONNX, PyTorch, TensorFlow, Python
- **Metrics**: Prometheus metrics built-in

**Model Repository Structure:**

```
model_repository/
├── asr_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan          # TensorRT engine
├── asr_decoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx          # ONNX model
├── vad/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt            # PyTorch model
└── pipeline/
    ├── config.pbtxt            # Ensemble/BLS config
    └── 1/
        └── model.py            # Python backend
```

**Config Example:**

```
# config.pbtxt for ASR encoder
name: "asr_encoder"
platform: "tensorrt_plan"
max_batch_size: 8

input [
  {
    name: "audio_features"
    data_type: TYPE_FP16
    dims: [-1, 80]
  }
]

output [
  {
    name: "encoder_output"
    data_type: TYPE_FP16
    dims: [-1, 256]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
```

### Benchmark TensorRT vs PyTorch

| Model | PyTorch (ms) | TensorRT FP16 (ms) | TensorRT INT8 (ms) | Speedup |
|-------|-------------|--------------------|--------------------|---------|
| Conformer-L (encoder) | 45 | 12 | 8 | 3.7-5.6x |
| Whisper-large-v3 | 850 | 280 | 190 | 3.0-4.5x |
| FastConformer | 25 | 7 | 5 | 3.6-5.0x |
| HiFi-GAN vocoder | 15 | 4 | 3 | 3.7-5.0x |

: TensorRT speedup trên A100 GPU (batch=1, 10s audio) <a id="tbl-trt-benchmark"></a>

## ONNX Runtime

### Tổng quan

ONNX Runtime [^onnxruntime2024] từ Microsoft - cross-platform inference:

- **CPU + GPU + Edge** support
- **Quantization** built-in (dynamic, static, QDQ)
- **Execution Providers**: CUDA, TensorRT, DirectML, OpenVINO, CoreML
- **Rất phổ biến** cho production deployment

### Speech Model Inference

```python
#| eval: false
#| code-fold: true
#| code-summary: "ONNX Runtime ASR Inference"
import numpy as np
from numpy import ndarray

# import onnxruntime as ort

# def create_asr_session(
#     model_path: str,
#     use_gpu: bool = True,
# ) -> ort.InferenceSession:
#     """Create ONNX Runtime session for ASR."""
#     providers: list[str] = []
#     if use_gpu:
#         providers.append("CUDAExecutionProvider")
#     providers.append("CPUExecutionProvider")
#
#     session = ort.InferenceSession(
#         model_path,
#         providers=providers,
#         sess_options=_get_session_options(),
#     )
#     return session

# def _get_session_options() -> ort.SessionOptions:
#     opts = ort.SessionOptions()
#     opts.graph_optimization_level = (
#         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     )
#     opts.intra_op_num_threads = 4
#     opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     return opts

# Example inference:
# session = create_asr_session("conformer.onnx")
# mel: ndarray = np.random.randn(1, 500, 80).astype(np.float32)
# result = session.run(None, {"mel_spectrogram": mel})
# logits: ndarray = result[0]  # [1, 125, vocab_size]
```

### Quantization với ONNX Runtime

| Quantization | Accuracy Drop | Speedup (CPU) | Speedup (GPU) |
|-------------|--------------|---------------|---------------|
| FP32 (baseline) | 0% | 1.0x | 1.0x |
| FP16 | < 0.1% | 1.0x | 1.5-2.0x |
| Dynamic INT8 | 0.1-0.5% | **2.0-3.0x** | 1.5x |
| Static INT8 (QDQ) | 0.2-1.0% | **2.5-4.0x** | 2.0-3.0x |

: ONNX Runtime Quantization results <a id="tbl-ort-quantization"></a>

## CTranslate2 & faster-whisper

### CTranslate2

CTranslate2 [^klein2020opennmt] từ OpenNMT - optimized cho **Transformer inference**:

- **8x less memory** so với PyTorch
- **INT8/FP16** quantization tự động
- **CPU + GPU** support
- Đặc biệt tối ưu cho **Whisper** models

### faster-whisper

faster-whisper [^faster_whisper] dùng CTranslate2 backend cho Whisper:

```python
#| eval: false
#| code-fold: true
#| code-summary: "faster-whisper Inference"
# from faster_whisper import WhisperModel

# def transcribe_fast(
#     audio_path: str,
#     model_size: str = "large-v3",
#     device: str = "cuda",
#     compute_type: str = "float16",  # float16, int8, int8_float16
# ) -> str:
#     """Fast Whisper inference with CTranslate2 backend."""
#     model = WhisperModel(
#         model_size,
#         device=device,
#         compute_type=compute_type,
#     )
#
#     segments, info = model.transcribe(
#         audio_path,
#         language="vi",
#         beam_size=5,
#         vad_filter=True,      # Built-in VAD
#         vad_parameters=dict(
#             min_silence_duration_ms=500,
#         ),
#     )
#
#     transcript: str = " ".join(
#         segment.text for segment in segments
#     )
#     return transcript

# Benchmark:
# PyTorch Whisper large-v3: ~850ms/10s audio (A100)
# faster-whisper large-v3 FP16: ~280ms/10s audio (A100)
# faster-whisper large-v3 INT8: ~180ms/10s audio (A100)
```

### WhisperX

WhisperX [^bain2023whisperx] mở rộng faster-whisper với:

- **Batched inference**: Process nhiều audio segments cùng lúc
- **Word-level timestamps**: Forced alignment với wav2vec2
- **Speaker diarization**: Tích hợp pyannote.audio
- **VAD**: Silero VAD để segment audio trước khi transcribe

| Feature | Whisper (OpenAI) | faster-whisper | WhisperX |
|---------|-----------------|----------------|----------|
| Speed | 1x | **3-4x** | **3-4x** |
| Word timestamps | Có (slow) | Có | **Có (fast, accurate)** |
| Speaker diarization | Không | Không | **Có** |
| Batch inference | Không | Không | **Có** |
| VAD pre-processing | Không | Có | **Có (Silero)** |

: Whisper variants comparison <a id="tbl-whisper-variants"></a>

## vLLM cho Speech LLMs

vLLM [^kwon2023efficient] cho serving Speech LLMs (Qwen2-Audio, SALMONN):

- **PagedAttention**: Efficient KV-cache management
- **Continuous batching**: Dynamic request scheduling
- **Tensor parallelism**: Multi-GPU serving

```python
#| eval: false
#| code-fold: true
#| code-summary: "vLLM Speech LLM Serving"
# from vllm import LLM, SamplingParams

# # Serve Qwen2-Audio
# llm = LLM(
#     model="Qwen/Qwen2-Audio-7B-Instruct",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.9,
# )

# # Inference with audio input
# sampling_params = SamplingParams(
#     temperature=0.7,
#     max_tokens=256,
# )
# prompt = "Transcribe the following audio in Vietnamese:"
# # result = llm.generate(prompt, sampling_params)
```

## OpenVINO (Intel)

OpenVINO [^openvino2024] từ Intel cho **CPU-optimized inference**:

- Tối ưu cho Intel CPUs (AVX-512, VNNI)
- INT8 quantization với NNCF
- Phù hợp cho edge deployment

| Model | PyTorch CPU (ms) | OpenVINO FP32 (ms) | OpenVINO INT8 (ms) |
|-------|-----------------|--------------------|--------------------|
| Whisper-small | 2500 | 1200 | 600 |
| Conformer-M | 180 | 85 | 45 |
| Silero VAD | 5 | 2 | 1.5 |

: OpenVINO speedup trên Intel Xeon (batch=1) <a id="tbl-openvino-benchmark"></a>

## BentoML & Ray Serve

### BentoML

BentoML [^bentoml2024] - framework cho ML model serving:

- **Declarative API** để define services
- **Adaptive batching** tự động
- **Multi-model composition** cho speech pipelines
- **Docker + Kubernetes** deployment built-in

### Ray Serve

Ray Serve [^moritz2018ray] cho **distributed model serving**:

- **Horizontal scaling** tự động
- **Multi-node deployment**
- **Model composition** cho complex pipelines
- Tích hợp với Ray ecosystem (Ray Train, Ray Data)

## Speech Pipeline Architecture

Một production speech pipeline điển hình:

<a id="eq-speech-pipeline"></a>

$$
\text{Audio} \xrightarrow{\text{VAD}} \text{Segments} \xrightarrow{\text{ASR}} \text{Text} \xrightarrow{\text{NLU}} \text{Intent}
$$

### Deployment Pattern

| Component | Engine | Hardware | Latency Target |
|-----------|--------|----------|---------------|
| VAD | ONNX Runtime | CPU | < 5ms |
| ASR Encoder | TensorRT | GPU | < 50ms |
| ASR Decoder | TensorRT | GPU | < 20ms |
| LM Rescoring | CTranslate2 | CPU/GPU | < 30ms |
| NLU | vLLM | GPU | < 100ms |

: Production Speech Pipeline components <a id="tbl-pipeline-components"></a>

### Latency Optimization Tips

1. **Batch dynamic shapes**: Pad to nearest power-of-2
2. **Cache KV-states**: Reuse previous computation cho streaming
3. **Model pruning**: Remove redundant layers/heads
4. **Speculative decoding**: Cho autoregressive models
5. **Pipeline parallelism**: Overlap VAD, ASR, NLU computation

> **⚠️ GPU Memory Budget**
>
> Trên 1 GPU A100 (80GB), có thể serve:
>
> - **Whisper large-v3** (FP16): ~3GB → batch_size=16 dễ dàng
> - **Qwen2-Audio-7B** (FP16): ~14GB → batch_size=4
> - **SeamlessM4T-v2-large**: ~10GB → batch_size=8
> - Remaining memory cho KV-cache và dynamic batching



## Recommendation

| Use Case | Engine đề xuất | Lý do |
|----------|---------------|-------|
| Whisper deployment | **faster-whisper** (CTranslate2) | 3-4x faster, INT8, simple API |
| ASR streaming | **TensorRT + Triton** | Lowest latency, dynamic batching |
| Speech LLM serving | **vLLM** | PagedAttention, continuous batching |
| CPU-only deployment | **ONNX Runtime** hoặc **OpenVINO** | Cross-platform, INT8 |
| Edge/mobile | **ONNX Runtime Mobile** | Lightweight, multi-platform |
| Complex pipelines | **Triton** hoặc **Ray Serve** | Multi-model, scaling |
| Quick prototype | **BentoML** | Simple API, Docker export |

: Inference Engine Recommendations <a id="tbl-engine-recommendations"></a>

## Phần Mở rộng: Speech Processing Libraries Ecosystem

Inference engine chỉ là một mảnh của puzzle. Production speech pipeline cần nhiều libraries để xử lý data preprocessing, augmentation, model loading, và optimization. Phần này khảo sát ecosystem rộng hơn theo từng category.

### A. Audio data preprocessing libraries

Trước khi feed vào model, audio cần được loaded, resampled, normalized. Đây là các thư viện chính:

#### A.1 `torchaudio`

PyTorch native audio library. Strengths: GPU-accelerated transforms, tight integration với PyTorch models.

```python
import torchaudio
import torchaudio.transforms as T

# Load (returns waveform, sample_rate)
waveform, sr = torchaudio.load("audio.wav")  # [channels, samples]

# Resample to 16kHz (chuẩn cho speech)
resampler = T.Resample(orig_freq=sr, new_freq=16000)
waveform_16k = resampler(waveform)

# Mel spectrogram on GPU
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
).cuda()
mel_spec = mel_transform(waveform_16k.cuda())  # [channels, n_mels, frames]
```

#### A.2 `librosa`

Pure Python (CPU only), academic standard cho audio analysis. Slower than torchaudio but more analysis features.

```python
import librosa

# Load
y, sr = librosa.load("audio.wav", sr=16000)  # 1D numpy array

# Many analysis functions
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
```

**Khi nào dùng**: research, exploratory data analysis, không có GPU available. Tránh cho production training (slow).

#### A.3 `soundfile`

Fast audio I/O. Read/write nhiều formats (WAV, FLAC, OGG).

```python
import soundfile as sf

# Read
data, sr = sf.read("audio.wav")

# Write
sf.write("output.wav", data, sr, subtype='PCM_16')
```

#### A.4 `pydub`

Higher-level audio manipulation (cắt, ghép, fade in/out). Wrapper trên ffmpeg.

```python
from pydub import AudioSegment

audio = AudioSegment.from_wav("input.wav")
clip = audio[1000:5000]  # 1-5 seconds
faded = clip.fade_in(500).fade_out(500)
faded.export("clip.wav", format="wav")
```

### B. Data augmentation libraries

Augmentation là essential cho ASR training. Boost accuracy 5-10% trên test set.

#### B.1 `audiomentations`

Most popular Python library cho audio augmentation.

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

augmented = augment(samples=audio_array, sample_rate=16000)
```

#### B.2 `torch-audiomentations`

GPU version of audiomentations. Differentiable, batched.

```python
from torch_audiomentations import Compose, Gain, PolarityInversion

apply_augmentation = Compose(
    transforms=[
        Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.5),
        PolarityInversion(p=0.5)
    ]
)
# Works on batched GPU tensors [batch, channels, samples]
augmented_batch = apply_augmentation(audio_batch, sample_rate=16000)
```

#### B.3 `pyroomacoustics` + `rir-generator`

Simulate room impulse responses cho realistic reverb augmentation.

```python
import pyroomacoustics as pra
import numpy as np

# Simulate small room
room = pra.ShoeBox([4, 5, 3], fs=16000, max_order=3)  # dimensions in meters
room.add_source([1, 1, 1], signal=clean_audio)
room.add_microphone([2, 3, 1.5])
room.simulate()
reverberant_audio = room.mic_array.signals[0]
```

#### B.4 `nlpaug` audio submodule

Multimodal augmentation library, có audio support đơn giản. Less features than audiomentations but easy API.

#### B.5 SpecAugment via torchaudio

```python
import torchaudio.transforms as T

freq_masking = T.FrequencyMasking(freq_mask_param=15)
time_masking = T.TimeMasking(time_mask_param=35)

# Apply to mel spectrogram
mel = compute_mel(audio)
mel_aug = time_masking(freq_masking(mel))
```

SpecAugment được áp dụng directly trên mel spectrogram, không phải trên waveform.

### C. Loading pretrained models (HuggingFace Transformers focus)

HF Transformers là framework phổ biến để tải và thử nghiệm các speech models hiện đại từ HuggingFace Hub.

#### C.1 Whisper

```python
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
import torch

# Load components
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,  # half precision for speed
    device_map="cuda:0",
)

# Inference
audio, sr = load_audio("test.wav", target_sr=16000)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
inputs = inputs.to("cuda:0", torch.float16)

predicted_ids = model.generate(
    inputs.input_features,
    language="vi",  # Vietnamese
    task="transcribe",
)
transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
```

#### C.2 Wav2Vec 2.0 / PhoWhisper

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# PhoWhisper (Vietnamese)
processor = Wav2Vec2Processor.from_pretrained("vinai/PhoWhisper-base")
model = Wav2Vec2ForCTC.from_pretrained("vinai/PhoWhisper-base")

# Process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
```

#### C.3 Loading custom checkpoint

```python
# Load from local path
model = WhisperForConditionalGeneration.from_pretrained("/path/to/local/whisper")

# Load from HF cache
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(local_dir)

# Load specific revision
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    revision="abc123def",  # specific commit
)
```

### D. Optimization stack

Production needs aggressive optimization. Stack overview:

#### D.1 ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Convert PyTorch -> ONNX
dummy_input = torch.randn(1, 80, 3000)  # whisper mel
torch.onnx.export(
    model,
    dummy_input,
    "whisper.onnx",
    opset_version=17,
    input_names=["mel"],
    output_names=["logits"],
)

# Run với ONNX Runtime
session = ort.InferenceSession(
    "whisper.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
outputs = session.run(None, {"mel": dummy_input.numpy()})
```

#### D.2 TensorRT-LLM

Cho LLM-based speech models (Qwen2-Audio, Moshi).

```bash
# Install
pip install tensorrt_llm

# Convert HF model -> TRT engine
python -m tensorrt_llm.examples.llama.build \
    --model_dir Qwen2-Audio-7B \
    --output_dir qwen2_audio_trt \
    --dtype float16 \
    --max_batch_size 16
```

#### D.3 vLLM cho Speech LLMs

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-Audio-7B-Instruct", dtype="float16")
sampling = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(prompts, sampling)
```

#### D.4 Triton Inference Server

Production serving với multi-model support.

```bash
# Model repository structure
model_repository/
├── whisper/
│   ├── config.pbtxt
│   └── 1/model.onnx
├── tts/
│   └── ...
└── llm/
    └── ...

# Launch
tritonserver --model-repository=/path/to/model_repository
```

#### D.5 Quantization libraries

- **bitsandbytes**: INT8 và INT4 quantization for HF models.
- **AWQ**: activation-aware quantization, popular for LLMs.
- **GPTQ**: weight-only quantization.
- **PyTorch native quantization**: torch.quantization for PyTorch models.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    quantization_config=quant_config,
    device_map="auto",
)
```

---

## Phần Mở rộng: WeNet Deep Guide

WeNet là một trong những ASR training/inference frameworks phổ biến nhất, đặc biệt cho streaming ASR và Chinese/Asian languages. Phần này là hướng dẫn deep dive về cách dùng WeNet trong production.

### WeNet 1 — Design philosophy

WeNet (Mobvoi/Xiaomi, 2021+) được thiết kế xoay quanh **U2 architecture** (Unified streaming + non-streaming): một mô hình train được, có thể chạy cả streaming và offline mode chỉ bằng cách thay decoding strategy.

Key features:

- **Joint CTC + attention loss**: best of both worlds.
- **Dynamic chunk training**: train với variable chunk sizes, deploy với fixed chunk.
- **Streaming friendly**: causal Conformer encoder, chunk-based attention.
- **Production-ready export**: ONNX, LibTorch, TensorRT.

### WeNet 2 — Installation

```bash
# Clone repo
git clone https://github.com/wenet-e2e/wenet.git
cd wenet

# Install Python deps
pip install -r requirements.txt

# Optional: install C++ runtime for production
cd runtime/server/x86
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### WeNet 3 — Data preparation (Kaldi-style)

WeNet uses Kaldi data format:

```
data/train/
├── wav.scp        # utterance_id /path/to/audio.wav
├── text           # utterance_id transcription
├── utt2spk        # utterance_id speaker_id
└── spk2utt        # speaker_id utterance_id [utterance_id ...]
```

Example `wav.scp`:

```
BAC009S0002W0123 /data/aishell/wav/train/S0002/BAC009S0002W0123.wav
BAC009S0002W0124 /data/aishell/wav/train/S0002/BAC009S0002W0124.wav
```

Example `text`:

```
BAC009S0002W0123 而 对 楼 市 成 交 抑 制 作 用 最 大 的 限 购
BAC009S0002W0124 也 成 为 地 方 政 府 的 调 控 工 具
```

Generate Kaldi-style data từ raw audio:

```bash
# Có example scripts trong examples/aishell/
cd examples/aishell/s0
bash local/aishell_data_prep.sh /path/to/aishell/data
```

### WeNet 4 — Configuration YAML

WeNet config example (`conf/train_unified_conformer.yaml`):

```yaml
# Model architecture
input_dim: 80
output_dim: 5000  # BPE vocab size

# Encoder: Conformer
encoder: conformer
encoder_conf:
    output_size: 256
    attention_heads: 4
    linear_units: 2048
    num_blocks: 12
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d  # Subsampling
    normalize_before: true
    cnn_module_kernel: 15
    use_cnn_module: True
    activation_type: 'swish'
    pos_enc_layer_type: 'rel_pos'
    selfattention_layer_type: 'rel_selfattn'

# Decoder: Bi-Transformer
decoder: bitransformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 3
    r_num_blocks: 3  # Right-to-left decoder for rescoring
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0

# Joint CTC + Attention loss
model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1
    length_normalized_loss: false
    reverse_weight: 0.3  # Bidirectional decoder weight

# Optimizer
optim: adam
optim_conf:
    lr: 0.002
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 25000

# Data
dataset_conf:
    filter_conf:
        max_length: 40960  # samples (frames)
        min_length: 0
        token_max_length: 200
        token_min_length: 1
    resample_conf:
        resample_rate: 16000
    speed_perturb: true  # Speed augmentation
    fbank_conf:
        num_mel_bins: 80
        frame_shift: 10
        frame_length: 25
        dither: 0.1
    spec_aug: true  # SpecAugment
    spec_aug_conf:
        num_t_mask: 2
        num_f_mask: 2
        max_t: 50
        max_f: 10
    shuffle: true
    sort: true
    batch_conf:
        batch_type: 'dynamic'
        max_frames_in_batch: 12000
```

### WeNet 5 — Training command

```bash
# Single GPU
python wenet/bin/train.py \
    --config conf/train_unified_conformer.yaml \
    --data_type raw \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/conformer \
    --num_workers 4 \
    --pin_memory

# Multi-GPU distributed
torchrun --nproc_per_node=4 --master_port=29500 \
    wenet/bin/train.py \
    --config conf/train_unified_conformer.yaml \
    --data_type raw \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/conformer
```

Monitor via TensorBoard:

```bash
tensorboard --logdir exp/conformer/tensorboard
```

Typical training time: 4-5 ngày on 8x V100 for AISHELL-1 (170h Chinese data).

### WeNet 6 — Decoding modes

WeNet supports 4 decoding modes:

1. **ctc_greedy_search**: fastest, lowest accuracy.
2. **ctc_prefix_beam_search**: standard CTC beam, balanced.
3. **attention_rescoring**: best accuracy, slowest. Uses bidirectional decoder.
4. **attention**: pure attention decoding (offline only).

```bash
# Offline decoding với attention rescoring (best accuracy)
python wenet/bin/recognize.py \
    --config exp/conformer/train.yaml \
    --test_data data/test/data.list \
    --checkpoint exp/conformer/avg_30.pt \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict data/dict/lang_char.txt \
    --ctc_weight 0.3 \
    --reverse_weight 0.3 \
    --result_file exp/conformer/test_attention_rescoring/text \
    --mode attention_rescoring
```

### WeNet 7 — Streaming inference setup

Streaming mode dùng chunk-based attention:

```python
import wenet
import torch

# Load model
model = wenet.load_model('exp/conformer/avg_30.pt')
model.eval()

# Streaming config
chunk_size = 16  # frames per chunk (~160ms)
num_left_chunks = 4  # Look back 4 chunks (640ms history)

# Simulate streaming
audio_chunks = stream_audio_from_mic()  # generator yielding chunks

for chunk in audio_chunks:
    # Decode this chunk
    result = model.streaming_decode(
        chunk,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        decoding_chunk_size=chunk_size,
        simulate_streaming=True,
    )
    print(f"Partial: {result.partial_text}")
    if result.is_final:
        print(f"Final: {result.text}")
```

### WeNet 8 — Export to ONNX/LibTorch

For production deployment:

```bash
# Export to ONNX (cho ONNX Runtime, TensorRT)
python wenet/bin/export_onnx_cpu.py \
    --config exp/conformer/train.yaml \
    --checkpoint exp/conformer/avg_30.pt \
    --output_dir exp/conformer/onnx \
    --chunk_size 16 \
    --num_decoding_left_chunks 4

# Export to LibTorch (cho C++ runtime)
python wenet/bin/export_jit.py \
    --config exp/conformer/train.yaml \
    --checkpoint exp/conformer/avg_30.pt \
    --output_file exp/conformer/final.zip
```

### WeNet 9 — Vietnamese fine-tuning recipe

Cho team Vi muốn fine-tune WeNet trên Vi data:

```bash
# 1. Prepare Vi data (e.g., from VLSP, VIVOS)
cd examples/aishell/s0  # use as template
mkdir -p data/vi_train

# Generate wav.scp, text, etc.
python scripts/prepare_vi_data.py \
    --audio_dir /path/to/vi_audio \
    --transcription_file /path/to/transcripts.txt \
    --output_dir data/vi_train

# 2. Generate dictionary (Vi tokens)
cat data/vi_train/text | python tools/text2token.py \
    --space "<space>" \
    --bpe_model bpe.model \
    > data/dict/vi_lang_char.txt

# 3. Modify config for Vi
cp conf/train_unified_conformer.yaml conf/train_vi.yaml
# Edit:
#   output_dim: <vocabulary_size>
#   speed_perturb: true (helpful for Vi tones)

# 4. Initialize từ pretrained Whisper or multilingual Wav2Vec
# (Optional: download pretrained checkpoint, load as init)

# 5. Train với multi-GPU
torchrun --nproc_per_node=2 \
    wenet/bin/train.py \
    --config conf/train_vi.yaml \
    --data_type raw \
    --train_data data/vi_train/data.list \
    --cv_data data/vi_dev/data.list \
    --model_dir exp/vi_conformer \
    --init_checkpoint /path/to/pretrained.pt
```

### WeNet 10 — Common pitfalls

1. **OOM during training**: reduce `max_frames_in_batch` từ 12000 xuống 8000 hoặc 6000.
2. **Slow convergence**: ensure `speed_perturb: true` (3-way augmentation).
3. **Streaming accuracy worse than offline**: tune `num_left_chunks` (more context = better accuracy, more latency).
4. **Export to ONNX fails**: ensure `chunk_size` divisible by subsampling factor (4).
5. **Vietnamese tone confusion**: use character-level dictionary, not BPE merged tokens that span tones.

---

## Tóm tắt

1. **TensorRT + Triton**: Gold standard cho NVIDIA GPU deployment, 3-5x faster than naive PyTorch.
2. **ONNX Runtime**: Cross-platform, dễ dùng, hỗ trợ nhiều backends.
3. **CTranslate2/faster-whisper**: Tối ưu cho Whisper, 3-4x faster, 8x less memory.
4. **vLLM**: Cho Speech LLM serving (Qwen2-Audio, Moshi, etc.).
5. **OpenVINO**: Intel CPU optimization.
6. **BentoML/Ray Serve**: High-level serving frameworks cho complex pipelines.
7. **WeNet**: End-to-end framework cho ASR training + production deployment, đặc biệt streaming.
8. **Production speech pipeline cần multiple engines** cho different components.
9. **Data preprocessing/augmentation libraries** (torchaudio, librosa, audiomentations) là essential cho training quality.
10. **Quantization** (INT8, AWQ, GPTQ) là key cho cost-effective deployment.



---

<!-- References (auto-generated from .bib) -->
[^nvidia2024tensorrt]: {NVIDIA}, "TensorRT: High-Performance Deep Learning Inference Optimizer and Runtime", NVIDIA Developer Documentation
[^nvidia2024triton]: {NVIDIA}, "Triton Inference Server: An Open-Source Inference Serving Software", NVIDIA Developer Documentation
[^onnxruntime2024]: {Microsoft}, "ONNX Runtime: Cross-Platform, High Performance ML Inferencing and Training Accelerator", Microsoft Open Source
[^klein2020opennmt]: Klein, Guillaume and Kim, Yoon and others, "OpenNMT: Neural Machine Translation Toolkit", Association for Machine Translation in the Americas
[^faster_whisper]: Klein, Guillaume and others, "faster-whisper: Faster Whisper Transcription with CTranslate2", GitHub repository
[^bain2023whisperx]: Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew, "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio", Interspeech
[^kwon2023efficient]: Kwon, Woosuk and Li, Zhuohan and others, "Efficient Memory Management for Large Language Model Serving with PagedAttention", Symposium on Operating Systems Principles
[^openvino2024]: {Intel}, "OpenVINO Toolkit", Intel Developer Documentation
[^bentoml2024]: {BentoML}, "BentoML: Unified Model Serving Framework", BentoML Documentation
[^moritz2018ray]: Moritz, Philipp and Nishihara, Robert and others, "Ray: A Distributed Framework for Emerging AI Applications", Symposium on Operating Systems Design and Implementation
