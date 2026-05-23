# Inference Engines

## Tổng quan

Chương này trình bày chi tiết các inference engines phổ biến nhất để **deploy speech processing models** trong production. Từ **TensorRT + Triton** (NVIDIA stack) đến **ONNX Runtime**, **CTranslate2**, và các alternatives khác.

!!! note "Tại sao cần Inference Engine riêng?"
    PyTorch inference **chậm** và **tốn memory** cho production:

    - Không có graph optimization
    - Dynamic shapes gây overhead
    - Không tận dụng hardware-specific optimizations (TensorCores, INT8)

    Inference engines tối ưu: **2-10x faster**, **50-70% less memory** so với raw PyTorch.


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

!!! warning "GPU Memory Budget"
    Trên 1 GPU A100 (80GB), có thể serve:

    - **Whisper large-v3** (FP16): ~3GB → batch_size=16 dễ dàng
    - **Qwen2-Audio-7B** (FP16): ~14GB → batch_size=4
    - **SeamlessM4T-v2-large**: ~10GB → batch_size=8
    - Remaining memory cho KV-cache và dynamic batching


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

## Tóm tắt

1. **TensorRT + Triton**: Gold standard cho NVIDIA GPU deployment - 3-5x faster
2. **ONNX Runtime**: Cross-platform, dễ dùng, hỗ trợ nhiều backends
3. **CTranslate2/faster-whisper**: Tối ưu cho Whisper - 3-4x faster, 8x less memory
4. **vLLM**: Cho Speech LLM serving (Qwen2-Audio, etc.)
5. **OpenVINO**: Intel CPU optimization
6. **BentoML/Ray Serve**: High-level serving frameworks cho complex pipelines
7. Production speech pipeline cần **multiple engines** cho different components



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
