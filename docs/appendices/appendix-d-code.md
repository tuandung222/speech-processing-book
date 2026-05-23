# Code Listings Index

## Overview

Tất cả code trong cuốn sách sử dụng:

- **Python ≥ 3.10** với 100% type hints
- **PyTorch ≥ 2.0** và **torchaudio**
- Inline tensor shape comments: `# [batch, seq, dim] - dtype`
- Code blocks có thể fold/unfold trong HTML output

## Part I: Foundations

### Chapter 1  -  From NLP to Speech

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 1.1 | Feature comparison (text vs audio) | `Tensor [1, T]`, `Tensor [80, T']` |

### Chapter 2  -  Audio Signal Fundamentals

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 2.1 | DFT implementation | `Tensor [N] → Tensor [N] (complex64)` |
| 2.2 | STFT with windowing | `Tensor [T] → Tensor [n_freq, n_frames] (complex64)` |
| 2.3 | Mel filterbank | `Tensor [n_mels, n_fft//2+1] - float32` |
| 2.4 | MFCC extraction | `Tensor [n_mfcc, T'] - float32` |
| 2.5 | SpecAugment | `Tensor [n_mels, T'] → Tensor [n_mels, T']` |
| 2.6 | Complete feature pipeline | `AudioFeatureExtractor` class |

## Part II: ASR

### Chapter 3  -  ASR Foundations

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 3.1 | CTC Model | `CTCModel(nn.Module)`  -  encoder + CTC head |
| 3.2 | CTC greedy decode | `Tensor [B, T, V] → list[list[int]]` |
| 3.3 | RNN-Transducer | `RNNTransducer(nn.Module)`  -  encoder + predictor + joint |

### Chapter 4  -  Self-Supervised Speech

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 4.1 | Gumbel Vector Quantizer | `GumbelVectorQuantizer(nn.Module)` |
| 4.2 | Wav2Vec 2.0 Conv Encoder | `Wav2Vec2ConvEncoder(nn.Module)` |
| 4.3 | Conformer Conv Module | `ConformerConvModule(nn.Module)` |
| 4.4 | Conformer Block | `ConformerBlock(nn.Module)` |
| 4.5 | SSL for ASR (fine-tuning) | `SSLForASR(nn.Module)` |

### Chapter 5  -  Whisper & Modern ASR

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 5.1 | Whisper Encoder | `WhisperEncoder(nn.Module)`  -  Conv + Transformer |
| 5.2 | Whisper Decoder | `WhisperDecoder(nn.Module)`  -  Causal Transformer |

## Part III: TTS & Audio Codecs

### Chapter 6  -  TTS Foundations & Vocoders

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 6.1 | Variance Predictor | `VariancePredictor(nn.Module)`  -  duration/pitch/energy |
| 6.2 | Length Regulator | `LengthRegulator(nn.Module)`  -  phoneme → mel expansion |
| 6.3 | HiFi-GAN ResBlock | `ResBlock(nn.Module)`  -  dilated convolutions |
| 6.4 | HiFi-GAN Generator | `HiFiGANGenerator(nn.Module)`  -  mel → waveform |

### Chapter 7  -  End-to-End TTS

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 7.1 | VITS Posterior Encoder | `PosteriorEncoder(nn.Module)`  -  audio → latent |
| 7.2 | Flow Matching Sampling | `flow_matching_sample()`  -  ODE Euler solver |

### Chapter 8  -  Audio Codecs

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 8.1 | Vector Quantize | `VectorQuantize(nn.Module)`  -  single codebook VQ |
| 8.2 | Residual VQ | `ResidualVQ(nn.Module)`  -  Q-layer RVQ |
| 8.3 | EnCodec Encoder | `EnCodecEncoder(nn.Module)`  -  strided conv encoder |

## Part IV: Speech LLMs & Vietnamese

### Chapter 9  -  Speech Language Models

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 9.1 | Dual-Stream Generator | `DualStreamGenerator(nn.Module)`  -  Moshi-style |

### Chapter 10  -  Vietnamese Speech Processing

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 10.1 | Whisper Vietnamese prep | `prepare_whisper_vietnamese_training()` |
| 10.2 | Vietnamese text normalizer | `normalize_vietnamese_text()` |

## Part V: Production

### Chapter 11  -  Production Speech Systems

| Listing | Description | Key Types |
|---------|-------------|-----------|
| 11.1 | VAD Segmenter | `VADSegmenter`  -  streaming voice detection |
| 11.2 | Dynamic Batcher | `DynamicBatcher`  -  batch ASR requests |

## Dependencies

```
# Core
torch>=2.0.0
torchaudio>=2.0.0

# Audio processing
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0

# Vietnamese NLP (Chapter 10)
underthesea>=6.0.0

# Inference optimization (Chapter 11)
# tensorrt-llm (optional, NVIDIA GPU required)
# triton-client (optional, for Triton server)
```

## Running the Code

Tất cả code listings được set `eval: false`  -  chúng là **reference implementations** minh họa architecture, không phải production-ready code. Để chạy:

```bash
# Setup environment
python -m venv speech-ai-env
source speech-ai-env/bin/activate
pip install torch torchaudio numpy scipy

# Copy code from any listing and run
python listing_example.py
```

!!! note "Code Convention"
    Mọi tensor operation đều có comment theo format:

    ```python
    result: Tensor = operation(input)  # [dim1, dim2, dim3] - dtype
    ```

    Ví dụ: `# [batch, seq_len, d_model] - float32` nghĩa là tensor shape `[B, L, D]` với dtype `torch.float32`.

