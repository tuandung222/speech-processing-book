# Training Frameworks

## Tổng quan

Chương này hướng dẫn chi tiết cách sử dụng các framework training speech models phổ biến nhất. Mục tiêu: **NLP/LLM data scientist đọc xong là biết dùng ngay** để train ASR, TTS, và các speech tasks khác.

> **📝 Chọn Framework nào?**
>
> Không có framework "tốt nhất" cho mọi trường hợp. Lựa chọn phụ thuộc vào:
>
> 1. **Task**: ASR, TTS, hay multi-task?
> 2. **Scale**: Research (1 GPU) hay production (multi-node)?
> 3. **Customization**: Cần modify architecture hay dùng off-the-shelf?
> 4. **Deployment**: Cần export ONNX/TensorRT hay chỉ cần training?



## So sánh Tổng quan

| Framework | Tổ chức | Tasks | Language | Production-ready | Learning Curve |
|-----------|---------|-------|----------|-----------------|----------------|
| **WeNet** | Seasalt AI | ASR | Python/C++ | Rất cao | Trung bình |
| **ESPnet** | CMU/JHU | ASR, TTS, ST, SE | Python | Trung bình | Cao |
| **SpeechBrain** | Mila | All speech | Python | Trung bình | Thấp |
| **NeMo** | NVIDIA | ASR, TTS, NLP | Python | Rất cao | Cao |
| **fairseq** | Meta | ASR, ST | Python | Trung bình | Cao |
| **k2/icefall** | Community | ASR | Python/C++ | Cao | Rất cao |
| **Coqui TTS** | Community | TTS | Python | Trung bình | Thấp |

: So sánh Speech Training Frameworks <a id="tbl-frameworks-comparison"></a>

## WeNet - Chi tiết <a id="sec-wenet"></a>

### Tổng quan WeNet

WeNet [^zhang2022wenet] là framework ASR production-grade từ Seasalt AI (trước đây Mobvoi):

- **Unified streaming/non-streaming** ASR trong 1 model
- **Dynamic chunk training** - train 1 model, deploy ở nhiều latency modes
- **C++ runtime** cho production deployment
- **ONNX/TensorRT export** built-in

### Cấu trúc Project

```
wenet/
├── wenet/
│   ├── transformer/     # Conformer, Transformer encoders
│   ├── transducer/      # RNN-T decoder
│   ├── utils/           # Data loading, checkpointing
│   └── bin/             # Training, inference scripts
├── examples/
│   ├── aishell/         # Chinese ASR recipe
│   ├── librispeech/     # English ASR recipe
│   └── multi_cn/        # Multi-dataset Chinese
├── runtime/
│   ├── core/            # C++ inference engine
│   ├── server/          # gRPC/HTTP server
│   └── onnxruntime/     # ONNX inference
└── tools/               # Kaldi-style tools
```

### Training Recipe (Step-by-step)

**Bước 1: Chuẩn bị Data**

WeNet sử dụng format đơn giản - mỗi line là JSON:

```json
{"key": "utt001", "wav": "/data/audio/001.wav", "txt": "xin chào"}
```

```python
#| eval: false
#| code-fold: true
#| code-summary: "Chuẩn bị data cho WeNet"
import json
from pathlib import Path
from typing import Dict, List

def prepare_wenet_data(
    wav_dir: Path,
    transcript_file: Path,
    output_file: Path,
) -> int:
    """Prepare data in WeNet format.

    Args:
        wav_dir: Directory containing .wav files
        transcript_file: TSV file with (utt_id, transcript)
        output_file: Output JSON-lines file

    Returns:
        Number of utterances processed
    """
    # Read transcripts
    transcripts: Dict[str, str] = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            parts: List[str] = line.strip().split("\t")
            if len(parts) >= 2:
                transcripts[parts[0]] = parts[1]

    # Write WeNet format
    count: int = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for utt_id, text in sorted(transcripts.items()):
            wav_path: Path = wav_dir / f"{utt_id}.wav"
            if wav_path.exists():
                entry: Dict[str, str] = {
                    "key": utt_id,
                    "wav": str(wav_path),
                    "txt": text,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    return count
```

**Bước 2: Config YAML**

WeNet sử dụng YAML config cho toàn bộ training pipeline. Ví dụ cho Conformer CTC/Attention:

```yaml
# conf/train_conformer.yaml
encoder: conformer
encoder_conf:
    output_size: 256
    attention_heads: 4
    linear_units: 2048
    num_blocks: 12
    input_layer: conv2d
    cnn_module_kernel: 15

decoder: transformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 6

model_conf:
    ctc_weight: 0.3      # CTC loss weight (0.3 CTC + 0.7 Attention)
    lsm_weight: 0.1      # Label smoothing
    length_normalized_loss: false

optim: adam
optim_conf:
    lr: 0.001
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 25000
max_epoch: 100
accum_grad: 4
```

**Bước 3: Training Commands**

```bash
# Single GPU training
python wenet/bin/train.py \
    --config conf/train_conformer.yaml \
    --data_type raw \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/conformer \
    --num_workers 4

# Multi-GPU training (DDP)
torchrun --nproc_per_node=4 wenet/bin/train.py \
    --config conf/train_conformer.yaml \
    --data_type raw \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/conformer
```

**Bước 4: Export và Inference**

```bash
# Export to ONNX
python wenet/bin/export_onnx_cpu.py \
    --config exp/conformer/train.yaml \
    --checkpoint exp/conformer/final.pt \
    --output_onnx_dir exp/conformer/onnx

# C++ runtime inference
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path test.wav \
    --model_path exp/conformer/final.zip \
    --dict_path data/dict/units.txt
```

### Dynamic Chunk Training

Điểm mạnh nhất của WeNet - **1 model cho cả streaming và offline**:

```python
#| eval: false
#| code-fold: true
#| code-summary: "Dynamic Chunk Size Selection"
import random
from typing import List

def select_chunk_size(
    training: bool = True,
    static_chunk_size: int = -1,  # -1 = dynamic
) -> int:
    """Select chunk size for training/inference.

    During training: randomly select chunk size.
    During inference: use static_chunk_size.

    Returns:
        chunk_size: number of frames per chunk.
                    -1 means full utterance (offline mode).
    """
    if not training:
        return static_chunk_size

    # Dynamic chunk training: random selection
    chunk_sizes: List[int] = [
        4, 8, 16, 25,   # streaming modes (various latencies)
        -1,              # offline mode (full attention)
    ]
    return random.choice(chunk_sizes)

# Inference modes:
# chunk_size = -1  -> offline (best accuracy)
# chunk_size = 16  -> streaming, ~320ms latency
# chunk_size = 4   -> streaming, ~80ms latency (lowest quality)
```

## ESPnet

### Tổng quan ESPnet

ESPnet [^watanabe2018espnet] là framework nghiên cứu toàn diện nhất:

- Hỗ trợ **ASR, TTS, ST, SE, SLU, codec, SSL** trong 1 framework
- **200+ recipes** cho các datasets phổ biến
- Tích hợp **HuggingFace** để dễ dàng share models
- **E-Branchformer** là default encoder

### Training ASR với ESPnet

```bash
# ESPnet2 recipe structure
cd egs2/librispeech/asr1
./run.sh --stage 1 --stop_stage 13 \
    --asr_config conf/tuning/train_asr_e_branchformer.yaml \
    --inference_config conf/decode_asr.yaml \
    --ngpu 4

# Stages:
# 1-4: Data preparation
# 5-7: LM training (optional)
# 8-9: Tokenizer (BPE/char)
# 10-11: ASR training
# 12-13: Decoding + scoring
```

### So sánh WeNet vs ESPnet

| Tiêu chí | WeNet | ESPnet |
|----------|-------|--------|
| Focus | Production ASR | Research (multi-task) |
| Streaming | Built-in (dynamic chunk) | Add-on |
| C++ Runtime | Có | Không |
| ONNX Export | Built-in | Community |
| TTS Support | Không | Có (VITS, etc.) |
| Recipes | ~20 | **200+** |
| Default Encoder | Conformer | E-Branchformer |
| Learning Curve | Trung bình | Cao |

: WeNet vs ESPnet <a id="tbl-wenet-vs-espnet"></a>

## SpeechBrain

### Tổng quan

SpeechBrain [^ravanelli2021speechbrain] từ Mila - thiết kế cho **dễ sử dụng**:

- **HyperPyYAML** config system (linh hoạt hơn YAML thông thường)
- Hỗ trợ **tất cả speech tasks** trong 1 framework
- **HuggingFace integration** sâu
- Rất tốt cho **prototyping nhanh**

### Training ASR với SpeechBrain

```python
#| eval: false
#| code-fold: true
#| code-summary: "SpeechBrain ASR Brain Class"
import torch
from torch import Tensor

# SpeechBrain uses a Brain class pattern:
#
# class ASRBrain(sb.Brain):
#     def compute_forward(self, batch, stage):
#         wavs = batch.sig           # [B, T] - float32
#         wav_lens = batch.sig_lengths  # [B] - float32 (relative lengths)
#         feats = self.hparams.compute_features(wavs)  # [B, T', F]
#         enc_out = self.hparams.encoder(feats)         # [B, T', D]
#         logits = self.hparams.ctc_head(enc_out)       # [B, T', V]
#         return {"logits": logits}
#
#     def compute_objectives(self, predictions, batch, stage):
#         return self.hparams.ctc_cost(
#             predictions["logits"],
#             batch.tokens,
#             predictions["wav_lens"],
#             batch.tokens_lengths,
#         )

# Example: inference with pretrained model
# import speechbrain as sb
# asr = sb.pretrained.EncoderDecoderASR.from_hparams(
#     source="speechbrain/asr-conformer-librispeech",
#     savedir="pretrained_models/asr-conformer",
# )
# transcription = asr.transcribe_file("audio.wav")
```

## NVIDIA NeMo

### Tổng quan

NeMo [^kuchaiev2019nemo] từ NVIDIA - framework production-grade:

- **ASR**: Conformer, FastConformer, Canary
- **TTS**: FastPitch, HiFi-GAN, VITS
- **NLP/LLM**: GPT, BERT, Megatron
- **TensorRT/Triton** integration
- **Multi-node training** với Megatron parallelism

### Training ASR với NeMo

```python
#| eval: false
#| code-fold: true
#| code-summary: "NeMo ASR Training"
# import nemo.collections.asr as nemo_asr
# from omegaconf import OmegaConf
# import pytorch_lightning as pl

# Option 1: Fine-tune pretrained model
# model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
#     "nvidia/stt_en_fastconformer_ctc_large"
# )

# Option 2: Train from scratch with config
# config = OmegaConf.load("conf/fastconformer_ctc.yaml")
# model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model)

# Setup trainer
# trainer = pl.Trainer(
#     devices=4,
#     accelerator="gpu",
#     strategy="ddp",
#     max_epochs=100,
#     accumulate_grad_batches=4,
#     precision="bf16-mixed",
# )
# trainer.fit(model)

# Export to ONNX
# model.export("model.onnx")
```

### NeMo cho Vietnamese

```bash
# Fine-tune Canary cho Vietnamese
python examples/asr/asr_finetune.py \
    --config-path=conf/canary \
    --config-name=canary_1b_finetune \
    model.train_ds.manifest_filepath=data/vi_train.json \
    model.validation_ds.manifest_filepath=data/vi_dev.json \
    trainer.devices=4 \
    trainer.max_epochs=50
```

## k2/icefall

### Tổng quan

k2 [^povey2021k2] và icefall là framework từ Daniel Povey (tác giả Kaldi):

- **Differentiable FST** operations (k2 library)
- **Zipformer** encoder (SOTA efficiency)
- **Pruned RNN-T** loss (memory efficient)
- **Rất nhanh** training và inference

### Đặc điểm nổi bật

| Feature | k2/icefall |
|---------|-----------|
| FST Operations | CUDA-accelerated, differentiable |
| Loss Functions | CTC, RNN-T, lattice-free MMI |
| Encoder | Zipformer (multi-scale) |
| Data Loading | Lhotse (lazy, scalable) |
| Pruned RNN-T | Chỉ compute loss trên relevant (t, u) pairs |

: k2/icefall features <a id="tbl-k2-features"></a>

## fairseq / fairseq2

fairseq [^ott2019fairseq] từ Meta:

- **Wav2Vec 2.0, HuBERT, data2vec** pretraining
- **SeamlessM4T** speech translation
- **XLS-R / XLSR** multilingual SSL

> **⚠️ fairseq maintenance**
>
> fairseq đang chuyển sang **fairseq2** và một số components sang HuggingFace. Cho research mới, nên dùng HuggingFace Transformers hoặc ESPnet thay thế.



## So sánh với NLP Training Frameworks

| Tiêu chí | HF Trainer | Lightning | WeNet | ESPnet | NeMo |
|----------|-----------|-----------|-------|--------|------|
| Speech-specific | Không | Không | ASR | All | All |
| Data pipeline | datasets | Manual | Built-in | Built-in | Built-in |
| Streaming ASR | Không | Không | **Có** | Có | Có |
| C++ Runtime | Không | Không | **Có** | Không | Có (Triton) |
| Multi-GPU | Có | Có | DDP | DDP | **Megatron** |
| Config system | TrainingArgs | Manual | YAML | YAML | OmegaConf |
| Pre-built recipes | Không | Không | ~20 | **200+** | ~50 |

: Speech Frameworks vs NLP Frameworks <a id="tbl-speech-vs-nlp-frameworks"></a>

> **📝 Khi nào dùng HF Trainer cho Speech?**
>
> HuggingFace Trainer phù hợp khi:
>
> 1. Fine-tune **pretrained models** (Whisper, Wav2Vec 2.0)
> 2. **Prototype nhanh** với ít code
> 3. Cần **HuggingFace Hub** integration
>
> Nhưng KHÔNG phù hợp khi cần **streaming ASR**, **CTC/RNN-T decoding** tối ưu, hay **production C++ runtime**.



## Recommendation

| Mục đích | Framework đề xuất |
|----------|-------------------|
| Production ASR (streaming) | **WeNet** hoặc **NeMo** |
| Research (multi-task) | **ESPnet** |
| Quick prototyping | **SpeechBrain** hoặc **HF Trainer** |
| Vietnamese ASR | **WeNet** hoặc **NeMo** (Canary fine-tune) |
| TTS | **ESPnet** hoặc **Coqui TTS** |
| SSL pretraining | **fairseq** hoặc **HF Trainer** |
| Maximum throughput | **k2/icefall** (Zipformer + pruned RNN-T) |

: Framework recommendations <a id="tbl-framework-recommendations"></a>

## Tóm tắt

1. **WeNet**: Best cho production streaming ASR - C++ runtime, ONNX export, dynamic chunk
2. **ESPnet**: Best cho research - 200+ recipes, multi-task, E-Branchformer
3. **SpeechBrain**: Best cho beginners - dễ học, HuggingFace integration
4. **NeMo**: Best cho NVIDIA stack - FastConformer, TensorRT, multi-node
5. **k2/icefall**: Best performance - Zipformer, pruned RNN-T
6. **HF Trainer**: Phù hợp fine-tune pretrained models, không cho from-scratch training



---

<!-- References (auto-generated from .bib) -->
[^zhang2022wenet]: Zhang, Binbin and Wu, Di and Peng, Zhendong and others, "WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit", Interspeech
[^watanabe2018espnet]: Watanabe, Shinji and Hori, Takaaki and others, "ESPnet: End-to-End Speech Processing Toolkit", Interspeech
[^ravanelli2021speechbrain]: Ravanelli, Mirco and Parcollet, Titouan and others, "SpeechBrain: A General-Purpose Speech Toolkit", arXiv preprint arXiv:2106.04624
[^kuchaiev2019nemo]: Kuchaiev, Oleksii and Li, Jason and others, "NeMo: A Toolkit for Building AI Applications Using Neural Modules", arXiv preprint arXiv:1909.09577
[^povey2021k2]: Povey, Daniel and others, "k2: FSA/FST Algorithms for PyTorch", GitHub repository
[^ott2019fairseq]: Ott, Myle and Edunov, Sergey and Baevski, Alexei and others, "fairseq: A Fast, Extensible Toolkit for Sequence Modeling", North American Chapter of the Association for Computational Linguistics: Demonstrations
