# Chương 17: Vietnamese Datasets và Benchmarks

## Vì sao chương này quan trọng

Speech AI cho tiếng Việt phụ thuộc vào data: data lớn, đa dạng, có labeling chất lượng. Chương này tập hợp các dataset, benchmark, và resource công khai cho ASR, TTS, voice agent tiếng Việt, kèm đánh giá thực dụng về điểm mạnh, hạn chế, và mức độ phù hợp với từng use case.

Đối với team product làm voice AI tại Việt Nam, chương này trả lời các câu hỏi cụ thể:

- Có những dataset ASR tiếng Việt nào public? VLSP 2020, VIVOS, Common Voice Vietnamese, FPT data có gì khác nhau? Tổng số giờ bao nhiêu?
- Benchmark nào được dùng để compare model? Hiệu suất SOTA hiện tại là bao nhiêu?
- Có dataset cho code-switching Việt-Anh không? Cho emotion? Cho speaker recognition?
- Dataset thương mại có gì? Giá bao nhiêu? License ra sao?

> **Cấu trúc chương**
>
> - **Phần 1**: ASR datasets (VLSP, VIVOS, Common Voice Vi, FPT, VinAI private).
> - **Phần 2**: TTS datasets và voice corpora.
> - **Phần 3**: benchmark VLSP và các benchmark cộng đồng.
> - **Phần 4**: PhoWhisper case study, kết quả công khai.
> - **Phần 5**: gaps và opportunities (code-switching benchmark, emotion, speaker for Vi).

## Tổng quan

Chương này tổng hợp toàn bộ **datasets**, **benchmarks**, **models**, và **hệ sinh thái** speech processing tại Việt Nam. Mục tiêu: cung cấp bản đồ đầy đủ để data scientists biết có gì available và cần gì để xây dựng hệ thống speech AI cho tiếng Việt.

> **📝 Thách thức riêng của Tiếng Việt**
>
> - **6 thanh điệu** (sắc, huyền, hỏi, ngã, nặng, ngang) - tonal language
> - **3 phương ngữ chính** (Bắc, Trung, Nam) với khác biệt lớn về phát âm
> - **Thiếu dữ liệu labeled** so với English, Mandarin
> - **Code-switching** phổ biến (Vietnamese + English)



## Training Datasets

### ASR Datasets

| Dataset | Giờ | Speakers | Phương ngữ | License | Nguồn |
|---------|-----|----------|------------|---------|-------|
| VLSP 2020 ASR | 100h | ~200 | Bắc, Nam | Research | VLSP |
| VLSP 2021 ASR | 100h | ~300 | 3 vùng | Research | VLSP |
| CommonVoice (vi) | ~50h | Community | Mixed | CC-0 | Mozilla |
| VietSuperSpeech | 12h | 1 (studio) | Bắc | MIT | VietTTS |
| FPT OpenSpeech | 25h | Multiple | Bắc | Research | FPT |
| VIVOS | 15h | 65 | Bắc, Nam | CC-BY-SA | AILAB |
| InfoRe (Bài đọc) | 25h | Multiple | Bắc | Research | InfoRe |
| Vinbigdata YTTD | 100h | Multiple | 3 vùng | Competition | Vinbigdata |
| LSVSC | ~100h | ~1000 | 3 vùng | Research | HUST |
| BKAI-NAVER | 100h | Multiple | 3 vùng | Research | BKAI |

: Vietnamese ASR Datasets <a id="tbl-vi-asr-datasets"></a>

### TTS Datasets

| Dataset | Giờ | Speakers | Chất lượng | License |
|---------|-----|----------|-----------|---------|
| VietSuperSpeech | 12h | 1 female | Studio (24-bit, 48kHz) | MIT |
| InfoRe TTS | 20h | 1 female | Studio | Research |
| VNTTS | 10h | 1 male | Studio | Research |
| LJSpeech-vi (community) | ~5h | 1 | Mixed | Various |

: Vietnamese TTS Datasets <a id="tbl-vi-tts-datasets"></a>

### Tổng kết Data Landscape

<a id="eq-vi-total-data"></a>

$$
\text{Total Vietnamese labeled speech} \approx 500\text{-}700 \text{ giờ (public)}
$$

> **⚠️ So sánh với English**
>
> English có hàng trăm nghìn giờ labeled data (LibriSpeech: 960h, GigaSpeech: 10,000h, SPGISpeech: 5,000h). Vietnamese vẫn còn **thiếu nghiêm trọng** cả về lượng và đa dạng.



## Benchmarks

### VLSP (Vietnamese Language and Speech Processing)

VLSP là workshop/competition thường niên từ 2013:

| Năm | Task | Dataset | Metric | Best Result |
|-----|------|---------|--------|-------------|
| 2020 | ASR | 100h broadcast | WER | 6.5% |
| 2021 | ASR | 100h mixed | WER | 8.2% |
| 2023 | ASR + TTS | Multi-domain | WER/MOS | ~7% / 4.1 |
| 2025 | ASR Streaming | Real-time | WER + Latency | Ongoing |

: VLSP Competition Results <a id="tbl-vlsp-benchmarks"></a>

### Các Benchmark Khác

| Benchmark | Tasks | Languages | Vietnamese? |
|-----------|-------|-----------|------------|
| FLEURS | ASR, LID | 102 | Có |
| CommonVoice | ASR | 100+ | Có (~50h) |
| ML-SUPERB | SSL evaluation | 143 | Có |
| Whisper benchmark | ASR | 99 | Có |
| SeamlessM4T | Translation | 100 | Có |

: International Benchmarks có Vietnamese <a id="tbl-international-benchmarks-vi"></a>

## Vietnamese Speech Models

### ASR Models

| Model | Tổ chức | Base | WER (VLSP) | Open-source |
|-------|---------|------|------------|-------------|
| PhoWhisper | VinAI | Whisper | ~7% | Có (HuggingFace) |
| Wav2Vec2-vi | VinAI | Wav2Vec 2.0 | ~10% | Có |
| XLSR-Vietnamese | Community | XLSR-53 | ~12% | Có |
| Viettel ASR | Viettel | Conformer | ~8% | Không (API) |
| Zalo ASR | Zalo/VNG | Conformer | ~9% | Không (API) |
| FPT.AI ASR | FPT | Transformer | ~10% | Không (API) |
| NVIDIA Canary-vi | NVIDIA | Canary | TBD | Có (NeMo) |

: Vietnamese ASR Models <a id="tbl-vi-asr-models"></a>

### PhoWhisper (VinAI)

PhoWhisper [^nguyen2024phowhisper] fine-tune Whisper cho tiếng Việt:

- Base: Whisper-large-v3
- Fine-tune data: ~500h Vietnamese
- Hỗ trợ 3 phương ngữ
- Available trên HuggingFace: `vinai/PhoWhisper-large`

```python
#| eval: false
#| code-fold: true
#| code-summary: "PhoWhisper Inference"
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torch import Tensor

# Load PhoWhisper
processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-large")
model = WhisperForConditionalGeneration.from_pretrained(
    "vinai/PhoWhisper-large"
)
model.eval()

def transcribe_vietnamese(
    audio: Tensor,           # [T] - float32, 16kHz mono
    sampling_rate: int = 16000,
) -> str:
    """Transcribe Vietnamese audio using PhoWhisper."""
    input_features: Tensor = processor(
        audio.numpy(),
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features  # [1, 80, 3000] - float32

    with torch.no_grad():
        predicted_ids: Tensor = model.generate(
            input_features,
            language="vi",
            task="transcribe",
        )  # [1, S] - int64

    transcription: str = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return transcription
```

### TTS Models

| Model | Tổ chức | Kiến trúc | MOS | Open-source |
|-------|---------|----------|-----|-------------|
| VietTTS | Community | VITS | ~3.8 | Có |
| Zalo TTS | Zalo/VNG | FastSpeech 2 | ~4.0 | Không (API) |
| FPT.AI TTS | FPT | Tacotron 2 | ~3.9 | Không (API) |
| Viettel TTS | Viettel | VITS2 | ~4.1 | Không (API) |
| CosyVoice-vi | Alibaba | CosyVoice | TBD | Có (fine-tune) |

: Vietnamese TTS Models <a id="tbl-vi-tts-models"></a>

## Voice Conversion (VC)

Voice conversion - biến đổi giọng nói người A thành giọng người B mà giữ nguyên nội dung:

<a id="eq-voice-conversion"></a>

$$
\hat{\mathbf{x}}_B = G(\mathbf{x}_A, \mathbf{s}_B)
$$

trong đó $G$ là generator, $\mathbf{s}_B$ là speaker embedding target.

### Các Approach

| Approach | Cần parallel data? | Chất lượng | Ví dụ |
|----------|-------------------|-----------|-------|
| Parallel VC | Có | Cao | GMM mapping |
| Non-parallel VC | Không | Cao | AutoVC, VITS-VC |
| Zero-shot VC | Không (chỉ cần 1 sample) | Trung bình-Cao | FreeVC, YourTTS |
| Any-to-Any VC | Không | Cao | VALL-E based |

: Voice Conversion approaches <a id="tbl-vc-approaches"></a>

## Các Tổ chức Nghiên cứu Speech tại Việt Nam

| Tổ chức | Focus | Sản phẩm/Output |
|---------|-------|-----------------|
| VinAI Research | ASR, NLP | PhoWhisper, PhoGPT, PhoBERT |
| Zalo AI (VNG) | ASR, TTS, NLP | Zalo AI APIs, competition |
| FPT.AI | Full-stack AI | FPT.AI Speech platform |
| Viettel AI | Telecom AI | Call center ASR/TTS |
| BKAI (HUST) | Academic research | Datasets, papers |
| AILAB (VNUHCM) | Academic research | VIVOS dataset |
| VietAI | Community | Courses, open-source |

: Hệ sinh thái Speech AI tại Việt Nam <a id="tbl-vi-ecosystem"></a>

## Thách thức và Hướng phát triển

### Thách thức

1. **Data scarcity**: Cần thêm hàng nghìn giờ labeled data
2. **Dialect diversity**: Model cần handle 3 phương ngữ + các accent địa phương
3. **Code-switching**: Vietnamese-English switching rất phổ biến
4. **Tonal accuracy**: 6 thanh điệu cần precision cao trong cả ASR và TTS
5. **Domain-specific**: Y tế, pháp lý, tài chính cần specialized models

### Hướng phát triển

1. **Self-supervised pretraining** trên Vietnamese audio (giảm dependency on labeled data)
2. **Multilingual transfer** từ models lớn (Whisper, SeamlessM4T)
3. **Synthetic data** generation bằng TTS → ASR augmentation
4. **Community datasets**: Mô hình CommonVoice cho Vietnamese
5. **Benchmark standardization**: VLSP cần thống nhất evaluation protocol

## Tóm tắt

1. Vietnamese có khoảng **500-700 giờ** public labeled speech data
2. **PhoWhisper** (VinAI) là model open-source tốt nhất cho Vietnamese ASR
3. **VLSP** là benchmark competition chính, tổ chức hàng năm
4. Hệ sinh thái: VinAI (research), Zalo/FPT/Viettel (products), BKAI/AILAB (academic)
5. Thách thức lớn nhất: data scarcity, dialect diversity, tonal accuracy



---

<!-- References (auto-generated from .bib) -->
[^nguyen2024phowhisper]: Nguyen, Thanh-Nhi and Nguyen, Dat Quoc, "PhoWhisper: Fine-tuning Whisper for Vietnamese Automatic Speech Recognition", arXiv preprint
