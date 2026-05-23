# Phụ lục E: Vietnamese Speech Resources

## Mục đích

Phụ lục E tập hợp các tài nguyên Speech AI cho tiếng Việt một cách có hệ thống, để người đọc có thể nhanh chóng tra cứu dataset, model, framework, API thương mại, và benchmark cộng đồng. Đây là reference material; nội dung phân tích chi tiết về tiếng Việt nằm trong Chương 16 (Vietnamese Speech Processing) và Chương 17 (Vietnamese Datasets và Benchmarks).

Mỗi mục được note theo các trường: tên, mô tả ngắn, link công khai, license, ghi chú thực dụng.

## Tài nguyên Tiếng Việt cho Speech AI

Phụ lục này tổng hợp toàn bộ tài nguyên (datasets, models, tools, APIs) cho speech processing tiếng Việt.

## A. Datasets

### ASR Datasets

| Dataset | Giờ | License | Link |
|---------|-----|---------|------|
| VLSP 2020 ASR | 100h | Research | vlsp.org.vn |
| VLSP 2021 ASR | 100h | Research | vlsp.org.vn |
| CommonVoice vi | ~50h | CC-0 | commonvoice.mozilla.org |
| VIVOS | 15h | CC-BY-SA | ailab.hcmus.edu.vn |
| VietSuperSpeech | 12h | MIT | github.com/nickviet/VietSuperSpeech |
| FPT OpenSpeech | 25h | Research | fpt.ai |
| BKAI-NAVER | 100h | Research | bkai.ai |

### TTS Datasets

| Dataset | Giờ | License | Chất lượng |
|---------|-----|---------|-----------|
| VietSuperSpeech | 12h | MIT | Studio 24-bit 48kHz |
| InfoRe TTS | 20h | Research | Studio |
| VNTTS | 10h | Research | Studio |

### NLP Datasets (hỗ trợ Speech)

| Dataset | Mô tả | Link |
|---------|-------|------|
| PhoBERT corpus | 20GB Vietnamese text | github.com/VinAIResearch/PhoBERT |
| VnCoreNLP | Word segmentation, POS | github.com/vncorenlp/VnCoreNLP |
| ViNewsQA | Question answering | Vietnamese QA dataset |

## B. Pretrained Models

### ASR Models

| Model | HuggingFace ID | Size | WER |
|-------|---------------|------|-----|
| PhoWhisper-small | vinai/PhoWhisper-small | 244M | ~12% |
| PhoWhisper-medium | vinai/PhoWhisper-medium | 769M | ~9% |
| PhoWhisper-large | vinai/PhoWhisper-large | 1.5B | ~7% |
| Wav2Vec2-vi | nguyenvulebinh/wav2vec2-base-vi | 95M | ~15% |
| XLSR-Vietnamese | Various community | 317M | ~12% |

### TTS Models

| Model | Source | Kiến trúc |
|-------|--------|----------|
| VietTTS | github.com/NTT123/vietTTS | VITS |
| Coqui TTS (vi) | Community fine-tune | VITS/Tacotron 2 |

## C. APIs & Services

| Service | ASR | TTS | Pricing | Docs |
|---------|-----|-----|---------|------|
| FPT.AI | Có | Có | Freemium | fpt.ai/speech |
| Zalo AI | Có | Có | Freemium | zalo.ai |
| Viettel AI | Có | Có | Enterprise | viettelai.vn |
| Google Cloud STT | Có | Có | Pay-per-use | cloud.google.com |
| Azure Speech | Có | Có | Pay-per-use | azure.microsoft.com |

## D. Tools & Libraries

| Tool | Mục đích | Vietnamese Support |
|------|----------|-------------------|
| VnCoreNLP | Word segmentation | Native |
| Underthesea | NLP toolkit | Native |
| PhoNLP | Multi-task NLP | Native |
| VietTTS | Text-to-Speech | Native |
| faster-whisper | ASR inference | Via PhoWhisper |

## E. Competitions & Workshops

| Event | Tổ chức | Tần suất | Topics |
|-------|---------|----------|--------|
| VLSP | Vietnamese NLP/Speech community | Hàng năm | ASR, TTS, NLP |
| Zalo AI Challenge | VNG/Zalo | Hàng năm | Various AI tasks |
| FPT AI Challenge | FPT | Không định kỳ | Speech, NLP |

## F. Research Groups

| Nhóm | Affiliation | Focus |
|------|------------|-------|
| VinAI Research | Vingroup | ASR, NLP, Computer Vision |
| Zalo AI Lab | VNG Corporation | Speech, NLP, Recommendation |
| FPT.AI Research | FPT Corporation | Full-stack AI |
| BKAI Lab | Hanoi University of Science and Technology | Speech, NLP |
| AILAB | VNUHCM University of Science | Speech, NLP |
| VietAI | Non-profit | Education, Open-source |
| NTT123 (VietTTS) | Independent | TTS |
